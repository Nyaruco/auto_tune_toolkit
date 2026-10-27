# AUTO TUNE — METAX 上的 SLO 自适应寻优

给每个场景自动找「满足 SLO 且吞吐最高」的 batch 与 request_rate，输出两张 CSV：**结论表**（每个「场景 × 模式」一行，对标用）和**全测点表**（每个真跑过的测试点一行，看趋势 / 找拐点）。

## 30 秒上手

```bash
cd /path/to/auto_tune_toolkit
python3 auto_tune.py --list
python3 auto_tune.py --scene dialog --mode full --host 127.0.0.1 --port 9292
python3 generate_report.py --all-modes --print-table
```

`full` 用场景默认 client（ais_bench 场景需 PATH 里有 `ais_bench`）；`donly` / `ponly` 还要配 bench_serving 路径，见[怎么跑](#怎么跑)。

也可以填好 `auto_tune.sh` 配置区后 `bash auto_tune.sh`；`MODE=donly` / `MODE=ponly` 切换模式，`RATE_SEARCH=1` 打开 rate 搜索，`NO_RATE_SEARCH=1` 让 full 只搜 batch。

## 按任务找入口

| 我想… | 看这里 |
| --- | --- |
| 第一次跑 / 断点续跑 / 只跑 D 或 P | [怎么跑](#怎么跑) |
| PD 分离怎么指向 | [服务端指向](#服务端指向指错会被逐条拒收) |
| 出 CSV / 看列含义 | [二、生成的 CSV](#二生成的-csv两张表) |
| 搜索为什么停 / 怎么调 | [四、搜索算法](#四搜索算法) |
| 换模型 / 加场景 / P95 | [八、扩展](#八跑新模型和加新场景改哪里) |
| 报错 / 结果不对 | [七、常见问题](#七常见问题) |
| 查全部参数 | [五、命令行参数](#五命令行参数) |

## 先统一几个说法

| 说法 | 含义 |
| --- | --- |
| batch | 并发数，对应客户端的 `--max-concurrency` |
| rate | 请求频率（req/s）；`inf` = 不限制，请求一次全发 |
| 打满点 | `rate=inf` 的测点 |
| 限速点 | rate 取有限值的测点 |
| SLO 余量 | `(红线 - 实测) / 红线`；30% 余量 = 只用掉 70% 时延预算 |
| 跑挂 | client 自身失败（`parse_ok=False`），不是 SLO 不达标 |

---

## 一、模式与跑法

| 模式 | 考核 | client | 输出长度 | 默认 rate | 吞吐口径 |
| --- | --- | --- | --- | --- | --- |
| `full`（默认） | TTFT **和** TPOT | 按场景配（缺省 `ais_bench`） | 场景原值 | 搜（两段） | Output Token |
| `donly` | **只看 TPOT** | `sglang.bench_serving` | 场景原值 | 打满，只搜 batch | Output Token |
| `ponly` | **只看 TTFT** | `sglang.bench_serving` | **压到 1** | 打满，只搜 batch | Input Token |
| `ponly_cached` | **只看 TTFT** | bench_serving + GSP | 场景或 `output_len` | 搜（两段） | Input Token |
| `full_cached` | TTFT **和** TPOT | bench_serving + GSP | 场景或 `output_len` | 搜（两段） | Output Token |

一句话选模式：端到端选 `full`；只压 D 选 `donly`；只压 P 选 `ponly`；共享前缀选 `*_cached`。

「搜（两段）」= 先定并发、再定 rate，见[第四节](#四搜索算法)；「打满」= `rate=inf`。

**「跑挂」和「不达标」是两回事：** 跑挂 = client 自己没跑起来（解释器找不到 / 服务不可达 / dataset 下载失败）。任一测点 `parse_ok=False` ⇒ 本场景本次整体作废，不写最优、`converged=False`；续跑时该点重测，报表仍出一行并注明「搜索未完成（未收敛）」，结论取已跑通点里吞吐最高者。

**搜不搜 request rate 由跑法决定：** `--no-rate-search`（仅 full）让 rate 打满、只搜 batch；`--rate-search`（仅 donly/ponly）也搜 rate。两者互斥，加在「本来就这样」一侧是空操作（不改 `search_rule`）；真改变时旧 state 重搜、已测点复用。开关只动 rate 一维，考核轴 / client / 输出长度不变（`--rate-search` 下 donly 仍 `--fake-prefill`，ponly 仍输出 1）；注意 `full --no-rate-search` ≠ `donly`（后者只看 TPOT、走 bench_serving）。详见[第四节](#四搜索算法)。

### cache 命中 prefill：`ponly_cached` / `full_cached`

GSP = `generated-shared-prefix`。这两个模式固定使用它，rate 默认参与搜索；实际命令行固定带：

```text
--dataset-name generated-shared-prefix --gsp-num-groups 1
--gsp-prompts-per-group <num_prompts>
--gsp-system-prompt-len <cached_input_len>
--gsp-question-len <non_cached_input_len>
--gsp-output-len <output_len> --gsp-range-ratio 1.0 --seed <每点唯一>
```

场景里增加这些字段；`input_len` 必须等于 `cached_input_len + non_cached_input_len`：

```python
make_scene(
    "cached_16k", "Cached 16K / 1K",
    input_len=16384, max_out_len=1024,
    cached_input_len=12288,       # 命中部分（system prompt）
    non_cached_input_len=4096,    # 未命中部分（question）
    output_len=1024,              # 可选；不填用 max_out_len
    slo_ttft=15.0, slo_tpot=-1, client=CLIENT_BENCH,
)
```

命中率 = `cached_input_len / (cached_input_len + non_cached_input_len)`；每个测点自动换 seed，避免相邻测试复用随机 prompt 导致全部命中。

```bash
MODE=ponly_cached   # 或 full_cached
python3 auto_tune.py --mode $MODE --scenes-file test_scenes_cached.py \
  --host 127.0.0.1 --port 9292 --bench-python python \
  --bench-model <model-path> --model-name <model-name>
```

### 怎么跑

首次跑 full 见[30 秒上手](#30-秒上手)。只搜 batch：

```bash
python3 auto_tune.py --scene dialog --mode full --no-rate-search --host 127.0.0.1 --port 9292
```

**跑 donly / ponly**（走 bench_serving，需要装 sglang 的解释器和模型目录）：

```bash
model_name="MyModel"
model_dir="/path/to/model"

common="--host 127.0.0.1 --port 9292 \
        --bench-python /path/to/env/bin/python \
        --bench-model $model_dir \
        --model-name $model_name \
        --state-dir runs/$model_name"

python3 auto_tune.py --mode donly $common        # 不指定 --scene 就全跑
python3 auto_tune.py --mode ponly $common
python3 auto_tune.py --mode ponly --rate-search $common   # 也搜 rate
```

**中断了**：直接重跑，已测点复用、已收敛场景跳过。想重搜：删对应 state，或加 `--restart`。

### 结果文件放在哪

各模式分文件存放，换模式跑不会覆盖上一种模式的数据：

| 模式 | 日志 | state | report |
| --- | --- | --- | --- |
| `full` | `logs/<场景>_b*_r*_<时间>.log` | `state_<场景>.json` | `report.json` |
| `donly` | `logs/<模型>/donly/` | `state_<场景>_donly.json` | `report_donly.json` |
| `ponly` | `logs/<模型>/ponly/` | `state_<场景>_ponly.json` | `report_ponly.json` |
| `ponly_cached` | `logs/<模型>/ponly_cached/` | `state_<场景>_ponly_cached.json` | `report_ponly_cached.json` |
| `full_cached` | `logs/<模型>/full_cached/` | `state_<场景>_full_cached.json` | `report_full_cached.json` |

state 规则：同一模式的两种跑法共用一个 state，靠 `search_rule` 区分；规则不同就重搜（已测点仍走缓存）。建议按模型分目录（`--state-dir runs/<模型>`）；同目录换模型，对不上的 state 会跳过并提示。

### donly / ponly 实际发出的命令

```bash
<python> -m sglang.bench_serving --model <模型目录> --backend sglang \
  --dataset-name random --num-prompts 320 --max-concurrency 64 --request-rate inf \
  --random-input-len 4096 --random-output-len 1024 --random-range-ratio 1 \
  --dataset-path <ShareGPT json> --host 127.0.0.1 --port 9292 --fake-prefill
# donly: --random-output-len=场景 max_out_len 且带 --fake-prefill；ponly: 输出 1、不带 --fake-prefill
```

加 `--rate-search` 只换 `--request-rate`（改成该段频率）和 `--num-prompts`（按[限速点口径](#每次发多少请求倍率与目标时长)折算）；并发、输出长度、`--fake-prefill` 照旧。

#### 服务端指向：指错会被逐条拒收

PD 分离下请求必须带 `bootstrap_host / port / room` 才会被 P / D 实例受理，而这三个字段由 **router** 补，客户端填不了。

| 跑法 | `--host/--port` 指向 | 服务端起法 | 原因 |
| --- | --- | --- | --- |
| `full` / `*_cached`（端到端） | **router** | 正常 PD 分离 | 房号由 router 补；cached 模式也没有 `--fake-prefill` |
| `donly` | **D 实例**（不经 router） | D 侧加 `--disaggregation-transfer-backend fake` | 靠 `--fake-prefill` 塞固定房号，上游限定只配合 fake backend |
| `ponly` | **独立单实例** | **关掉** PD 分离 | 客户端没有补房号的途径 |

所以 `donly` / `ponly` 不能对着同一个部署连着跑。漏了这条时 HTTP 仍 `200 OK`，服务端日志刷 `Invalid request: Disaggregated request received without bootstrap room id`，auto_tune 每个点都判不合格（`失败请求=N > 0`），很容易误读成机器扛不住。

**bench_serving 的三个路径**（都有缺省值）：

| 路径 | 参数 | 缺省 |
| --- | --- | --- |
| 装 sglang 的解释器 | `--bench-python`（或 `BENCH_PYTHON`） | `python3` |
| 模型目录（tokenizer） | `--bench-model` | `scenes.py` 的 `MODEL` |
| ShareGPT json | `--bench-dataset` | `scenes.py` 的 `BENCH_DATASET`（传 `""` 关掉） |

> **random 数据集为什么还要 `--dataset-path`**：缺了它这版 loader 会去 HuggingFace 下载，离线机器直接报错退出。传本机 json 只是阻止联网，请求仍由 `--random-*-len` 生成。

### full 用哪个 client

`full` 不强制 ais_bench：场景里写 `client="sglang.bench_serving"` 就用 bench_serving。临时整批切换用 `--client bench`；`donly/ponly` 恒用 bench_serving，`--client` 不生效。

| client | 请求来源 | 时延统计量 |
| --- | --- | --- |
| `ais_bench` | 场景 jsonl（真实长度分布） | 平均 / 中位 / P90 / P99 |
| `sglang.bench_serving` | 按长度定长生成 | 平均 / 中位 / P50 / P90 / P99 |

换 client 会让该场景 state 失效并重搜（每个测点记了打它的 client）。

### 改了参数，旧结论还能用吗

| 改了 | 记在哪 | 后果 |
| --- | --- | --- |
| SLO、`input_len`、输出长度、`dataset`、`--thru-tol`、`--fail-tolerant` | `state["scene_fp"]` | **只重判**：测点从 state 复用，按新判据重比（`--thru-tol` 的键名是 `thru_need`） |
| `--prompts-mult`、`--long-input-mult`、`--long-input-len` | `state["measure_fp"]` | **要重测**：旧数字是别的请求数下测的，缓存整体丢掉；只有真正跨档的场景作废 |
| `--rate-target-s` | 不进指纹 | 只影响限速点跑多久，不影响负载形状 |
| `--no-rate-search`、`--rate-search`、`--mode` | `state["search_rule"]` | 重搜（已测点仍缓存）：两种跑法的最优点不同 |

### 每次发多少请求：倍率与目标时长

| 测点 | num_prompts |
| --- | --- |
| 打满点（第一段所有点、`--no-rate-search` 的全部点） | `batch × 本场景倍率` |
| 限速点（第二段） | `min(batch × 倍率, max(rate × --rate-target-s, 100))`；下限 100 条是分位数判定的最小样本量 |

倍率按场景 `input_len` 分两档：

| 档 | 条件 | 倍率 |
| --- | --- | --- |
| 普通档 | `input_len ≤ --long-input-len`（缺省 16384） | `--prompts-mult`（缺省 5） |
| 长输入档 | `input_len > --long-input-len` | `--long-input-mult`（缺省 2） |

长输入单条请求慢，5 倍会让单点跑很久（60K、batch=7 时 35 条约 3.4 分钟），而分位数只看并发数够不够，2 倍就够。出表用**测点当时**的 `num_prompts`（逐点记在 state 里）。

---

## 二、生成的 CSV：两张表

```bash
python3 generate_report.py --all-modes                  # 结论表 + 全测点表（最常用）
python3 generate_report.py --all-points                 # 只要全测点表
python3 generate_report.py --all-modes --print-table    # 顺便打印人可读的表
python3 generate_report.py --all-modes --out 我的.csv --out-points 明细.csv
```

跑完任意一种模式会自动重新汇总（后跑的模式并入），也可随时手工重出。两张表的数据源不同：

| 表 | 读什么 | 分场景分次跑的后果 |
| --- | --- | --- |
| 全测点表 | **只读** `state_*.json` | 完整，所有跑过的场景都在 |
| 结论表 | 优先 `report_<mode>.json`，没有才回落 state | 只含**最近一次** report 里的场景 |

所以单场景重跑后，`report.json` 只剩该场景，结论表也跟着只剩一行（全测点表不受影响）。要全量结论表：不带 `--scene` 跑一次，或 `rm report*.json && python3 generate_report.py --all-modes` 从 state 汇总。

### 两张表长什么样

两张表都是单行表头，第一行就是列名。

| 表 | 列数 | 主要内容 |
| --- | --- | --- |
| 结论表 | 20（`--all-modes`）/ 19（单模式） | 模型 · 模式 · 测试用例 · 输入输出 · SLO · precision → 请求数 · 频率 · 并发 → 时延 → QPS · Output · Total |
| 全测点表 | 25 | 模式 · 模型 · 测试用例 · SLO · 判定统计量 · 频率 · 并发 · 请求数 · QPS · TTFT · TPOT · E2E · 三种吞吐 · 满足要求 · 最优 · 备注 |

结论表的判定列随口径增减，所以列数不固定。

#### 「测试用例」拆成三列

拆开是为了在 Excel 里能排序筛选。

| 列 | 内容 | 例 |
| --- | --- | --- |
| 测试用例 | 场景名（去掉尾部长度括号） | `对话场景` |
| 输入/输出 | 场景配置的输入 / 最大输出长度，1024 的整数倍写 `K` | `4K/1K` |
| SLO 要求 | 两条要求用「；」连接；不考核的轴换成说明文字 | `ttft_p90 < 2s；tpot_p90 < 35ms` |

长度按场景的 `input_len` / `max_out_len` 现算，不解析场景名；写的是**配置长度**，不是实际发的（ponly 输出会压到 1）。`slo_*_text` 是场景里写死的报表文本，换判定口径时要同步改。

#### 时延列怎么读

判定口径列打头（默认 P90），后面并列 Mean / P99；E2E 只有 Mean / P99（两个 client 都打不出 E2E P95，补丁只补了 TTFT/TPOT）。

- 结论表的判定列表头跟着 `--latency-stat` 变；口径本身就是 Mean/P99 时跳过重复列，所以列数不固定。全测点表的 Mean/P99 是固定列。
- TTFT / E2E 单位秒，TPOT 毫秒。
- 一条结果都没搜到时，表头仍按数据里的 `latency_stat`，不兜底 P90；时延格子留空（= 没测到）。`—` = 本模式不考核。

#### 本模式不考核的轴写 `—`

| 轴 | `full` | `donly` | `ponly` |
| --- | --- | --- | --- |
| TTFT | 数值 | `—`（`--fake-prefill` 跳过 prefill） | 数值 |
| TPOT | 数值 | 数值 | `—`（输出 1 token，无请求内 ITL） |
| E2E | 数值 | `—` | `—` |

`ponly_cached` / `full_cached` 按同一规则（前者同 ponly，后者同 full）。「SLO 要求」列对应那条也会写「（donly 模式不考核 TTFT）」，两处一致。单模式出表认 `--report` 文件顶层的 `mode` 字段（同时决定读哪个 state）；老 report 没这个字段时当 full。

---

## 三、判定统计量：`--latency-stat`

默认两轴都按 P90 判；可以整体换，也可以只覆盖某一轴。

```bash
python3 auto_tune.py --latency-stat p99  ...                   # 两轴都按 P99
python3 auto_tune.py --latency-stat mean ...                   # 两轴都按平均值
python3 auto_tune.py --latency-stat p90 --ttft-stat p99 ...    # TTFT 按 P99，TPOT 按 P90
```

| 参数 | 作用 |
| --- | --- |
| `--latency-stat STAT` | 同时管两轴，缺省 `p90`；可选 mean / median / p50 / p90 / p95 / p99 |
| `--ttft-stat STAT` / `--tpot-stat STAT` | 只覆盖对应轴 |

开跑前按**各场景实际 client** 逐个校验，选了该 client 打不出来的统计量会直接报错，不会静默回退。bench_serving 的 P95 由本目录 `bench_serving.py` 补出，需覆盖到压测机器，见[第八节 D](#d-以-p95-为标准)。

---

## 四、搜索算法

### 合格与停止条件

SLO 是硬门槛，**它的余量决定还要不要继续搜**；吞吐只用于择优和平台期判定。

- **合格** = TTFT / TPOT 满足 SLO，且失败请求数满足 `--fail-tolerant`。
- **SLO 余量门**：平台期看最优点的最小余量（考核轴取 min；donly/ponly 只看考核的那一轴）。余量 ≤ `SLO_FRONTIER_MARGIN`（30%，`search.py` 常量）⇒ 逼近 SLA，平台期正常停；> 30% ⇒ 继续放大并发。
- 30% 是噪声安全带：单点 TTFT 噪声 ±20~30%，实测余量 m 的最坏真实余量约 `1.3m − 0.3`，m = 30% 时仍有约 +9%。
- `--max-rounds` 是**保险丝**：余量一直大、平台期持续时，连续这么多档无实质吞吐提升就强制停，避免在容量墙上无限烧测试。`--thru-tol` 保留兼容，不再参与中止判断。

### 搜 rate 的跑法：先定并发，再定频率（两段）

适用范围：`full` / `full_cached`（默认），或加了 `--rate-search` 的 `donly` / `ponly`。**完全不参考友商数据**。

| 阶段 | 做什么 | 产出 |
| --- | --- | --- |
| 第一段 | `rate=inf`，从 `--sweep-start-batch` 起倍增上探 → 回搜 | 打满时的最大合格并发 |
| 第二段 | 从该并发按 `--batch-growth`（默认 1.5x）放大 batch；每档固定并发，搜该并发下的最大合格 rate；上界按余量继续扩张（见下） | 每档的最大合格 rate |
| 择优 | 每档比较吞吐，更新全局最优；单次回落继续探索 | 全局最优点 |
| 停止 | 平台期且最优余量 ≤ 30%；余量还大时由 `--max-rounds` 兜底 | 收敛 / 未收敛 |
| 收尾 | 复测最优（缓存命中直接返回）→ ±step 局部扰动 | 最终结论 |

#### 第二段的关键规则

- **rate 基准值**：`clamp(20000 / input_len, 0.05, 500)`，即「每秒约 20K 输入 token」折算的请求频率。常数在 `search.py` 的 `RATE_REF_IN_TOK_PER_S`。短输入场景可能算出比实际容量还高的值，会先夹到实测容量，否则整段搜索在退化区打转。
- **结论只从有限 rate 的点里挑**（`rate=0` 是打满哨兵）。第二段一个合格点都没有才退回打满点并打印说明，CSV 的「请求频率」写 `inf`。限速后队列更短，按最优 rate 跑往往能到更高并发。
- **第二段不会因为「限速下并发不是瓶颈」立即停**；只有平台期且余量耗尽才结束（见余量门）。
- **rate 上界按余量扩张**：初始上界 = 实测最高合格吞吐 × 1.5；上界点合格后，余量 > 30% 且 achieved 仍随 rate 涨就继续 ×1.5。跨线、撞 SLO 墙、吞吐不再跟随的探针只用于下结论：**不参与最终择优**，也不更新下一档起搜上界（1.5x 步长下跨线点可能只剩 2% 余量，噪声下不可信）。
- **rate 轴 5% 相对容差**：二分区间收到「**已合格端** × `--rate-rel-tol`」以内即停（下限 `--rate-step`）。承诺结论最多比真实边界低 5%（设 0 恢复精细搜索）；基准取已合格端，避免上界远高于真实边界时容差被放大；代价是每档约多 1 个点。
- **代价**：搜索点数比打满跑法多。粗略量级：打满 10~15 点，搜 rate 15~20 点（命中缓存的不重测），每点几十分钟。这是「用测试时间换吞吐」。

### 打满跑法：只搜 batch

适用范围：`donly` / `ponly` 默认，以及 `full --no-rate-search`。rate 打满后负载完全由并发决定（`--request-rate inf`），只剩 batch 一个变量。

| 步骤 | 做什么 |
| --- | --- |
| 1. 上探 | 从 `--sweep-start-batch`（缺省 1）起 1 → 2 → 4 → 8… 翻倍；合格就继续，第一个不合格点停下（到 `--max-batch` 也停） |
| 下探 | 起点自己不合格 ⇒ 先测 `batch=1`；合格就在 `[1, 起点)` 回搜，否则判「打满时没有可用点」 |
| 2. 回搜 | 在最后一个合格点和第一个不合格点之间回搜（`--batch-step` 为粒度） |
| 3. 择优 | 取所有合格点里吞吐最高的（通常就是边界点） |

- `--sweep-start-batch` 只改**从哪开始**，不改结论：抬高它省掉几个低并发点，起点不合格会自动下探，最多多花一次 `batch=1`。超过 `--max-batch` 会被收到上界。
- 不做并发放大、不做 ±步长扰动 —— 上探 + 回搜已覆盖边界两侧。**任何跑法的第一段都走这条规则**。
- `--max-rounds` / `--rate-step` / `--rate-rel-tol` 只在搜 rate 的跑法里生效。
- 连 `batch=1` 打满都不合格 ⇒ 打满时无可用点，最优记为 batch=1 且 `satisfied=False`。搜 rate 的跑法不受此限：第二段仍可能找到限速下的合格点。
- 结论表「请求频率」：打满跑法写 `inf`；`--rate-search` 写实测频率（多十几个测点）。

#### 回搜打法：`--backsearch`（所有模式通用）

| 打法 | 做法 | 特点 |
| --- | --- | --- |
| `secant`（默认） | 用两端实测时延按幂律 `v = A·x^α` 外推交点 | 少打高并发点；曲线越平滑越准 |
| `bisect` | 每轮取中点，只问合不合格 | 区间稳定减半，打点数固定 |

用幂律不用直线：TTFT 有固定开销，直线会把它吸收成斜率。外推先过合理性检查（该轴变化 < 2%、值反向、交点出区间都放弃），挑不出落点自动退回二分；落点永远夹在 lo/hi 之间。只推真正被考核的轴（如 donly 只推 TPOT）。打法写进 `search_rule`，换打法 ⇒ 旧 state 重搜（已测点仍缓存）。

### 场景没有 TTFT / TPOT 要求时

| 情形 | 行为 |
| --- | --- |
| full 下某轴 SLO = -1（agent 不限 TPOT；ultra_long 两轴都不限） | 该轴完全不判定，`satisfied` 只由另一轴决定；仍受吞吐判据约束 |
| donly 但没定义 TPOT SLO（agent、ultra_long） | 只剩吞吐判据 ⇒ 上探到吞吐过不了线为止；开跑前打 ⚠ 警告但**仍执行** |
| ponly 但没定义 TTFT SLO（ultra_long） | 同上，打 ⚠ 警告 |
| donly/ponly 下「另一轴」超标 | 不影响判定，这正是这两个模式的意义 |

---

## 五、命令行参数

完整帮助：`python3 auto_tune.py --help`、`python3 generate_report.py --help`。

### `auto_tune.py` · 场景与服务

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--scene / -s` | 只跑指定场景，可逗号分隔或重复给出 | 全部 |
| `--scenes-file` | 换一套场景配置（别名 `--scenes`） | `scenes.py` |
| `--mode / -m` | full / donly / ponly / ponly_cached / full_cached | full |
| `--client` | 覆盖各场景 client（`ais_bench` / `sglang.bench_serving`，简写 `ais` / `bench`）；donly/ponly 不受影响 | 按场景配 |
| `--prompts-mult` | 普通档打满点 `num_prompts = batch × 它` | 5 |
| `--long-input-len` | 长输入档阈值：`input_len` **严格大于**它才走长档 | 16384 |
| `--long-input-mult` | 长输入档打满点 `num_prompts = batch × 它` | 2 |
| `--rate-target-s` | 限速点按「跑够这么多秒」给 num_prompts（下限 100，传 0 关掉） | 180 |
| `--host` / `--port` | 服务地址（也可用 `HOST_IP` / `HOST_PORT`） | — / 12345 |
| `--list` | 列出场景后退出 | — |

### `auto_tune.py` · 搜索行为

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--no-rate-search` | **仅 full**：跳过第二段，rate 打满；donly/ponly 忽略 | False |
| `--rate-search` | **仅 donly / ponly**：也搜 request rate；full 忽略。与上一项互斥 | False |
| `--max-rounds` | 余量仍大时平台期的保险丝（仅搜 rate 的跑法）；逼近 SLA 时由 `--plateau-patience` 先停 | 8 |
| `--batch-growth` | 第二段并发放大倍率（仅搜 rate） | 1.5 |
| `--plateau-patience` | 连续多少档无实质提升进入平台期；仅当最优余量 ≤ 30% 才真正停 | 3 |
| `--batch-step` | batch 步长（所有模式，也是回搜停止粒度） | 2 |
| `--sweep-start-batch` | 上探起点（所有模式）；起点不合格会自动下探；超过上界会被夹住 | 1 |
| `--max-batch` | batch 上界（所有模式） | 256 |
| `--backsearch` | 回搜打法：`secant` / `bisect` | secant |
| `--rate-step` | rate 最小步长（仅搜 rate；也是容差下界和微调步长） | 0.05 |
| `--rate-rel-tol` | rate 回搜相对容差（仅搜 rate）；设 0 关闭 | 0.05 |
| `--thru-tol` | 兼容参数；新策略不再用于中止 | -0.02 |
| `--fail-tolerant` | 允许的最大失败请求数 | 0 |
| `--restart` | 忽略已收敛 state，强制重搜 | False |

### `auto_tune.py` · 判定与路径

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--latency-stat` / `--ttft-stat` / `--tpot-stat` | 判定统计量（见第三节） | p90 |
| `--bench-python` | 跑 `-m sglang.bench_serving` 的解释器 | `BENCH_PYTHON` → `python3` |
| `--bench-model` | 模型目录；bench 模式作 `--model`，ais_bench 写进 `AIS_MODEL`/`AIS_PATH` | `scenes.py` 的 `MODEL` |
| `--bench-dataset` | bench_serving 的 `--dataset-path`（传 `""` 关掉） | `scenes.BENCH_DATASET` |
| `--model-name` | 模型短名：日志目录、CSV 文件名、「模型」列都用它 | 按 `--bench-model` 推 |
| `--state-dir` | state / report / 汇总 CSV 目录 | 脚本同目录 |
| `--log-dir` | 日志根目录（非 full 会再拼 `<模型>/<mode>`） | 脚本同目录 `logs/` |

### `generate_report.py`

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--all-modes` | 同时出结论表 + 全测点表（最常用） | False |
| `--all-points` | 只出全测点表（读 `state_*.json`） | False |
| `--out` | 结论表输出路径 | 单模式 `metax_report.csv`；`--all-modes` 为 `metax_summary_<模型>.csv` |
| `--out-points` | 全测点表输出路径 | `metax_report_<模型>.csv` |
| `--report` | 单模式用的 report 路径 | `report.json` |
| `--state-dir` | state / report 所在目录 | 脚本同目录 |
| `--no-states` | 只读 `--report`，忽略 state | False |
| `--scene` | 只输出某场景 | 全部 |
| `--scenes-file` | 与跑测时同一份场景配置（别名 `--scenes`） | `scenes.py` |
| `--print-table` | 打印人可读的对照表 | False |
| `--precision` | 报表 precision 列 | INT8 |
| `--model-name` | 覆盖报表「模型」列 | 从数据里认 |

---

## 六、输出文件

日志 / state / report 的路径见[结果文件放在哪](#结果文件放在哪)，CSV 见[二、生成的 CSV](#二生成的-csv两张表)；结论表 = `metax_summary_<模型>.csv`，全测点表 = `metax_report_<模型>.csv`。日志是原始输出 + 头部元信息（含完整命令行），state 里是所有历史测点 + 最优 + 收敛标记（断点续跑用），report 是各模式的全场景最终报告。

报表元数据只由人给两列：`--model-name`（缺省从数据推）和 `--precision`（缺省 INT8）。部署拓扑（机器 / 卡数 / 并行配置）**不写进报表**，避免被当成实测成绩。

---

## 七、常见问题

**Q：`--latency-stat p95` 直接报错？**
bench_serving 侧已支持，但要把本目录 `bench_serving.py` 覆盖到跑压测的机器；忘了覆盖时每个点都 `parse_ok=False`，结论表零数据行、表头仍写 P95。ais_bench 侧仍不支持，需换 p90/p99/mean/median，或按[第八节 D](#d-以-p95-为标准)改 client。

**Q：donly/ponly 找不到解释器，或 bench_serving 报 `Network is unreachable ... huggingface.co`？** 前者用 `--bench-python` 指定装了 sglang 的环境（或设 `BENCH_PYTHON`）；后者是 random loader 缺 `--dataset-path`，确认 `--bench-dataset` 指向本机 ShareGPT json。

**Q：某个测点 `parse_ok=False`，会被当成不达标吗？** 不会。该点不进缓存、不参与判定，续跑重测；但本场景本次整体不作数、不写结论（`converged=False`）。修好重跑即可，已跑通的点会复用。

**Q：`--sweep-start-batch` 调大会不会跳过真实边界？** 不会。起点不合格会先测 `batch=1`，再把 `[1, 起点)` 交给同一套回搜；只有 `batch=1` 也不合格才判无可用点。最多多花一次测试。

**Q：服务端日志刷 `... without bootstrap room id`？** 请求绕过 router 打到了分离部署的 P/D 实例。donly 有 `--fake-prefill` 没事，ponly 和指到 worker 的 full 会被拒收。指向见[第一节](#donly--ponly-实际发出的命令)。

**Q：report 还没生成，想先看结果？** `generate_report.py --all-modes` 会回落到 `state_*.json` 取「已达标点里吞吐最高的」，并打印提示；全测点表本来就只读 state。

**Q：加了 `--rate-search`，结论表频率还是 `inf`？** 先看会话开头写「不搜」还是「参与搜索」。写「不搜」说明这次没搜（加错命令，或加在 full 上——full 本来就搜，是空操作）。写「参与搜索」却仍出 `inf` ⇒ 第二段没有合格点，如实退回打满点，控制台会打印说明。

**Q：换模型后结果串味？** state 记了模型短名，对不上会跳过。建议 `--state-dir runs/<模型>`；同目录换模型时别混放 `report_*.json`。

**Q：出表报 `PermissionError: ... metax_report_xxx.csv`？** 目录或 CSV 属主是 root。换可写 `--state-dir`，或 `sudo chown -R $(id -un) runs/`；日志同理用 `--log-dir` 绕开。

---

## 八、跑新模型和加新场景：改哪里

### A. 换一个模型（端口 / 模型路径 / dataset 全变）

复制一份脚本，只改配置区这 6 行；`--state-dir` 跟着 `model_name` 走（`runs/<model_name>`），与上一个模型隔离：

```bash
host_ip="127.0.0.1"                              # --host
port=9300                                          # --port
model_path="/path/to/new/model/vllm_quant_model"   # --bench-model
model_name="MyModel"                               # --model-name + --state-dir
bench_python="/opt/conda/envs/sglang/bin/python"   # --bench-python
bench_dataset="/path/to/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"   # --bench-dataset
```

> **换拓扑（卡数 / 机器数）必须换 `--state-dir` 或先删 state**：state 只记「模型 + 模式」，同模型不同拓扑不会被识别，会把旧拓扑的结论当成已收敛。

### B. 换成 sglang 官方 client

donly/ponly 本来就固定用 bench_serving。full 可临时整批切换（`--client bench`），或在 `scenes.py` 里给对应场景加 `client=CLIENT_BENCH,`。换 client 后该场景旧测点不再复用。

### C. 改成 5 倍并发

**已经是默认**（普通档 5、长输入档 2）。

- 临时改：`--prompts-mult 8` / `--long-input-mult 3`
- 改默认值：`runner.py` 的 `DEFAULT_PROMPTS_MULT` / `DEFAULT_PROMPTS_MULT_LONG`
- 注意：倍率变了旧 state 会连测点一起作废重测 —— 请求数一变，数字本身就是另一组。

### D. 以 P95 为标准

#### bench_serving：已支持

本目录 `bench_serving.py` 是上游版本加了 P95 TTFT/TPOT（字段、计算、打印、JSON 输出 4 处）。覆盖到压测机器：

```bash
cp <sglang 安装目录>/bench_serving.py <sglang 安装目录>/bench_serving.py.upstream
cp bench_serving.py <sglang 安装目录>/bench_serving.py
```

`runner.py` 的 `CLIENT_LATENCY_STATS["sglang.bench_serving"]` 已登记 `p95`；忘了覆盖会解析不到 P95、判 `parse_ok=False`。

#### ais_bench：仍不支持

开跑前校验会拦下，需要自己补：

| 步骤 | 改哪里 |
| --- | --- |
| ① 打出 P95 | 改 ais_bench 的 perf 报表，加一列 P95 |
| ② 登记统计量 | `runner.py` 的 `CLIENT_LATENCY_STATS["ais_bench"]` 加 `"p95"`，并在 `AIS_COL_TO_STAT` 加列名映射（如 `"P95": "p95"`） |

解析侧不用改：`_BS_LAT_RE` 按名字匹配 `P95 TTFT (ms)`。报表判定列表头会跟着写成 `P95 首token平均时延（s）`，但「SLO 要求」文本来自场景的 `slo_ttft_text` / `slo_tpot_text`，换口径时要一起写成 `ttft_p95 < 3s` 之类。

### E. 加一个新场景

只改 `scenes.py`，照抄一条 `make_scene(...)` 改字段：

```python
make_scene(
    "my_scene", "我的场景(8K/512)",
    input_len=8192, max_out_len=512,
    slo_ttft=3.0, slo_tpot=0.04,          # 单位秒；-1 = 不限制
    dataset=_ds("8_my_scene.jsonl"),      # client=ais_bench 时必填
    # client=CLIENT_BENCH,                # 想走 bench_serving 才写
)
```

| 字段 | 说明 |
| --- | --- |
| `key` | 短名：`--scene`、日志名、`state_<key>.json` 都用它，不能重复 |
| `name` | 长名：报表「测试用例」列 |
| `input_len` / `max_out_len` | 输入长度 / 输出上限（ponly 压到 1） |
| `slo_ttft` / `slo_tpot` | SLO 上限（秒），-1 = 不限制 |
| `seed_batch` / `seed_rate` | 可选，仅备忘，**不参与搜索** |
| `client` | `ais_bench`（缺省）/ `sglang.bench_serving` |
| `dataset` | ais_bench 读的 jsonl；bench client 不用 |
| `slo_ttft_text` / `slo_tpot_text` | 报表 SLO 文本，缺省按数值生成 |

`scenes.py` 导入时执行 `validate_scenes()`：key 重复、client 不认识、ais_bench 缺 `dataset`、必填为空，都会立刻报错并指出哪一条。加完用 `--list` 确认。

### F. 换一整套场景（另一张 SLO 表）

单独写场景文件，跑测和出表都用 `--scenes-file` 指过去：

```bash
python3 auto_tune.py    --mode full --scenes-file my_scenes.py ...
python3 generate_report.py --all-modes --scenes-file my_scenes.py --state-dir ...
```

- 场景文件一般只写 `SCENES` 就够：`make_scene` / `CLIENT_BENCH` / `MODEL` / `validate_scenes` 等未定义项从 `scenes.py` 继承（`scene_file.py` 负责挂钩）。
- 示例可参考 `test_scenes_cmp_1010.py`：9 个场景（16K~256K），全是 bench_serving，SLO 按 P95 判；`test_scenes_cached.py` 是 cached 模式的单场景示例。
- 场景写法照 `scenes.py` 的 `make_scene(...)`；改了 SLO 阈值旧 state 会自动作废重判。
- **跑测和出表两个入口都要给 `--scenes-file`** —— 出表也按场景集 key 找 state，漏了会出空表。

---

## 九、目录结构与依赖

```text
auto_tune_toolkit/
├── auto_tune.py          # 主入口：搜索每场景最优 batch/rate
├── auto_tune.sh          # 唯一启动脚本（改配置区后用）
├── generate_report.py    # 出两张 CSV
├── search.py             # 搜索算法 + 合格判据（第一段所有跑法共用；搜 rate 再走第二段）
├── runner.py             # 单次测试执行与结果解析（两个 client 都在这里）
├── scenes.py             # 场景配置（SLO / 长度 / client / 数据集的唯一来源）
├── scene_file.py         # --scenes-file 加载器
├── bench_serving.py      # 上游 sglang bench_serving + P95 TTFT/TPOT
├── merge_json_to_csv.py  # JSON 汇总工具
├── test_scenes_*.py      # 场景示例（cached / P95）
├── test_*.py             # 搜索策略 / cached 模式回归测试
└── logs/  runs/          # 运行后生成
```

- 依赖：`python3` 标准库即可，无第三方依赖。ais_bench 场景需 PATH 里有 `ais_bench`；bench_serving 场景需 `--bench-python` 指向装了 sglang 的解释器。
- 本目录只含工具本身，不含历史 runs/logs 数据（从原工作目录复制时未带过来）。
