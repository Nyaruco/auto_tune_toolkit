#!/usr/bin/env python3
"""
自动寻优 —— 为每个场景找到最优 (batch, request_rate)
====================================================

吞吐是目标，TTFT/TPOT 是硬门槛：先找 SLO 可行的最大并发，再继续放大并发，
每档搜索仍满足 SLO 的最大 request rate。单次吞吐回落不会立即停止；连续
--plateau-patience 档无实质提升后进入平台期，且只有最优余量已 <= 30% 才真正停
（余量还大就继续放大并发，rate 上界也继续往 SLA 扩，由 --max-rounds 保险丝兜底）。

  起点     request-rate=inf 下从 --sweep-start-batch（缺省 1）起倍增找 SLO 最大并发
           （起点自己不合格就下探 [1, 起点)）；随后按 --batch-growth 放大并发，
           rate 从「输入长度 x 2 万 token/s」折算的基准值起（见 search.baseline_rate）
  模式     full  = TTFT + TPOT 都考核，batch / rate 两轴都搜
                  加 --no-rate-search 则不搜 rate（直接打满），只搜 batch
           donly = 只考核 TPOT，正常输出长度，默认 rate 打满、只搜 batch（= 并发）
           ponly = 只考核 TTFT，输出长度压到 1，同上
           ponly_cached = 只考核 TTFT，使用 generated-shared-prefix 数据集，
                          默认搜索 request rate；每次测试使用唯一 seed，
                          场景需配置 cached_input_len / non_cached_input_len
           full_cached = TTFT + TPOT 都考核，同样使用 generated-shared-prefix
                         和唯一 seed；场景配置与 ponly_cached 相同
           搜不搜 rate 由跑法定，不由模式定（见 runner.saturates_run）：donly/ponly
           加 --rate-search 就改成跟 full 一样搜 rate（第二段同一套规则），此时考核轴 /
           client / 输出长度还是各自模式的。两个开关互斥，且落在"本来就这样"那侧的
           是空操作（--mode full --rate-search == --mode full）。
  client   full 按场景配（scenes.py 的 client 字段，ais_bench / bench_serving）
           donly / ponly 固定 python -m sglang.bench_serving（要用它的
           --request-rate / --fake-prefill / 输出长度控制）
           —— 同 ZYD/d-only-sla.sh、p-only-sla.sh
  num_prompts 打满点 = batch * --prompts-mult（缺省 5）
              长输入场景（input_len 严格大于 --long-input-len，缺省 16K）改用
              --long-input-mult（缺省 2）—— 长输入单条请求慢，倍数小一点就够
              ponly_cached 下该值写入 --gsp-prompts-per-group
              限速点 = min(那个, max(rate * --rate-target-s, 100))（缺省 180 秒）
              —— 限速下并发够接住 rate 就不是瓶颈，跑再久也买不到新信息；
                 100 条是分位数判定的最小样本量（rate 很低时按秒数折算只有几十条）
  PD 部署参数 (P/D 节点 / 卡数 / 机器数) 仅作元数据记录，不参与算法 ——
  服务已由部署侧就绪，本脚本通过 HOST_IP/HOST_PORT 访问

  用 bench_serving 时得确认这几个路径:
    --bench-python  装 sglang 的解释器（默认 python3 / 环境变量 BENCH_PYTHON）
    --bench-model   模型目录，即 bench_serving 的 --model（默认 scenes.py 的 MODEL）
    --bench-dataset 默认本机的 ShareGPT_V3_unfiltered_cleaned_split.json，一般不用改
                    （random loader 缺它就会去 HuggingFace 下载，离线机器直接失败）
  输入长度不用设：按场景 input_len 用 --random-input-len 生成。

用法:
  cd /path/to/auto_tune

  # 单场景：对话场景 (4K/1K)
  python3 auto_tune.py --scene dialog --host 127.0.0.1 --port 12345

  # D only / P only（bench_serving，rate 打满，只搜 batch=并发）
  python3 auto_tune.py --scene dialog --mode donly --host 127.0.0.1 --port 12345 \
      --bench-python /path/to/env/bin/python

  # P only 也要「请求频率」这一维（跟 full 第二段同一套规则）
  python3 auto_tune.py --scene dialog --mode ponly --rate-search \
      --host 127.0.0.1 --port 12345 --bench-python /path/to/env/bin/python

  # 临时整批换 client / 改并发倍数
  python3 auto_tune.py --client bench --prompts-mult 8 ...

  # full 但不要「请求频率」这一维（少搜十来个测点，每个几十分钟）
  python3 auto_tune.py --mode full --no-rate-search --scene dialog ...

  # 重置（忽略已收敛 state，强制重新搜索）
  python3 auto_tune.py --scene dialog --restart ...

  场景 / SLO / 各场景用哪个 client 都配在 scenes.py，加新场景只改那一个文件。

输出:
  logs/         原始 benchmark 日志（每条 (scene, batch, rate, ts) 一份）
  state_*.json  每场景断点续跑文件
  report.json   全部场景的最优结果汇总

环境变量:
  HOST_IP / HOST_PORT 也可通过环境传入（命令行参数优先）
  BENCH_PYTHON 同 --bench-python

依赖:
  client=ais_bench 时需 PATH 中有可执行的 ais_bench
  client=bench_serving 时需 --bench-python 指向装了 sglang 的解释器
  Python3 (无第三方依赖)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# 保证能 import 同目录模块
sys.path.insert(0, str(Path(__file__).parent.resolve()))

# --scenes-file 换场景配置 —— 必须在下面 `from scenes import ...` 之前生效，
# 那几个名字是从 sys.modules['scenes'] 上取的，见 scene_file.py。
import scene_file

SCENES_FILE = scene_file.bootstrap(sys.argv[1:], Path(__file__).parent)

from scenes import (SCENES, SCENE_BY_KEY, MODEL, BENCH_DATASET, CLIENTS,
                    normalize_client, derive_model_name)
from search import (search, FIXED_BATCH_CAP, baseline_rate, DEFAULT_THRU_TOL,
                    DEFAULT_BATCH_GROWTH, DEFAULT_PLATEAU_PATIENCE,
                    SWEEP_START_BATCH, BACKSEARCH_SECANT, BACKSEARCH_MODES)
from runner import (MODE_FULL, CACHED_MODES, MODE_CHOICES, MODE_DESC,
                    mode_missing_slo, mode_out_len, mode_axes,
                    saturates_run, BENCH_RATE_INF,
                    client_for, BenchClientCfg, DEFAULT_PROMPTS_MULT,
                    DEFAULT_PROMPTS_MULT_LONG, DEFAULT_LONG_INPUT_LEN,
                    is_long_input,
                    DEFAULT_RATE_TARGET_S, RATE_TARGET_MIN_PROMPTS,
                    CLIENT_BENCH, StatCfg, normalize_stat, stat_label,
                    entry_lat_ms, LATENCY_STATS, DEFAULT_LATENCY_STAT,
                    cache_input_lens, cache_hit_ratio, is_cached_mode)
import generate_report


def stat_arg(v: str) -> str:
    """argparse 的 --latency-stat 类型：归一化并当场校验，输错了立刻报错。"""
    s = normalize_stat(v)
    if s not in LATENCY_STATS:
        raise argparse.ArgumentTypeError(
            f"未知统计量 {v!r}，可选: {', '.join(LATENCY_STATS)}")
    return s


# ──────────────────────────────────────────────────────────
# 参数
# ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scene", "-s", action="append", metavar="KEY[,KEY...]",
                        help="只跑指定场景，默认全部。可逗号分隔或重复给出，"
                             "如 --scene dialog,knowledge ；"
                             "不跑 ultra_long 时直接把它排除即可")
    parser.add_argument("--scenes-file", "--scenes", default="", metavar="FILE",
                        dest="scenes_file",
                        help="换一套场景配置（如 scenes_custom.py），缺省用同目录的 "
                             "scenes.py。写法照 scenes.py，一般只定义 SCENES / "
                             "SCENE_BY_KEY 即可，make_scene / CLIENT_* 可以直接 "
                             "`from scenes import ...` 复用。指定文件在工具 import "
                             "场景模块之前生效 —— --list、跑的场景、事后的汇总表"
                             "看到的都是它那套。")
    parser.add_argument("--mode", "-m", choices=list(MODE_CHOICES),
                        default=MODE_FULL,
                        help="测试模式: full=TTFT+TPOT 都考核 (默认), client 按场景配; "
                             "donly=正常输出长度, 只考核 TPOT; "
                             "ponly=输出长度压到 1, 只考核 TTFT; "
                             "ponly_cached=generated-shared-prefix cache 命中场景, "
                             "只考核 TTFT; "
                             "full_cached=generated-shared-prefix cache 命中场景, "
                             "TTFT 和 TPOT 都考核。"
                             "donly/ponly 固定用 python -m sglang.bench_serving "
                             "(random 数据集), 默认 request rate 打满 (只按 batch=并发 "
                             "找边界); 要连 request rate 一起搜加 --rate-search。"
                             "日志写入 logs/<模型>/<mode>/, "
                             "state/report 也按模式分开存")
    parser.add_argument("--client", default="", metavar="NAME",
                        help="压测 client, 覆盖 scenes.py 里各场景的 client: "
                             f"{' / '.join(CLIENTS)} (简写 ais / bench 也行)。"
                             "缺省按场景配。donly/ponly 固定 bench_serving, 不受影响。"
                             "换 client 会让该场景的 full state 失效并重搜")
    parser.add_argument("--prompts-mult", type=int, default=DEFAULT_PROMPTS_MULT,
                        metavar="N",
                        help="num_prompts = batch * N (默认 %d)。改大一点可以拉长压测"
                             "时长、让吞吐更稳, 但每个测试点也更慢" % DEFAULT_PROMPTS_MULT)
    parser.add_argument("--long-input-len", type=int, default=DEFAULT_LONG_INPUT_LEN,
                        metavar="LEN",
                        help="长输入档的阈值 (默认 %d = 16K)。场景 input_len **严格大于**"
                             "它就走长输入档。三种模式都适用"
                             % DEFAULT_LONG_INPUT_LEN)
    parser.add_argument("--long-input-mult", type=int, default=DEFAULT_PROMPTS_MULT_LONG,
                        metavar="N",
                        help="长输入档的 num_prompts 倍率 (默认 %d)：input_len > "
                             "--long-input-len 的场景用 batch * N 而不是 batch * "
                             "--prompts-mult。长输入单条请求慢，倍数小一点就够，"
                             "能显著缩短单点耗时" % DEFAULT_PROMPTS_MULT_LONG)
    parser.add_argument("--host",
                        default=os.environ.get("HOST_IP", ""),
                        help="HOST_IP，也可走环境变量")
    parser.add_argument("--port",
                        default=os.environ.get("HOST_PORT", "12345"),
                        help="HOST_PORT")
    parser.add_argument("--max-rounds",  type=int, default=8,
                        help="第二段平台期保险丝 (默认 8)。只对搜 rate 的跑法有效: "
                             "最优余量仍 > 30%% 时，连续这么多档无实质吞吐提升就强制停止"
                             "（已逼近 SLA 时由 --plateau-patience 先停）")
    parser.add_argument("--batch-growth", type=float, default=DEFAULT_BATCH_GROWTH,
                        help=f"第二段并发放大倍率 (默认 {DEFAULT_BATCH_GROWTH:g})。"
                             "每档都会重新搜索满足 SLO 的最大 request rate")
    parser.add_argument("--plateau-patience", type=int,
                        default=DEFAULT_PLATEAU_PATIENCE,
                        help=f"第二段连续多少档无实质吞吐提升后进入平台期 "
                             f"(默认 {DEFAULT_PLATEAU_PATIENCE})。仅当最优余量已 <= 30%%"
                             f"（逼近 SLA）才真正停；余量仍大就继续放大并发，"
                             f"由 --max-rounds 保险丝兜底")
    parser.add_argument("--batch-step",  type=int, default=2,
                        help="batch 最小步长 (默认 2)。并发上探后回搜的停止粒度 "
                             "(可给 1 搜得更细); 搜 rate 的跑法里第二段的 batch 微调也用它")
    parser.add_argument("--backsearch",  choices=BACKSEARCH_MODES,
                        default=BACKSEARCH_SECANT,
                        help="撞线后在「已知合格并发」和「已知不合格并发」之间找最大可用值"
                             "的打法 (默认 secant)。secant=割线回搜: 拿两端实测的时延数值"
                             "外推幂律交点，直接落到边界附近，比二分少打好几个高并发点"
                             "(高并发那头的点最贵)；解不出交点时自动退回二分。"
                             "bisect=老办法, 每轮取中点把区间减半。两者收敛到同一个边界，"
                             "打法变了旧 state 会作废重搜")
    parser.add_argument("--max-batch",   type=int, default=0,
                        help="batch 搜索上界。缺省 %d —— 三种模式一样，从 --sweep-start-batch "
                             "起倍增上探，这个值只决定最多探到哪 (不再参考友商 seed_batch)"
                             % FIXED_BATCH_CAP)
    parser.add_argument("--sweep-start-batch", type=int, default=SWEEP_START_BATCH,
                        help=f"第一段倍增上探的**起点** (默认 {SWEEP_START_BATCH})。"
                             "缺省从 1 起翻倍，低并发区那几个点在大机器上纯属白跑，"
                             "抬高起点能省掉它们；起点自己不合格时会自动下探 [1, 起点) "
                             "把边界找回来，所以抬高起点不会让结论偏小。"
                             "必须 ≥ 1，且超过 --max-batch 时收到上界")
    # 百分号写 %%：argparse 的 _expand_help 还会对 help 再 % 格式化一次，裸 % 会炸
    parser.add_argument("--thru-tol",    type=float, default=DEFAULT_THRU_TOL,
                        help=f"兼容参数，不再用于中止搜 rate 的跑法。当前策略以 SLO 为硬门槛，"
                             f"吞吐只用于最终择优；连续无提升由 --plateau-patience 控制。"
                             f"缺省 {DEFAULT_THRU_TOL:.2f} 的旧命令行仍可运行")
    parser.add_argument("--rate-step",   type=float, default=0.05,
                        help="rate 最小步长 (默认 0.05)。只对搜 rate 的跑法有效 (full, 或 "
                             "donly/ponly + --rate-search): 既是二分容差的绝对值下限"
                             "(取它和「已合格端 x --rate-rel-tol」里的较大者)，也是最优 rate "
                             "邻域扰动测试的步长; rate 打满时不参与搜索")
    parser.add_argument("--rate-rel-tol", type=float, default=0.05,
                        help="rate 轴相对搜索容差 (默认 0.05 = 5%%)。二分到区间宽度 < "
                             "已合格端*该比例 即停，最优 rate 最多比真实边界低这个比例，"
                             "换来二分次数从 ~9 次降到 ~4 次；设 0 可关闭（恢复精细搜索）")
    _rs = parser.add_mutually_exclusive_group()     # 一个说"别搜 rate"、一个说"要搜"，同给没意义
    _rs.add_argument("--no-rate-search", action="store_true",
                     help="full 专用: 跳过第二段, request rate 直接打满, 只搜并发。"
                          "考核轴 / client / 输出长度都还是 full 的, 只是不要「请求频率」"
                          "这一维 —— 省掉第二段的十来个测点 (每个几十分钟)。"
                          "注意它不是 donly/ponly: 那两个是**另外的模式**, 换模式还会"
                          "换 client。两种跑法的 state 互不复用 (规则名不同, 自动重搜)")
    _rs.add_argument("--rate-search", action="store_true",
                     help="donly / ponly 专用: 这两个模式默认 request rate 直接打满, 加这个"
                          "开关改成跟 full 一样搜 request rate (同一套规则: 同样的起点折算 / "
                          "容差 / 死轴判据)。考核轴 / client / 输出长度还是 donly/ponly 的 "
                          "—— donly 照样 --fake-prefill, ponly 照样把输出压到 1。"
                          "要报「请求频率」这一列、或想知道多高的请求频率还扛得住时用它, "
                          "代价是第二段的十来个测点。两种跑法的 state 互不复用")
    parser.add_argument("--rate-target-s", type=float, default=DEFAULT_RATE_TARGET_S,
                        metavar="SEC",
                        help="限速测点跑多久 (默认 %g 秒)。限速点的 num_prompts = "
                             "min(batch * 本场景倍率, max(rate * 它, %d)) —— 长输入档的"
                             "场景那个倍率是 --long-input-mult; 按 batch "
                             "给的话并发越大跑越久, 而限速下并发够接住 rate 就不是瓶颈"
                             "了, 多出来的时间买不到新信息; 下限 %d 条是分位数判定要的"
                             "最小样本量 (rate 很低时 %g 秒只折算出几十条)。设 0 关掉, "
                             "退回按 batch 给"
                             % (DEFAULT_RATE_TARGET_S, RATE_TARGET_MIN_PROMPTS,
                                RATE_TARGET_MIN_PROMPTS, DEFAULT_RATE_TARGET_S))
    # ── 判定用的时延统计量 ──
    parser.add_argument("--latency-stat", type=stat_arg, default=DEFAULT_LATENCY_STAT,
                        metavar="STAT",
                        help="TTFT/TPOT 用哪个统计量判定 SLO: "
                             "mean / median / p50 / p90 / p95 / p99 (默认 p90)。"
                             "同时管两轴，可用 --ttft-stat / --tpot-stat 单独覆盖。"
                             "注意两个 client 打出来的统计量不一样 —— ais_bench(full) "
                             "没有 p50/p95；sglang bench_serving 的 p95 要靠本目录那份"
                             "打过补丁的 bench_serving.py（覆盖到 sglang 安装目录）。"
                             "选了没有的会直接报错，并提示去改哪个文件的哪一处。")
    parser.add_argument("--ttft-stat", type=stat_arg, default=None, metavar="STAT",
                        help="只覆盖 TTFT 的统计量（默认跟 --latency-stat）")
    parser.add_argument("--tpot-stat", type=stat_arg, default=None, metavar="STAT",
                        help="只覆盖 TPOT 的统计量（默认跟 --latency-stat）")
    parser.add_argument("--fail-tolerant", type=int, default=0,
                        help="允许的失败请求数 (默认 0)")
    parser.add_argument("--restart",     action="store_true",
                        help="忽略已收敛 state，强制重新搜索")
    parser.add_argument("--list",        action="store_true",
                        help="列出场景后退出")
    parser.add_argument("--log-dir",     default="",
                        help="日志输出目录（默认 ./logs）")
    parser.add_argument("--state-dir",   default="",
                        help="state 文件目录（默认当前目录）")
    # ── bench_serving 的路径（donly/ponly 必用；full 里 client=bench 的场景才用）──
    parser.add_argument("--bench-python", default="",
                        help="跑 `python -m sglang.bench_serving` 的解释器（装 sglang 的那个 "
                             "环境，如 /path/to/env/bin/python）。缺省取环境变量 BENCH_PYTHON，"
                             "再缺省 python3")
    parser.add_argument("--bench-model",  default="",
                        help="模型目录。用 bench_serving 时是它的 --model（加载 tokenizer "
                             "算 token 数）；用 ais_bench 时同一个路径写进 AIS_MODEL/AIS_PATH。"
                             "缺省 scenes.py 里的 MODEL = %s" % MODEL)
    parser.add_argument("--model-name",   default="",
                        help="模型短名，用于日志目录 logs/<模型>/<mode>/ 与汇总 CSV 的"
                             "文件名/「模型」列。缺省从模型路径推：取 basename，遇到 "
                             "vllm_quant_model 这类占位名则退一级取父目录名。"
                             "换模型跑时建议显式指定，避免和上一个模型的结果并进同一张表")
    parser.add_argument("--bench-dataset", default=BENCH_DATASET,
                        help="bench_serving 的 --dataset-path，缺省 scenes.BENCH_DATASET"
                             "（本机那份 ShareGPT_V3_unfiltered_cleaned_split.json）。"
                             "请求虽然由 --dataset-name random 生成，但 random loader 缺了它"
                             "就会去 HuggingFace 下载同一个文件，离线机器上会直接失败。"
                             "传空串 = 不传该参数")
    return parser.parse_args()


def list_scenes():
    print(f"{'key':<14}  {'name':<28}  {'in':<7}  {'out':<6}  {'ttft':<6}  "
          f"{'tpot':<7}  {'rate基准':<9}  {'client':<20}  cache")
    print("-" * 132)
    for s in SCENES:
        ttft = "—" if s["slo_ttft"] < 0 else f"{s['slo_ttft']}s"
        tpot = "—" if s["slo_tpot"] < 0 else f"{s['slo_tpot']*1000:.0f}ms"
        cache = ""
        if s.get("cached_input_len") is not None or s.get("non_cached_input_len") is not None:
            try:
                cache = (f"{s.get('cached_input_len')}/"
                         f"{s.get('non_cached_input_len', s.get('none_cached_input_len'))}"
                         f" hit={cache_hit_ratio(s):.1%}")
            except ValueError:
                cache = "配置不完整"
        # rate 基准 = 输入长度折算的 full 搜索起点（见 search.baseline_rate）
        print(f"{s['key']:<14}  {s['name']:<28}  {s.get('input_len', '—'):<7}  "
              f"{s['max_out_len']:<6}  {ttft:<6}  {tpot:<7}  "
              f"{baseline_rate(s):<9}  {s.get('client', '—'):<20}  {cache}")
    print("\n  in/out = input_len / max_out_len (token)；donly/ponly 按 in 生成 random 输入, "
          "ponly 的 out 固定 1")
    print("  rate基准 = 按输入长度折算的 full 搜索起点；并发从初始值倍增上探，"
          "两者都不参考友商")


def main():
    args = parse_args()

    if args.list:
        list_scenes()
        return

    stat = StatCfg.resolve(args.latency_stat, args.ttft_stat, args.tpot_stat)

    base = Path(__file__).parent.resolve()
    log_dir   = Path(args.log_dir)   if args.log_dir   else base / "logs"
    state_dir = Path(args.state_dir) if args.state_dir else base

    # 模型路径与模型短名（换模型跑时，日志目录 / 汇总 CSV 的文件名与「模型」列都跟着变）
    model_path = args.bench_model or MODEL
    model_name = args.model_name or derive_model_name(model_path)
    if not model_name:
        sys.exit("模型短名是空的 —— 用 --scenes-file 换的场景文件里 MODEL_NAME 留了空，"
                 "跑的时候请用 --model-name 指定（日志目录和汇总 CSV 都要用它）")

    # 非 full 模式：日志单独归到 logs/<模型>/<mode>/，不和 full 的日志混在一起
    # （full 保持原样 logs/xxx.log，兼容已有 state 里记录的 log 路径）
    if args.mode != MODE_FULL:
        log_dir = log_dir / model_name / args.mode
    log_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    if not args.host:
        print("[警告] HOST_IP 未提供 —— client 会用其默认服务地址。"
              "若服务在远端需通过 --host / HOST_IP 指定。", file=sys.stderr)

    # bench_serving 的路径配置（donly/ponly 必用；full 里 client=bench 的场景才用）
    bench_cfg = BenchClientCfg(
        model=model_path,
        python=args.bench_python or os.environ.get("BENCH_PYTHON", "python3"),
        dataset=args.bench_dataset,
    )

    # 选场景：--scene dialog,knowledge 或 --scene dialog --scene knowledge；不填=全部
    wanted = [k.strip() for s in (args.scene or []) for k in s.split(",") if k.strip()]
    bad = [k for k in wanted if k not in SCENE_BY_KEY]
    if bad:
        print(f"[错误] 未知场景: {', '.join(bad)}\n"
              f"       可选: {', '.join(s['key'] for s in SCENES)}", file=sys.stderr)
        sys.exit(2)
    chosen = [s for s in SCENES if not wanted or s["key"] in wanted]

    if args.sweep_start_batch < 1:
        print(f"[错误] --sweep-start-batch 必须 ≥ 1（给的是 {args.sweep_start_batch}）",
              file=sys.stderr)
        sys.exit(2)
    if args.max_batch < 0:
        # 0 = 用缺省 FIXED_BATCH_CAP（见 --max-batch 的 default=0）；负数没有任何含义，
        # 放到搜索里只会让上探起点被"收到上界"收到负数上。
        print(f"[错误] --max-batch 不能为负（给的是 {args.max_batch}；0 = 缺省 "
              f"{FIXED_BATCH_CAP}）", file=sys.stderr)
        sys.exit(2)

    if args.mode in CACHED_MODES:
        for s in chosen:
            try:
                cached_len, noncached_len = cache_input_lens(s)
            except ValueError as e:
                print(f"[错误] {e}", file=sys.stderr)
                sys.exit(2)
            total_len = cached_len + noncached_len
            if s.get("input_len") and int(s["input_len"]) != total_len:
                print(
                    f"[错误] 场景 {s['key']} 的 input_len={s['input_len']}，"
                    f"但 cached_input_len + non_cached_input_len={total_len}；"
                    "cache 模式要求两者一致",
                    file=sys.stderr,
                )
                sys.exit(2)
            s["input_len"] = total_len

    # --client 是全局覆盖（临时整批换压测方式），缺省按 scenes.py 里各场景自己配的。
    # donly/ponly 固定 bench_serving，覆盖不生效 —— 交给 client_for 判。
    client_override = normalize_client(args.client) if args.client else ""
    if client_override and client_override not in CLIENTS:
        print(f"[错误] 未知 --client {args.client!r}，可选: "
              f"{' / '.join(CLIENTS)}（简写 ais / bench 也行）", file=sys.stderr)
        sys.exit(2)

    def scene_client(s: dict) -> str:
        # 覆盖要不要生效由 client_for 判（donly/ponly 恒 bench_serving）——
        # 这里只负责把覆盖值塞进去，别再自己判一遍模式。
        return client_for({**s, "client": client_override} if client_override else s,
                          args.mode)

    # 两个 client 打出来的统计量不一样，按场景实际用的那个 client 逐个校验。
    # 在这里拦下来，免得白跑几十分钟才在日志解析那一步发现"没有这个分位"。
    stat_errs: dict = {}                     # client → (错误说明, [场景 key])
    for s in chosen:
        client = scene_client(s)
        err = stat.check_supported(client)
        if err:
            stat_errs.setdefault(client, (err, []))[1].append(s["key"])
    if stat_errs:
        for client, (err, keys) in stat_errs.items():
            print(f"[错误] 场景 {', '.join(keys)} 用的 client 是 {client}: {err}",
                  file=sys.stderr)
        print(f"       本次请求: --latency-stat {args.latency_stat}"
              + (f" --ttft-stat {args.ttft_stat}" if args.ttft_stat else "")
              + (f" --tpot-stat {args.tpot_stat}" if args.tpot_stat else ""),
              file=sys.stderr)
        sys.exit(2)

    # bench_serving 的请求由 --random-input-len 生成，场景必须给出 input_len。
    # 缺了就在开跑前一次性报出来（否则每个测试点都要失败一次，白等很久）。
    lack = [s["key"] for s in chosen
            if not s.get("input_len") and scene_client(s) == CLIENT_BENCH]
    if lack:
        print(f"[错误] 用 bench_serving 的场景需要 input_len（--random-input-len），"
              f"但以下场景没定义: {', '.join(lack)}\n"
              f"       请在 scenes.py 里补 input_len 后重跑。", file=sys.stderr)
        sys.exit(2)

    if CLIENT_BENCH in {scene_client(s) for s in chosen}:
        if shutil.which(bench_cfg.python) is None and not Path(bench_cfg.python).exists():
            print(f"[警告] --bench-python '{bench_cfg.python}' 不存在或不在 PATH —— "
                  f"每次测试都会失败。请指向装 sglang 的解释器，例如 "
                  f"--bench-python /path/to/env/bin/python（或 export BENCH_PYTHON=...）。",
                  file=sys.stderr)

    print("\n" + "=" * 78)
    print("  AUTO TUNE — 自适应寻优")
    print("=" * 78)
    print(f"  HOST          : {args.host}:{args.port}")
    print(f"  模型短名       : {model_name}   (日志目录/汇总 CSV 用它)")
    # rate 打不打满这一件事，本文件里三处（模式行、num_prompts 行、下面 results 里的
    # rate_saturated）都用同一个判定：runner.saturates_run。两开关各只对一个方向有意义，
    # 落在"本来就这样"那一侧就是空操作（--mode full --rate-search 等于没加），归一化在
    # saturates_run 里，跟 search() 用同一份 —— 两边不会再算出不同结论。
    _sat = saturates_run(args.mode, no_rate_search=args.no_rate_search,
                         rate_search=args.rate_search)
    print(f"  模式           : {args.mode} — {MODE_DESC.get(args.mode, '')}"
          f"   rate {'打满 ' + BENCH_RATE_INF + '（不搜）' if _sat else '参与搜索'}")
    used_clients = {c: sum(1 for s in chosen if scene_client(s) == c)
                    for c in dict.fromkeys(scene_client(s) for s in chosen)}
    print("  client         : " + "   ".join(
        f"{c}（{n} 个场景）" for c, n in used_clients.items()))
    _chk_ttft, _chk_tpot = mode_axes(args.mode)
    print(f"  判定统计量     : TTFT {stat_label(stat.ttft)}"
          f"{'' if _chk_ttft else '(本模式不考核)'}   "
          f"TPOT {stat_label(stat.tpot)}{'' if _chk_tpot else '(本模式不考核)'}"
          f"   (--latency-stat)")
    if CLIENT_BENCH in used_clients:
        print(f"  bench 解释器   : {bench_cfg.python}   (python -m sglang.bench_serving)")
        print(f"  bench 模型路径 : {bench_cfg.model_or_default()}   (--model)")
        if is_cached_mode(args.mode):
            print(f"  bench 数据集   : generated-shared-prefix（不使用 --dataset-path）")
            print(f"  输入长度       : generated-shared-prefix "
                  f"cached + non_cached，按场景配置")
        else:
            print(f"  bench 数据集   : {bench_cfg.dataset or '（不传 --dataset-path —— random loader 会去 HuggingFace 下载, 无外网会失败）'}")
            print(f"  输入长度       : 按场景 input_len 用 --random-input-len 生成, 不读场景 jsonl")
    # 倍率分长短两档，逐个场景解析（判据在 runner.is_long_input），横幅按实际用到的档说
    _long_keys = [s["key"] for s in chosen
                  if is_long_input(s, args.long_input_len)]
    _mult_desc = f"打满点 batch * {args.prompts_mult}"
    if _long_keys:
        _mult_desc += (f"；输入 > {args.long_input_len} 的场景 batch * "
                       f"{args.long_input_mult}（{len(_long_keys)} 个: "
                       f"{', '.join(_long_keys)}）")
    print(f"  num_prompts    : {_mult_desc}"
          + ("" if _sat else f"；限速点 min(那个, rate * {args.rate_target_s:g}s)")
          + ("   (rate 已打满, 并发 = 全量请求数)" if _sat else ""))
    if is_cached_mode(args.mode):
        print(f"  cached input   : cached + non_cached -> "
              f"--gsp-system-prompt-len / --gsp-question-len")
        print(f"  seed           : 每个实际测点自动生成唯一 seed")
    print(f"  日志目录       : {log_dir}")
    print(f"  state 目录     : {state_dir}")
    print(f"  待测场景 ({len(chosen)}): {', '.join(s['key'] for s in chosen)}")
    _warn = [(s, m) for s, m in ((s, mode_missing_slo(s, args.mode)) for s in chosen) if m]
    if _warn:
        print("-" * 78)
        print(f"  ⚠ {args.mode} 模式只考核单一轴, 但以下场景没有定义该轴的 SLO ——")
        print(f"    没有任何时延约束, 所有测试点都算达标, 搜索会一路顶到上界")
        print(f"    (退化成「纯求最大吞吐」):")
        for s, m in _warn:
            print(f"      {s['name']:<26}  {args.mode} 只考核 {m.upper()}, 场景未定义该 SLO")
    print("=" * 78)

    results = []

    for sc in chosen:
        # state 按模式分文件：否则 donly/ponly 的结果会污染 full 的搜索断点
        sfx = "" if args.mode == MODE_FULL else f"_{args.mode}"
        state_path = state_dir / f"state_{sc['key']}{sfx}.json"
        if args.restart and state_path.exists():
            print(f"\n  [--restart] 删除旧 state: {state_path}")
            state_path.unlink()

        # 总是拷一份：下面要往场景里塞 --client 覆盖，不能改到 SCENES 里的对象
        sc_meta = {**sc}
        if client_override:
            sc_meta["client"] = client_override
        sc_client = scene_client(sc)

        state = search(
            sc_meta,
            host_ip=args.host, host_port=args.port,
            log_dir=log_dir, state_path=state_path,
            max_rounds=args.max_rounds,
            batch_step_min=args.batch_step,
            rate_step_min=args.rate_step,
            rate_rel_tol=args.rate_rel_tol,
            thru_tol=args.thru_tol,
            fail_tolerant=args.fail_tolerant,
            mode=args.mode,
            max_batch_override=args.max_batch or None,
            bench_cfg=bench_cfg,
            model_name=model_name,
            stat=stat,
            prompts_mult=args.prompts_mult,
            prompts_mult_long=args.long_input_mult,
            long_input_len=args.long_input_len,
            rate_target_s=args.rate_target_s,
            no_rate_search=args.no_rate_search,
            rate_search=args.rate_search,
            batch_growth=args.batch_growth,
            plateau_patience=args.plateau_patience,
            sweep_start_batch=args.sweep_start_batch,
            backsearch=args.backsearch,
        )

        # search() 遇到 "state 文件模式/模型不匹配" 会原样返回那边的 state ——
        # 别把它当成本次的结果写进报告（否则汇总表里会出现别的模式/别的模型的数据）
        skipped = (state.get("mode", MODE_FULL) != args.mode
                   or (state.get("model_name") and state["model_name"] != model_name))
        results.append({
            "scene_key":     sc["key"],
            "scene_name":    sc["name"],
            "mode":           args.mode,
            "model_name":     model_name,
            "client":         sc_client,
            "max_out_len":    mode_out_len(sc, args.mode),
            "input_len":      sc.get("input_len"),
            # 结论里「请求频率」列写 inf 的情况：本次 rate 就是打满的（_sat），以及搜
            # rate 的跑法定不出有限 rate 而退回打满点的兜底 —— 最后这条只有 state
            # 知道，所以优先信它，否则报告会把 rate=0.0 当成一个实测频率写出去。
            "rate_saturated": (not skipped and
                               (state.get("optimal") or {}).get("rate_saturated")) or _sat,
            "latency_stat":   stat.as_dict(),
            "max_batch":      None if skipped else state.get("max_batch"),
            "skipped":        skipped,
            "optimal":        None if skipped else state.get("optimal"),
            "num_tests":      0 if skipped else len(state.get("tests", [])),
            "converged":      False if skipped else state.get("converged", False),
        })

    # 打包汇总报告
    report = {
        "generated_at": datetime.now().isoformat(),
        "mode":          args.mode,
        "latency_stat":  stat.as_dict(),
        "model":         model_name,
        "model_path":    model_path,
        "host":         args.host,
        "port":         args.port,
        "results":      results,
    }
    # 非 full 模式写 report_<mode>.json，避免覆盖 full 模式的 report.json
    report_path = state_dir / ("report.json" if args.mode == MODE_FULL
                               else f"report_{args.mode}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 简表打印
    print("\n" + "=" * 90)
    print(f"  最终结果汇总   [mode={args.mode} — {MODE_DESC.get(args.mode, '')}"
          f" | 判定统计量 {stat.label()}]")
    print("=" * 90)
    print(f"  {'场景':<24}  {'batch':<6}  {'rate':<7}  {'TTFT(ms)':<10}  "
          f"{'TPOT(ms)':<10}  {'out tok/s':<12}  {'SLO':<5}")
    print("-" * 90)
    for r in results:
        opt = r["optimal"] or {}
        b   = opt.get("batch", "-")
        # 打满的结论写 inf，别把哨兵 rate=0.0 当实测频率印出来 —— 跟 CSV「请求频率」
        # 列同一套判据（那里也是先看 rate_saturated 再看 rate）。
        rr  = ("inf" if opt.get("rate_saturated") else opt.get("rate", "-")) if opt else "-"
        # 老 report 里没有 latency_stat → 按 p90 取（老数据的口径）
        t   = entry_lat_ms(opt, "ttft", stat.ttft) if opt else None
        p   = entry_lat_ms(opt, "tpot", stat.tpot) if opt else None
        oto = opt.get("out_tok_thr")
        okf = "✓" if opt.get("satisfied") else "✗"
        ttft = f"{t:.2f}"     if t   else "-"
        tpot = f"{p*1000:.2f}"if p   else "-"
        otot = f"{oto:.2f}"   if oto else "-"
        print(f"  {r['scene_name'][:24]:<24}  {str(b):<6}  {str(rr):<7}  "
              f"{ttft:<10}  {tpot:<10}  {otot:<12}  {okf:<5}")
    print("=" * 90)
    print(f"  报告 → {report_path}")
    print(f"  日志 → {log_dir}")

    # 跑完这个模型的这一种模式后，把该模型已有的 full/donly/ponly 结果汇总成一张 CSV。
    # 每次运行都重写 —— 先跑 full 只有 full 的行，再跑 donly/ponly 会自动并进来。
    try:
        summary = generate_report.aggregate_all_modes(state_dir, model_name=model_name)
        if summary:
            print(f"  结论表 → {summary}   (full + donly + ponly 各一行/场景, 对标用)")
            print(f"  全测点 → {state_dir / f'metax_report_{model_name}.csv'}"
                  f"   (每个跑过的测试点一行, 带「满足要求」标记)")
    except Exception as e:
        print(f"  [警告] 汇总 CSV 生成失败: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
