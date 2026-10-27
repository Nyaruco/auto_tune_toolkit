#!/usr/bin/env python3
"""
Runner —— 执行单次测试并解析结果。

压测 client 由「场景的 client 字段 + mode」共同决定（见 client_for）:

  ais_bench —— 读场景 jsonl:
      export AIS_PATH / AIS_MODEL / AIS_MAX_OUT_LEN / AIS_NUM_PROMPTS /
             AIS_BATCH_SIZE / AIS_REQUEST_RATE
      ais_bench --models vllm_api_stream_chat \
                --custom-dataset-path <dataset> \
                --custom-dataset-data-type qa \
                --custom-dataset-infer-method gen \
                -m perf --debug

  sglang.bench_serving —— random 数据集（不读文件），与 ZYD/d-only-sla.sh、
      p-only-sla.sh 的命令一致:
      <python> -m sglang.bench_serving --model <模型路径> --backend sglang \
             --dataset-name random --num-prompts <batch×倍数> \
             --max-concurrency <batch> --request-rate <rate|inf> \
             --random-input-len <input_len> --random-output-len <out_len> \
             --random-range-ratio 1 --dataset-path <ShareGPT json>

mode 决定考核哪一轴、以及 rate 要不要打满:
  full    TTFT + TPOT 都考核，rate 是搜索变量
  donly   只考核 TPOT，rate 打满；加 --fake-prefill（prefill 由 P 侧完成）
  ponly   只考核 TTFT，rate 打满；输出长度压到 1
  ponly_cached
          共享前缀 cache 命中场景：只考核 TTFT，rate 参与搜索；
          使用 generated-shared-prefix 数据集，每次测试使用唯一 seed
  full_cached
          TTFT + TPOT 都考核，使用 generated-shared-prefix 数据集，
          每次测试使用唯一 seed；场景配置与 ponly_cached 相同
  不考核的那一轴等价于 SLO 设为"不限制"。

  「考核哪一轴」(mode) 与「用哪个 client」是两件独立的事：full 同样可以用
  bench_serving（改 scenes.py 的 client 字段，或命令行 --client 整体覆盖）。
  donly / ponly 固定用 bench_serving —— 它们要靠 --request-rate inf 把 rate 打满，
  ais_bench 给不了这个语义。

时延统计量:
  判定 SLO 用哪个统计量由 StatCfg 决定（CLI: --latency-stat，可再按轴覆盖）。
  两个 client 打印的统计量不同（ais_bench 无 P50/P95；bench_serving 的 P95 要打过
  补丁的那份 —— 见 CLIENT_LATENCY_STATS / CLIENT_STAT_FIX），
  选了打不出来的统计量会在开跑前直接报错并给出改 client 的位置 —— 见 CLIENT_STAT_FIX。
  解析时把日志里所有统计量一股脑存进 *_ms 字典，判定时按需取。

HOST_IP / HOST_PORT 通过 PD 部署的外部入口访问。本脚本不管理 PD 部署（假设已由部署侧
就绪），PD 节点数 / 卡数 / 机器数只作为元数据记录进结果。
"""

from __future__ import annotations

import math
import os
import re
import select
import secrets
import subprocess
import time
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Optional

from scenes import (MODEL, BENCH_DATASET,
                    CLIENT_AIS, CLIENT_BENCH, normalize_client)


# ====================================================================================
# 测试模式
# ====================================================================================
MODE_FULL  = "full"
MODE_DONLY = "donly"
MODE_PONLY = "ponly"
MODE_PONLY_CACHED = "ponly_cached"
MODE_FULL_CACHED = "full_cached"
CACHED_MODES = (MODE_PONLY_CACHED, MODE_FULL_CACHED)
MODE_CHOICES = (MODE_FULL, MODE_DONLY, MODE_PONLY, MODE_PONLY_CACHED,
                MODE_FULL_CACHED)

# 只描述「这个模式考核哪一轴」，不写 rate 打不打满 —— 那取决于本次跑法
# (--rate-search / --no-rate-search)，由 saturates_run 判，各处按它单独打印。
MODE_DESC = {
    MODE_FULL:  "完整考核 (TTFT + TPOT)",
    MODE_DONLY: "D only — 只考核 TPOT",
    MODE_PONLY: "P only — 只考核 TTFT (输出长度=1)",
    MODE_PONLY_CACHED: "P only cached — 只考核 TTFT (共享前缀 cache 命中场景)",
    MODE_FULL_CACHED: "Full cached — TTFT + TPOT 都考核 (共享前缀 cache 命中场景)",
}

PONLY_OUT_LEN = 1          # P only 模式把输出长度压到 1，只测 prefill
PONLY_CACHED_DATASET = "generated-shared-prefix"

# num_prompts = batch × 该倍数。两种 client 用同一个倍率（--prompts-mult 可改）。
DEFAULT_PROMPTS_MULT = 5

# 长输入档：input_len **严格大于**这个值时改用 DEFAULT_PROMPTS_MULT_LONG 倍（--long-input-mult）。
# 长输入单条请求慢（32K 一条 ~2s），× 5 会把一个测点拖成几十分钟，多的几条对分位数没有
# 额外信息量。正好 16K 不算长输入。
DEFAULT_LONG_INPUT_LEN    = 16384
DEFAULT_PROMPTS_MULT_LONG = 2

# 限速测点的目标时长(秒) —— 见 num_prompts_for。--rate-target-s 可改。
DEFAULT_RATE_TARGET_S = 180.0
# 限速测点的条数下限（同上）。低 rate 的场景按 180s 折算只有几十条，分位数判定不可信。
RATE_TARGET_MIN_PROMPTS = 100

# "打满" request rate：AIS_REQUEST_RATE < 0.001 时 ais_bench 不加任何 pacing
# (tasks/utils.py: "sending all requests simultaneously")，请求一次性发出，
# 实际负载完全由 AIS_BATCH_SIZE(=最大并发) 决定 —— 等价于 sglang bench_serving 的
# `--max-concurrency B --request-rate inf`。rate 打满的跑法见 saturates_run。
SATURATE_RATE  = 0.0
SATURATE_LABEL = "打满"
BENCH_RATE_INF = "inf"     # bench_serving 的"不限速"写法 (打满)


# ====================================================================================
# 时延统计量（--latency-stat / --ttft-stat / --tpot-stat）
# ====================================================================================
# 判定 SLO 时拿哪个统计量去比：mean / median / p50 / p90 / p95 / p99。
# 注意两个 client 打出来的统计量并不一样（见下），选了它打不出来的统计量会直接报错，
# 而不是悄悄退回 p90 —— 悄悄换口径会让两次实验的数字没法比。
LATENCY_STATS        = ("mean", "median", "p50", "p90", "p95", "p99")
DEFAULT_LATENCY_STAT = "p90"

STAT_LABEL = {
    "mean": "Mean", "median": "Median",
    "p50": "P50", "p90": "P90", "p95": "P95", "p99": "P99",
}

# 每个 client 实际会打印的统计量（以真实日志为准，不是"理论上能算"）
#   ais_bench      : 报表列 Average/Min/Max/Median/P75/P90/P99/N → 无 P50、无 P95
#   bench_serving  : TTFT/TPOT 打 Mean/Median/P50/P90/P95/P99（上游只到 P99，P95 是
#                    本目录这份 bench_serving.py 补的，见 CLIENT_STAT_FIX）→ 全有
CLIENT_LATENCY_STATS = {
    "ais_bench":           ("mean", "median", "p90", "p99"),
    "sglang.bench_serving": ("mean", "median", "p50", "p90", "p95", "p99"),
}

# 缺统计量时给用户的修复线索（报错信息里用）
CLIENT_STAT_FIX = {
    "ais_bench":
        "ais_bench 的 perf 报表列是固定的 Average/Min/Max/Median/P75/P90/P99/N。\n"
        "     要么换统计量（--latency-stat p90 / p99 / mean / median），\n"
        "     要么去 ais_bench 安装目录改它打印这张表的代码，把缺的列补上 ——\n"
        "     补完还要在 runner.py 的 CLIENT_LATENCY_STATS / AIS_COL_TO_STAT 里\n"
        "     登记这个统计量，否则开跑前仍会被这里拦下。",
    "sglang.bench_serving":
        "上游的 sglang bench_serving 只打 Mean/Median/P50/P90/P99（P95 只给 ITL）——\n"
        "     TTFT/TPOT 的 P95 是本目录这份 bench_serving.py 补上去的。机器上还是上游那份\n"
        "     就会缺 P95，把本目录的 bench_serving.py 覆盖到 <sglang 安装目录>/bench_serving.py\n"
        "     （即 sglang/bench_serving.py）再重跑。\n"
        "     不想打补丁就只能换统计量（--latency-stat p90 / p99 / mean / median）。",
}


def normalize_stat(s: str) -> str:
    """把用户写法归一化成内部名: P90/p90/90 → p90；avg/average → mean。"""
    t = (s or "").strip().lower()
    if t in ("avg", "average"):
        return "mean"
    if t == "中位数":
        return "median"
    if t.isdigit():
        return "p" + t
    return t


def stat_label(stat: str) -> str:
    return STAT_LABEL.get(stat, stat.upper())


@dataclass(frozen=True)
class StatCfg:
    """本次跑用哪个统计量判定 SLO（TTFT / TPOT 可分别指定）。

    --latency-stat 一个开关同时管两轴，--ttft-stat / --tpot-stat 单独覆盖其中一轴。
    """
    ttft: str = DEFAULT_LATENCY_STAT
    tpot: str = DEFAULT_LATENCY_STAT

    @classmethod
    def resolve(cls, latency_stat: Optional[str] = None,
                ttft_stat: Optional[str] = None,
                tpot_stat: Optional[str] = None) -> "StatCfg":
        base = normalize_stat(latency_stat or DEFAULT_LATENCY_STAT)
        return cls(ttft=normalize_stat(ttft_stat) if ttft_stat else base,
                   tpot=normalize_stat(tpot_stat) if tpot_stat else base)

    def __post_init__(self):
        for axis in ("ttft", "tpot"):
            v = getattr(self, axis)
            if v not in LATENCY_STATS:
                raise ValueError(
                    f"未知统计量 {v!r}（{axis}）—— 可选: {', '.join(LATENCY_STATS)}")

    def label(self) -> str:
        if self.ttft == self.tpot:
            return stat_label(self.ttft)
        return f"TTFT {stat_label(self.ttft)} / TPOT {stat_label(self.tpot)}"

    def as_dict(self) -> dict:
        return {"ttft": self.ttft, "tpot": self.tpot}

    def axes(self) -> list:
        """本次真正用到的 (轴, 统计量)，用于能力校验"""
        return [("ttft", self.ttft), ("tpot", self.tpot)]

    def unsupported(self, client: str) -> list:
        """该 client 打不出来的 (轴, 统计量) 列表（空 = 都能拿到）。"""
        avail = CLIENT_LATENCY_STATS.get(client, ())
        return [(ax, st) for ax, st in self.axes() if st not in avail]

    def check_supported(self, client: str) -> Optional[str]:
        """返回错误说明（None = 支持）。开跑前调用，避免白跑一轮几十分钟。"""
        bad = self.unsupported(client)
        if not bad:
            return None
        have = " / ".join(stat_label(s) for s in CLIENT_LATENCY_STATS.get(client, ()))
        items = "、".join(f"{ax.upper()} 的 {stat_label(st)}" for ax, st in bad)
        return (f"用 {client} 时日志里没有 {items}（它只打 {have}）。\n"
                f"     {CLIENT_STAT_FIX.get(client, '')}")


def default_stat(cfg: Optional["StatCfg"]) -> "StatCfg":
    return cfg or StatCfg()


def stat_missing_hint(client: str, stat: "StatCfg", axes: list) -> str:
    """跑完却发现日志里没有要的统计量时的修复提示（axes = 缺的轴名列表）。"""
    have = " / ".join(stat_label(s) for s in CLIENT_LATENCY_STATS.get(client, ()))
    items = "、".join(f"{ax.upper()} 的 {stat_label(getattr(stat, ax))}" for ax in axes)
    opt = " ".join(f"--{ax}-stat p90" for ax in axes)
    return (f"日志里找不到 {items}（{client} 只打 {have}）\n"
            f"       处理: 换统计量（{opt}），或 {CLIENT_STAT_FIX.get(client, '')}")


def mode_axes(mode: str) -> tuple[bool, bool]:
    """返回该模式考核哪两轴: (是否考核 TTFT, 是否考核 TPOT)。"""
    return (mode != MODE_DONLY, mode not in (MODE_PONLY, MODE_PONLY_CACHED))


def is_cached_mode(mode: str) -> bool:
    """是否为 generated-shared-prefix 的 cache 命中模式。"""
    return mode in CACHED_MODES


def mode_saturates_rate(mode: str) -> bool:
    """**模式默认**要不要打满 request rate：donly / ponly 默认打满，full 默认搜。

    这是模式自带的默认值，不是本次跑法的最终判定 —— donly/ponly 加 --rate-search 就
    改成跟 full 一样搜 rate，full 加 --no-rate-search 就改成打满。要判"本次跑法"用
    saturates_run，别直接用它。
    """
    return mode in (MODE_DONLY, MODE_PONLY)


def saturates_run(mode: str, *, no_rate_search: bool = False,
                  rate_search: bool = False) -> bool:
    """**本次跑法**的 request rate 是不是打满（客户端收到 inf、rate 轴不参与搜索）。

    donly/ponly 默认打满，给了 rate_search 就改成跟 full 一样搜；full 默认搜，给了
    no_rate_search 才打满。对"本来就这样"的那一侧开关是空操作（full + rate_search、
    donly + no_rate_search 都等于没加）—— 归一化放在这儿，避免同一个开关按模式
    在两处算出不同结论。

    两个开关同时给（CLI 上互斥，这里是兜底）：rate_search 赢 —— 它表达的是更具体的
    那个意图（"这个模式也去搜 rate"），而 no_rate_search 对打满模式本来就是空操作。
    """
    if mode_saturates_rate(mode):
        return not rate_search
    return bool(no_rate_search)


def rate_is_saturated(mode: str, rate: float, *, saturate: Optional[bool] = None) -> bool:
    """**这一个点**是不是「打满」。

    saturate 是本次跑法（saturates_run 的结果）；不传就按模式默认判。传它是因为
    donly/ponly + --rate-search 下模式默认会给出错误答案。

    打满只在两种情况下发生：本次跑法不搜 rate（客户端收到 inf），或 rate 取到哨兵值
    SATURATE_RATE —— 后者是搜 rate 的跑法里第一段倍增上探、以及第二段定不出有限 rate
    的退回点走的路。

    rate=0.0 是「打满」的哨兵，**不是**「每秒 0 个请求」：传给 bench_serving 必须翻成
    inf，否则它内部算 1.0/request_rate 会直接 ZeroDivisionError（full + bench_serving
    的场景在第一段上探就会撞上）。
    """
    if saturate is None:
        saturate = saturates_run(mode)
    return bool(saturate) or rate <= 0


def client_for(scene: dict, mode: str) -> str:
    """本场景在本模式下用哪个压测 client。

    donly / ponly 固定 bench_serving（要用它的 --request-rate / --fake-prefill /
    输出长度控制，ais_bench 没有这些语义）；full 按场景的 client 字段（缺省 ais_bench）。
    跟搜不搜 rate 无关 —— ponly + --rate-search 换个 rate 值照样是 bench_serving。
    """
    if mode in (MODE_DONLY, MODE_PONLY) or is_cached_mode(mode):
        return CLIENT_BENCH
    return normalize_client(scene.get("client") or CLIENT_AIS)


# ── bench_serving 的路径配置（这几个路径得按机器实际情况设置）──
@dataclass
class BenchClientCfg:
    """donly / ponly 跑 `python -m sglang.bench_serving` 需要的路径。

    model   : --model，模型目录（bench_serving 要用它加载 tokenizer 算 token 数）。
              默认 scenes.MODEL。
    python  : 用哪个解释器跑 `-m sglang.bench_serving`（装 sglang 的那个环境）。
              默认取环境变量 BENCH_PYTHON，否则 python3。
    dataset : --dataset-path，默认 scenes.BENCH_DATASET（本机那份 ShareGPT json）。
              虽然 --dataset-name random 是生成式请求，但这个版本的 random loader
              没拿到 --dataset-path 就会去 HuggingFace 下载同一个文件，离线机器直接
              报 '[Errno 101] Network is unreachable' —— 所以固定传本地路径。
              传空串可以关掉（回到"让它自己去下"的行为）。
    """
    model:   str = MODEL
    python:  str = os.environ.get("BENCH_PYTHON", "python3")
    dataset: str = BENCH_DATASET

    def model_or_default(self) -> str:
        return self.model or MODEL


def mode_out_len(scene: dict, mode: str) -> int:
    """该模式下实际使用的 max_out_len。

    P only 只测预取 → 输出长度 1。ponly_cached 的 GSP 数据集可以单独指定
    output_len；没指定时退回场景的 max_out_len。
    """
    if mode == MODE_PONLY:
        return PONLY_OUT_LEN
    if is_cached_mode(mode):
        value = scene.get("output_len", scene.get("gsp_output_len"))
        return int(value if value is not None else scene["max_out_len"])
    return scene["max_out_len"]


def cache_input_lens(scene: dict) -> tuple[int, int]:
    """ponly_cached 的 (cache 命中长度, 非命中长度)。

    兼容用户配置里的拼写 ``none_cached_input_len``；优先使用更明确的
    ``non_cached_input_len``。
    """
    cached = scene.get("cached_input_len")
    noncached = scene.get("non_cached_input_len")
    if noncached is None:
        noncached = scene.get("none_cached_input_len")
    if cached is None or noncached is None:
        raise ValueError(
            f"场景 {scene.get('key', '?')} 的 ponly_cached 模式需要 "
            "cached_input_len 和 non_cached_input_len"
        )
    cached, noncached = int(cached), int(noncached)
    if cached < 0 or noncached < 0 or cached + noncached <= 0:
        raise ValueError(
            f"场景 {scene.get('key', '?')} 的 cache 长度非法: "
            f"cached_input_len={cached}, non_cached_input_len={noncached}"
        )
    return cached, noncached


def cache_hit_ratio(scene: dict) -> float:
    cached, noncached = cache_input_lens(scene)
    return cached / max(1, cached + noncached)


def mode_missing_slo(scene: dict, mode: str) -> Optional[str]:
    """退化检测：返回"该模式唯一要考核、但场景并没定义 SLO"的轴名，否则 None。

    donly / ponly 都只考核一个轴，所以一旦那一轴没定义 SLO，就没有任何时延约束 ——
    所有测试点都算达标，搜索会一路顶到上界（退化成"纯求最大吞吐"）。
    full 模式不报此警告：它至少还考核另一轴，且"某轴不限制"本来就是各场景的设计
    （agent 不限 TPOT、ultra_long 两者都不限）。
    """
    if mode in (MODE_FULL, MODE_FULL_CACHED):
        return None
    axis = "ttft" if mode in (MODE_PONLY, MODE_PONLY_CACHED) else "tpot"
    lim = scene.get("slo_ttft" if axis == "ttft" else "slo_tpot", -1)
    return axis if (lim is None or lim <= 0) else None



# ====================================================================================
# 结果数据结构
# ====================================================================================
@dataclass
class BenchResult:
    # 入参
    scene_key:  str
    batch:      int
    rate:       float
    num_prompts: int
    mode:       str = "full"          # full / donly / ponly
    max_out_len: int = 0              # 实际使用的输出长度 (ponly=1)
    host_ip:    str = ""
    host_port:  str = ""
    # 指标（解析自 client 输出）
    #   时延按统计量存字典: {"mean": .., "median": .., "p50": .., "p90": .., "p99": ..}
    #   —— 缺哪个统计量就没那个键（ais_bench 没有 p50/p95；bench_serving 打全了）
    ttft_ms:         dict = field(default_factory=dict)
    tpot_ms:         dict = field(default_factory=dict)
    e2el_ms:         dict = field(default_factory=dict)
    req_throughput:  Optional[float] = None     # req/s
    out_tok_thr:     Optional[float] = None     # Output Token Throughput (集群)
    in_tok_thr:      Optional[float] = None     # Input Token Throughput (集群, ponly 排序用)
    total_tok_thr:   Optional[float] = None     # Total Token Throughput (集群)
    failed_requests: int = -1
    success_requests: int = -1
    max_concurrency: Optional[float] = None
    seed: Optional[int] = None          # ponly_cached 每次测试使用的新 seed
    # 运行状态
    elapsed_s:   float = 0.0
    log_path:    str = ""
    parse_ok:    bool = False
    error:       str = ""
    # 主导因素（调试用）
    note:        str = ""

    # ── 统计量访问 ────────────────────────────────────────────────────────────
    def lat_ms(self, axis: str, stat: str = DEFAULT_LATENCY_STAT) -> Optional[float]:
        """取某轴某个统计量的时延(ms)。axis: ttft / tpot / e2el。"""
        return lat_ms_from(getattr(self, f"{axis}_ms", None), stat)

    def lat_str(self, axis: str) -> str:
        """把该轴**所有**统计量拼成一行，方便控制台核对（没有的统计量不显示）。"""
        d = getattr(self, f"{axis}_ms", None) or {}
        if not d:
            return "N/A"
        return " ".join(f"{stat_label(s)}={d[s]:.2f}" for s in LATENCY_STATS if s in d) + " ms"

    @classmethod
    def from_dict(cls, d: dict, **overrides) -> "BenchResult":
        """从 state / report 的条目还原。

        兼容老数据：老的 state/report 里没有 ttft_ms 字典，只有平铺的 *_p90_ms
        —— 这里把它转成 {"p90": ...}，于是"按统计量取数"的代码不用分新旧。
        """
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        for axis in ("ttft", "tpot", "e2el"):
            lat = dict(kw.get(f"{axis}_ms") or {})
            old = d.get(f"{axis}_p90_ms")
            if old is not None and "p90" not in lat:
                lat["p90"] = old
            kw[f"{axis}_ms"] = lat
        kw.update(overrides)
        return cls(**kw)


def lat_ms_from(lat: Optional[dict], stat: str = DEFAULT_LATENCY_STAT) -> Optional[float]:
    """从一个 {统计量: 时延} 字典里取值。"""
    if not lat:
        return None
    v = lat.get(stat)
    return float(v) if v is not None else None


def entry_lat_ms(entry: dict, axis: str, stat: str = DEFAULT_LATENCY_STAT) -> Optional[float]:
    """从 state / report 里的**普通 dict** 条目取时延(ms)。

    兼容老数据：老条目没有 {axis}_ms 字典，只有 *_p90_ms 平铺字段
    （两处判断都在，读老 state 生成的报表才不会一片空白）。
    """
    lat = entry.get(f"{axis}_ms")
    if lat:
        v = lat.get(stat)
        if v is not None:
            return float(v)
    if stat == DEFAULT_LATENCY_STAT:
        v = entry.get(f"{axis}_p90_ms")
        if v is not None:
            return float(v)
    return None


def entry_stat_of(entry: dict) -> StatCfg:
    """条目所属的统计量配置（老条目没有这个字段 → p90，与老数据的口径一致）。"""
    st = entry.get("latency_stat") or {}
    return StatCfg(ttft=st.get("ttft", DEFAULT_LATENCY_STAT),
                   tpot=st.get("tpot", DEFAULT_LATENCY_STAT))


# ====================================================================================
# 解析 ais_bench 输出
# ====================================================================================
# 实测的报表长这样（取自真实日志）：
#   │ Performance Parameters   │ Stage   │ Average  │ Min  │ Max  │ Median │ P75 │ P90 │ P99 │ N │
#   │ TTFT                     │ total   │ 7575.8 ms│ 7108.4 ms │ …                        │ 96 │
# 列名 → 统计量：Average→mean、Median→median、P90→p90、P99→p99
# （Min/Max/P75/N 不参与判定，跳过）
# 按**列名**取数，不按位置 —— 这样 client 换版本多加一列也不会整体错位。
AIS_COL_TO_STAT = {"Average": "mean", "Median": "median", "P90": "p90", "P99": "p99"}
# 拿不到表头时的兜底列序（老版本的固定 8 列）
AIS_DEFAULT_COLS = ["Average", "Min", "Max", "Median", "P75", "P90", "P99", "N"]

_AIS_HEADER_RE = re.compile(r"│\s*Performance Parameters\s*│(.*?)│\s*$", re.MULTILINE)


def _ais_header_cells(text: str, before: int = -1) -> list:
    """找 <before> 之前最近的一次报表表头，返回**与数据行 body 对齐**的列名列表。

    数据行的 body 从 "total" 之后开始（_scan_row_at 已吃掉 [指标名, total]），
    对应表头里 Average 起的那 8 列；表头正则又已经吃掉了 "Performance Parameters"
    这一格。所以这里把剩下的首格 Stage 也去掉，两边就严格对齐：body 第 i 列 = 第 i 项。

    before < 0 表示取第一个表头。找不到返回 []。
    """
    best = None
    for m in _AIS_HEADER_RE.finditer(text):
        if before >= 0 and m.start() > before:
            break
        best = m
    if best is None:
        return []
    cells = [c.strip() for c in best.group(1).split("│")]
    return cells[1:] if cells and cells[0] == "Stage" else cells


def _scan_row_at(text: str, name: str, unit: str = "ms"):
    """匹配 "│ <name> │ total │ … │" 那一行。

    返回 (行首偏移, body)，body = "│ total │" 之后的整段（按 │ 切开就是各列）。
    例如 TTFT 行的各列是 [7575.8 ms, 7108.4 ms, …]。
    """
    pat = re.compile(
        r"│\s*" + re.escape(name) + r"\s*│\s*total\s*│(.*?)│\s*$",
        re.MULTILINE,
    )
    m = pat.search(text)
    if not m:
        return None
    return m.start(), m.group(1)


def _scan_row(text: str, name: str, unit: str = "ms") -> Optional[list]:
    """返回该行 total 下所有数值列表（按出现顺序）。若不匹配返回 None。"""
    hit = _scan_row_at(text, name, unit)
    if not hit:
        return None
    _, body = hit
    if unit == "ms":
        # 收集所有形如 "<数字> ms" 的部分
        nums = re.findall(r"([\d.]+)\s*ms", body)
    elif unit == "req/s":
        nums = re.findall(r"([\d.]+)\s*req/s", body)
    elif unit == "token/s":
        nums = re.findall(r"([\d.]+)\s*token/s", body)
    else:
        nums = re.findall(r"([\d.]+)", body)
    nums = [float(n) for n in nums]
    return nums or None


def parse_bench_log(text: str) -> dict:
    """从 ais_bench 完整 stdout 文本中解析关键指标。

    时延存成 {统计量: 数值} 字典（键见 AIS_COL_TO_STAT）—— client 没打的统计量
    （P50/P95）字典里就没有，判定时按需取，取不到由 StatCfg.check_supported 兜底报错。
    """
    res: dict = {}

    for name, axis in (("TTFT", "ttft"), ("TPOT", "tpot"), ("E2EL", "e2el")):
        hit = _scan_row_at(text, name, "ms")
        if not hit:
            continue
        offset, body = hit
        cols = _ais_header_cells(text, offset) or AIS_DEFAULT_COLS
        cells = [c.strip() for c in body.split("│")]
        lat = res.setdefault(f"{axis}_ms", {})
        for i, cell in enumerate(cells):
            col = cols[i] if i < len(cols) else ""
            stat = AIS_COL_TO_STAT.get(col)
            if not stat:
                continue
            m = re.search(r"([\d.]+)", cell)
            if m:
                lat[stat] = float(m.group(1))

    # 单行汇总类指标
    def _scan_unit(name: str, unit: str) -> Optional[float]:
        nums = _scan_row(text, name, unit)
        if nums:
            return float(nums[0])
        return None

    res["req_throughput"]   = _scan_unit("Request Throughput",  "req/s")
    res["out_tok_thr"]      = _scan_unit("Output Token Throughput",  "token/s")
    res["total_tok_thr"]    = _scan_unit("Total Token Throughput",  "token/s")
    # OutputTokenThroughput (无空格的别名) 兼容
    if res["out_tok_thr"] is None:
        res["out_tok_thr"] = _scan_unit("OutputTokenThroughput", "token/s")

    # Total / Failed / Success / Max Concurrency
    m = re.search(r"Failed Requests\s*│\s*total\s*│\s*(\d+)", text)
    if m: res["failed_requests"] = int(m.group(1))
    m = re.search(r"Success Requests\s*│\s*total\s*│\s*(\d+)", text)
    if m: res["success_requests"] = int(m.group(1))
    m = re.search(r"Max Concurrency\s*│\s*total\s*│\s*([\d.]+)", text)
    if m: res["max_concurrency"] = float(m.group(1))

    return res


# ====================================================================================
# 解析 sglang bench_serving 输出（donly / ponly）
# ====================================================================================
# 结果块形如（左侧标签左对齐补齐 40 列，右侧数值补齐 10 列）：
#   ============ Serving Benchmark Result ============
#   Max request concurrency:                 16
#   Successful requests:                     80
#   Request throughput (req/s):              0.19
#   Output token throughput (tok/s):         37.86
#   Total token throughput (tok/s):          7791.60
#   P90 E2E Latency (ms):                    82650.31
#   P90 TTFT (ms):                           77539.91
#   P95 TTFT (ms):                           80312.44
#   P90 TPOT (ms):                           31.99
#   P95 TPOT (ms):                           33.07
# 参考 d-only-sla.sh 用的也是同样的匹配（grep 'P90 TPOT (ms)': ... 取 tail -1）。
#
# 时延不在这里按统计量写死 —— bench_serving 打的是 Mean/Median/P50/P90/P95/P99
# （TTFT、TPOT 都一样；上游没有 P95，是本目录那份补的），所以用一个正则一次抓全。
_BS_LAT_RE = re.compile(
    r"^(Mean|Median|P50|P90|P95|P99)\s+(TTFT|TPOT|E2E Latency)\s+\(ms\):\s*([\d.]+)",
    re.MULTILINE,
)
_BS_AXIS = {"TTFT": "ttft", "TPOT": "tpot", "E2E Latency": "e2el"}

_BS_SCALARS = {
    "req_throughput": r"^Request throughput \(req/s\):\s*([\d.]+)",
    "out_tok_thr":    r"^Output token throughput \(tok/s\):\s*([\d.]+)",
    "in_tok_thr":     r"^Input token throughput \(tok/s\):\s*([\d.]+)",
    "total_tok_thr":  r"^Total token throughput \(tok/s\):\s*([\d.]+)",
    "max_concurrency": r"^Max request concurrency:\s*([\d.]+)",
}


def parse_bench_serving_log(text: str, num_prompts: Optional[int] = None) -> dict:
    """从 bench_serving 完整输出中解析关键指标。

    num_prompts: 本次发出的请求数，用来兜底算失败请求数
                 （老版本只打印 Successful requests，没有 Failed requests 行）。
    """
    res: dict = {}
    for key, pat in _BS_SCALARS.items():
        # 取最后一次出现（一个日志文件里正常只有一次测试；万一追加了多次，
        # 也要跟参考脚本的 `grep ... | tail -1` 一样以最后一次为准）
        vals = re.findall(pat, text, re.MULTILINE)
        if vals:
            res[key] = float(vals[-1])

    # 时延：一次抓全所有统计量，按轴分桶。后出现的块覆盖前面的（同 tail -1 口径）
    for stat_w, axis_w, val in _BS_LAT_RE.findall(text):
        res.setdefault(f"{_BS_AXIS[axis_w]}_ms", {})[stat_w.lower()] = float(val)

    m = re.search(r"^Failed requests:\s*(\d+)", text, re.MULTILINE)
    if m:
        res["failed_requests"] = int(m.group(1))
    m = re.search(r"^Successful requests:\s*(\d+)", text, re.MULTILINE)
    if m:
        ok = int(m.group(1))
        res["success_requests"] = ok
        if "failed_requests" not in res and num_prompts:
            # 没有 Failed 行时按 "发出 - 成功" 推：注意 bench_serving 的 warmup 请求
            # 不计入 Successful，也不计入 num_prompts，所以这个差值是可信的。
            res["failed_requests"] = max(0, int(num_prompts) - ok)
    return res


# ====================================================================================
# 构造 sglang bench_serving 命令（donly / ponly）
# ====================================================================================
def new_cache_seed() -> int:
    """每次 benchmark 使用独立 seed。

    generated-shared-prefix 会按 seed 缓存生成的 prompt；如果每次复用同一个 seed，
    同一组前缀会再次生成并直接命中服务端 cache，测不到真实的 prefill 压力。
    """
    return secrets.randbelow(2_147_483_646) + 1


def build_bench_serving_cmd(scene: dict, batch: int, rate: float, mode: str, *,
                            host_ip: str, host_port: str,
                            cfg: Optional[BenchClientCfg] = None,
                            num_prompts: Optional[int] = None,
                            prompts_mult: int = DEFAULT_PROMPTS_MULT,
                            rate_target_s: float = DEFAULT_RATE_TARGET_S,
                            saturate: Optional[bool] = None,
                            seed: Optional[int] = None) -> list[str]:
    """按 ZYD/d-only-sla.sh、p-only-sla.sh 的写法拼命令。

    rate: 打满的点翻成 `--request-rate inf`（请求一次性发出），限速的点原样传 —— 语义
    跟 ais_bench 的 AIS_REQUEST_RATE 一致。哪些点算打满见 rate_is_saturated：本次跑法
    不搜 rate（donly/ponly 的默认，或 full + --no-rate-search）、或者 rate 取哨兵值
    SATURATE_RATE=0.0（搜 rate 的跑法里第一段上探和第二段的退回点）都算。

    saturate: 本次跑法的打满判定（saturates_run 的结果）。不传按模式默认 —— 传它是因为
    donly/ponly + --rate-search 下模式默认会给出错误答案。

    num_prompts: 调用方（run_one）算好传进来，保证它跟日志头/ais_bench 那侧是同一个数。
    不传才按 prompts_mult / rate_target_s 现算（给直接调本函数的测试用）。

    普通 ponly/donly 请求由 --dataset-name random 生成（不读场景 jsonl），但
    --dataset-path 仍要传：random loader 缺它就会去 HuggingFace 下载 ShareGPT json，
    离线机器上会直接失败。

    ponly_cached 改为 --dataset-name generated-shared-prefix：
      --gsp-system-prompt-len  = cached_input_len（cache 命中部分）
      --gsp-question-len       = non_cached_input_len（未命中部分）
      --gsp-prompts-per-group  = num_prompts（num_groups=1）
    每次调用必须带不同的 seed，否则生成内容重复且可能全部命中服务端 cache。
    """
    cfg = cfg or BenchClientCfg()
    sat = rate_is_saturated(mode, rate, saturate=saturate)
    if num_prompts is None:
        num_prompts = num_prompts_for(batch, prompts_mult, 0.0 if sat else rate,
                                      rate_target_s)
    rate_str = BENCH_RATE_INF if sat else str(rate)

    if is_cached_mode(mode):
        cached_len, noncached_len = cache_input_lens(scene)
        if seed is None:
            seed = new_cache_seed()
        cmd = [
            cfg.python, "-m", "sglang.bench_serving",
            "--model",                  cfg.model_or_default(),
            "--backend",                "sglang",
            "--dataset-name",           PONLY_CACHED_DATASET,
            "--max-concurrency",        str(batch),
            "--request-rate",           rate_str,
            "--gsp-num-groups",         "1",
            "--gsp-prompts-per-group",  str(num_prompts),
            "--gsp-system-prompt-len",  str(cached_len),
            "--gsp-question-len",       str(noncached_len),
            "--gsp-output-len",         str(mode_out_len(scene, mode)),
            "--gsp-range-ratio",        "1.0",
            "--seed",                   str(int(seed)),
        ]
        if host_ip:
            cmd += ["--host", host_ip]
        if host_port:
            cmd += ["--port", str(host_port)]
        return cmd

    in_len = int(scene.get("input_len", 0) or 0)
    if in_len <= 0:
        raise ValueError(f"场景 {scene['key']} 缺 input_len，bench_serving 需要它生成 input")
    cmd = [
        cfg.python, "-m", "sglang.bench_serving",
        "--model",                  cfg.model_or_default(),
        "--backend",                "sglang",
        # "--dataset-name",           "random-ids",
        "--dataset-name",           "random",
        "--num-prompts",            str(num_prompts),
        "--max-concurrency",        str(batch),
        "--request-rate",           rate_str,
        "--random-input-len",       str(in_len),
        "--random-output-len",      str(mode_out_len(scene, mode)),
        "--random-range-ratio",     "1",
    ]
    if host_ip:
        cmd += ["--host", host_ip]
    if host_port:
        cmd += ["--port", str(host_port)]
    if cfg.dataset:                       # 缺了 random loader 会去 HF 下载，离线机器会失败
        cmd += ["--dataset-path", cfg.dataset]
    if mode == MODE_DONLY:
        cmd += ["--fake-prefill"]         # prefill 由 P 侧完成 → TTFT 无参考意义
    return cmd


def is_long_input(scene: dict,
                  long_input_len: int = DEFAULT_LONG_INPUT_LEN) -> bool:
    """这个场景算不算**长输入档**：input_len **严格大于**阈值（--long-input-len）。

    严格大于 ⇒ 正好 16K 的 case 走普通档。档位只有这一个判据，prompts_mult_for 和横幅
    都调它，别各写各的比较。
    """
    return int(scene.get("input_len") or 0) > int(long_input_len)


def prompts_mult_for(scene: dict, mult: int = DEFAULT_PROMPTS_MULT,
                     long_mult: int = DEFAULT_PROMPTS_MULT_LONG,
                     long_input_len: int = DEFAULT_LONG_INPUT_LEN) -> int:
    """本场景该用哪个 num_prompts 倍率 —— 长输入档还是普通档（判据见 is_long_input）。

    在 search() 里**解析一次**，往下只传解析结果：run_one / SearchContext / 测量指纹都不
    需要知道"倍率分两档"，也就不会两处各解析一遍对不上。三种模式共用（mode 只决定输出
    长度，跟发多少条无关）。
    """
    return int(long_mult) if is_long_input(scene, long_input_len) else int(mult)


def num_prompts_for(batch: int, mult: int = DEFAULT_PROMPTS_MULT,
                    rate: float = 0.0,
                    target_s: float = DEFAULT_RATE_TARGET_S) -> int:
    """本次发出的请求数（两种 client 统一）。

    mult 传的是**已经解析过的**倍率（--prompts-mult，长输入场景是 --long-input-mult，
    见 prompts_mult_for）—— 这里不再自己判输入长度。

    打满（rate <= 0）：batch × 倍率 —— 时长 ≈ 倍率 × e2e，跟 batch 无关，合适。

    限速：再按 batch × 倍率走的话，时长 = batch × 倍率 / rate —— **并发越大跑越久**，
    可限速下并发只要够接住这个 rate 就不是瓶颈了 —— 判据是实测 req/s 有没有跟上
    请求的 rate（`achieved_rate`），跟不上的点实际已退化成打满点。多跑的时间一条
    新信息都买不到。0917 的 16K 就是这么烧的：batch=89 的限速点 422s，
    batch=256 的同一个工况 1134s。
    所以改按目标时长给，并且**取与 batch×倍率的小值** —— 只可能更短，不会把原本就
    短的点拉长（rate 高于容量时按 rate 算会反而更长，那时实测根本跑不到这个 rate）。

    条数有下限 RATE_TARGET_MIN_PROMPTS：目标时长折算出来的条数太少时，判定用的
    分位数（p90/p95/p99）在几十个样本上基本等于最大值，"达标/不达标"就成了掷骰子。
    """
    sat = max(int(batch) * int(mult), int(mult))
    if rate > 0 and target_s > 0:
        want = max(int(math.ceil(rate * target_s)), RATE_TARGET_MIN_PROMPTS)
        return max(1, min(sat, want))
    return sat


# ====================================================================================
# 实时转发子进程输出（bench_serving 的 tqdm 进度条靠 \r 原地刷新）
# ====================================================================================
TQDM_RE = re.compile(r"\d+%\|.*\|\s*\d+/\d+")


def _stream_bench_output(proc, log_f, show_progress: bool = True) -> None:
    """边跑边转发输出：进度条实时打在控制台（原位刷新）且不写进日志文件。

    按行读会一直憋到进程结束 —— tqdm 用 '\\r' 刷新、只在最后换行，所以这里直接在
    文件描述符上增量读，遇到 '\\r' 或 '\\n' 就切一段出来。
    """
    fd = proc.stdout.fileno()
    buf = b""
    live = 0
    while True:
        try:
            ready, _, _ = select.select([fd], [], [], 1.0)
        except (OSError, ValueError):
            break
        if not ready:
            if proc.poll() is not None:
                break
            continue
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        buf += chunk
        while True:
            cuts = [b for b in (buf.find(b"\n"), buf.find(b"\r")) if b >= 0]
            if not cuts:
                break
            i = min(cuts)
            seg, buf = buf[:i].decode("utf-8", "replace"), buf[i + 1:]
            if TQDM_RE.search(seg):                     # 进度条
                if show_progress:
                    live = max(live, len(seg))
                    print(f"\r{seg:<{live}}", end="", flush=True)
                continue
            if seg.strip():
                if live:
                    print()
                    live = 0
                print(seg, flush=True)
                log_f.write(seg + "\n")
    if buf.strip():
        log_f.write(buf.decode("utf-8", "replace") + "\n")
    if live:
        print()


# ====================================================================================
# 执行单次 benchmark
# ====================================================================================
def run_one(scene: dict, batch: int, rate: float, *,
            host_ip: str, host_port: str,
            log_dir: Path, tag: str = "",
            show_progress: bool = True,
            show_cmd: bool = True,
            mode: str = MODE_FULL,
            bench_cfg: Optional[BenchClientCfg] = None,
            stat: Optional[StatCfg] = None,
            prompts_mult: int = DEFAULT_PROMPTS_MULT,
            rate_target_s: float = DEFAULT_RATE_TARGET_S,
            saturate: Optional[bool] = None,
            seed: Optional[int] = None) -> BenchResult:
    """执行一次测试并返回解析后的 BenchResult。

    用哪个 client 由场景的 client 字段 + mode 决定（见 client_for）：ais_bench 走
    环境变量 AIS_*（读场景 jsonl），sglang.bench_serving 走命令行参数（random 数据集）。
    num_prompts 见 num_prompts_for：打满点 = batch × prompts_mult，限速点按 rate_target_s
    折算（并发越大越省，限速下并发不是瓶颈）。prompts_mult 传的是**已解析的**倍率
    （长输入场景是长输入档那个值，解析在 search() 里做，见 prompts_mult_for）。

    saturate: 本次跑法的打满判定（saturates_run 的结果，搜索侧传 ctx.saturate）。
              不传按模式默认 —— donly/ponly + --rate-search 必须传，否则限速点会被
              当成打满，rate 被翻成 inf 发出去。

    stat: 本次判定用的统计量。日志里所有统计量都会解析下来，这里只用它判断
          "本次要考核的那一轴，需要的统计量到底拿到了没有"（没拿到 → parse_ok=False）。

    show_cmd: 跑之前把完整命令行打到控制台（跟日志头里的 `# cmd:` 同一条）。
              ais_bench 的 batch/rate 走环境变量，所以前缀里一并带上 AIS_*/HOST_*，
              整行可以直接拷出来手跑。
    """
    client = client_for(scene, mode)
    bench_mode = client == CLIENT_BENCH
    bench_cfg = bench_cfg or BenchClientCfg()
    stat = default_stat(stat)

    # 这个点是不是打满 —— 本函数里只算这一次，下面 num_prompts / 日志头 / 命令拼装都用它
    sat = rate_is_saturated(mode, rate, saturate=saturate)

    # num_prompts 只在**这一处**算，再传给命令拼装 —— 两个 client、日志头、控制台必须
    # 是同一个数（以前 bench 那边又算了一遍，规则一改就打架：日志头 320、命令发 220）。
    # 打满的点恒按 batch 给：rate 传什么都不生效（会翻成 inf），拿它折算只会把点缩短。
    num_prompts = num_prompts_for(batch, prompts_mult, 0.0 if sat else rate, rate_target_s)
    max_out_len = mode_out_len(scene, mode)     # ponly → 1
    trial_seed = seed
    if is_cached_mode(mode) and trial_seed is None:
        trial_seed = new_cache_seed()

    env = os.environ.copy()
    injected: dict = {}                         # 只列我们注入的变量，用于打印命令行前缀
    if not bench_mode:
        injected = {
            "AIS_PATH":         bench_cfg.model_or_default(),
            "AIS_MODEL":        bench_cfg.model_or_default(),
            "AIS_MAX_OUT_LEN":  str(max_out_len),
            "AIS_NUM_PROMPTS":  str(num_prompts),
            "AIS_BATCH_SIZE":   str(batch),
            "AIS_REQUEST_RATE": str(rate),
        }
        if host_ip:
            injected["HOST_IP"]   = host_ip
        if host_port:
            injected["HOST_PORT"] = host_port
        env.update(injected)
        cmd = [
            "ais_bench",
            "--models",                       "vllm_api_stream_chat",
            "--custom-dataset-path",          scene["dataset"],
            "--custom-dataset-data-type",     "qa",
            "--custom-dataset-infer-method",  "gen",
            "-m", "perf", "--debug",
        ]
    else:
        cmd = build_bench_serving_cmd(scene, batch, rate, mode,
                                      host_ip=host_ip, host_port=host_port,
                                      cfg=bench_cfg, num_prompts=num_prompts,
                                      saturate=sat, seed=trial_seed)

    # full 模式的 batch/rate 走环境变量 —— 把它们并进命令行里，
    # 日志头和控制台看到的就是同一条可以原样拷出来手跑的命令
    cmd_line = "".join(f"{k}={v} " for k, v in injected.items()) + " ".join(cmd)

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_b{batch}_r{rate}_{ts}" if not tag else f"_{tag}_b{batch}_r{rate}_{ts}"
    log_path = log_dir / f"{scene['key']}{suffix}.log"

    res = BenchResult(
        scene_key=scene["key"], batch=batch, rate=rate, num_prompts=num_prompts,
        mode=mode, max_out_len=max_out_len,
        host_ip=host_ip, host_port=host_port, log_path=str(log_path),
        seed=trial_seed,
    )

    # 打满的点真实发出去的是 --request-rate inf，表头也照这个写 —— 否则会写出
    # `rate=0.0 (打满, 不限速)` 这种自相矛盾的行（0.0 只是哨兵值）。
    # ais_bench 那侧不套：0.0 本来就是它的"不限速"，表头如实写它收到的值。
    rate_str = (f"{BENCH_RATE_INF} ({SATURATE_LABEL}, 不限速)"
                if bench_mode and sat else f"{rate}")
    header = (
        f"# scene={scene['key']}  batch={batch}  rate={rate_str}  "
        f"num_prompts={num_prompts}  max_out_len={max_out_len}\n"
        f"# mode={mode} ({MODE_DESC.get(mode, mode)})  client={client}\n"
        f"# host={host_ip}:{host_port}\n"
        + (f"# seed={trial_seed} (缓存命中场景每点唯一)\n"
           if is_cached_mode(mode) else "")
        + f"# 判定统计量: {stat.label()}"
        f"{' (--latency-stat)' if stat.ttft == stat.tpot else ' (--ttft-stat/--tpot-stat)'}\n"
        f"# cmd: {cmd_line}\n"
    )
    if is_cached_mode(mode):
        cached_len, noncached_len = cache_input_lens(scene)
        header += (
            f"# cached_input_len={cached_len}  "
            f"non_cached_input_len={noncached_len}  "
            f"cache_hit_ratio={cache_hit_ratio(scene):.4f}\n"
            f"# dataset={PONLY_CACHED_DATASET}  seed={trial_seed}\n"
        )
    elif bench_mode:
        header += (
            f"# input_len={scene.get('input_len')} (random 数据集生成, 不读 {scene.get('dataset', '')})\n"
        )

    # 进度行（POST=... RECV=...）控制台每 10s 原位刷新，不写入日志
    progress_re = re.compile(r"^(POST=|RECV=|FAIL=|FIN=|\s*$)")

    # 完整命令行打到控制台（跟日志头里的 # cmd: 逐字一致），方便手动复跑 / 排查
    if show_cmd:
        print(f"  $ {cmd_line}", flush=True)

    start = time.time()
    print_new_line = False
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(header)
            if bench_mode:
                # 这里原来写死 inf —— full + bench_serving 下每个点都是有限 rate，
                # 写死会把限速点也记成"打满"。
                f.write(f"# 测试参数: max_concurrency={batch}, "
                        f"request_rate={BENCH_RATE_INF if sat else rate}"
                        f"{' (' + SATURATE_LABEL + ')' if sat else ''}"
                        f"{', fake-prefill' if mode == MODE_DONLY else ''}\n")
            else:
                f.write(f"# 测试参数: AIS_BATCH_SIZE={batch}, AIS_REQUEST_RATE={rate}\n")
            f.flush()
            if bench_mode:
                # bench_serving 的 tqdm 进度条靠 \r 原地刷新，需要按 fd 增量读
                proc = subprocess.Popen(
                    cmd, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                )
                _stream_bench_output(proc, f, show_progress=show_progress)
            else:
                proc = subprocess.Popen(
                    cmd, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    bufsize=1, text=True,
                )
                last_progress_w = 0
                for line in proc.stdout:
                    if show_progress and progress_re.match(line):
                        pass  # 不显示进度行
                    else:
                        if line.startswith("Progress:"):
                            print_new_line = True
                            s = line.rstrip()
                            last_progress_w = max(last_progress_w, len(s))
                            print(f"\r{s:{last_progress_w}}", end="", flush=True)
                        else:
                            if print_new_line:
                                print_new_line= False
                                print()
                            print(line, end="", flush=True)
                    f.write(line)
            proc.wait()
            rc = proc.returncode
        res.elapsed_s = time.time() - start

        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        parsed = (parse_bench_serving_log(text, num_prompts) if bench_mode
                  else parse_bench_log(text))

        if rc != 0:
            res.error = f"{client} 退出码 {rc}"
        res.ttft_ms         = parsed.get("ttft_ms") or {}
        res.tpot_ms         = parsed.get("tpot_ms") or {}
        res.e2el_ms         = parsed.get("e2el_ms") or {}
        res.req_throughput  = parsed.get("req_throughput")
        res.out_tok_thr     = parsed.get("out_tok_thr")
        res.in_tok_thr      = parsed.get("in_tok_thr")
        res.total_tok_thr   = parsed.get("total_tok_thr")
        res.failed_requests = parsed.get("failed_requests", -1)
        res.success_requests = parsed.get("success_requests", -1)
        res.max_concurrency = parsed.get("max_concurrency")

        # 判定是否成功解析到关键指标 —— 只要求本模式真正要用的那些:
        #   ponly 输出长度=1，TPOT 可能缺失/退化，不能拿它当必要条件
        #   donly 只考核 TPOT，TTFT 缺失不影响判定
        # 时延要求的是**本次判定用的那个统计量**（如 P95），不是笼统的"有没有 TTFT"
        check_ttft, check_tpot = mode_axes(mode)
        need = {"throughput": res.req_throughput is not None}
        if check_ttft:
            need["ttft"] = res.lat_ms("ttft", stat.ttft) is not None
        if check_tpot:
            need["tpot"] = res.lat_ms("tpot", stat.tpot) is not None
        res.parse_ok = all(need.values())
        if not res.parse_ok:
            lack = [k for k, ok in need.items() if not ok]
            res.error = "关键指标缺失 (" + "/".join(lack) + ")"
            if "throughput" in lack:
                res.error += " —— client 没打出吞吐结果块，看日志确认这次是不是真跑起来了"
            else:
                res.error += " —— " + stat_missing_hint(client, stat, lack)

    except FileNotFoundError:
        res.error = (f"未找到命令 {cmd[0]} —— " +
                     ("请用 --bench-python 指定装 sglang 的解释器（或设 BENCH_PYTHON）"
                      if bench_mode else "请确保 ais_bench 已安装并能在 PATH 中访问"))
        res.elapsed_s = time.time() - start
    except Exception as e:
        res.error = f"运行异常: {e}"
        res.elapsed_s = time.time() - start

    return res


# ====================================================================================
# SLO 判定
# ====================================================================================
def slo_check(res: BenchResult, scene: dict, *, fail_tolerant: int = 0,
              mode: str = MODE_FULL, stat: Optional[StatCfg] = None) -> dict:
    """返回 {satisfied, ttft_margin, tpot_margin, reasons[], check_ttft, check_tpot}

    mode 决定考核哪一轴：不考核的轴按"不限制"处理（lim=-1），
    于是 margin=None、不产生 reasons，search() 里对应的 violate 判定自然为 False。

    stat 决定拿哪个统计量去比（默认 P90，见 --latency-stat）。
    """
    stat = default_stat(stat)
    check_ttft, check_tpot = mode_axes(mode)
    ttft_lim = scene.get("slo_ttft", -1) if check_ttft else -1
    tpot_lim = scene.get("slo_tpot", -1) if check_tpot else -1

    reasons: list[str] = []
    ttft_ok = True
    tpot_ok = True

    ttft_ms = res.lat_ms("ttft", stat.ttft)
    tpot_ms = res.lat_ms("tpot", stat.tpot)

    if ttft_lim > 0:
        if ttft_ms is None:
            ttft_ok = False
            reasons.append(f"TTFT {stat_label(stat.ttft)} 缺失")
        else:
            ttft_s = ttft_ms / 1000.0
            ttft_ok = ttft_s < ttft_lim
            if not ttft_ok:
                reasons.append(f"TTFT {stat_label(stat.ttft)}={ttft_s:.3f}s 超出限制 {ttft_lim}s")

    if tpot_lim > 0:
        if tpot_ms is None:
            tpot_ok = False
            reasons.append(f"TPOT {stat_label(stat.tpot)} 缺失")
        else:
            tpot_s = tpot_ms / 1000.0
            tpot_ok = tpot_s < tpot_lim
            if not tpot_ok:
                reasons.append(f"TPOT {stat_label(stat.tpot)}={tpot_s*1000:.2f}ms "
                               f"超出限制 {tpot_lim*1000:.1f}ms")

    fail_ok = (res.failed_requests if res.failed_requests is not None else -1) <= fail_tolerant
    if not fail_ok:
        reasons.append(f"失败请求={res.failed_requests} > {fail_tolerant}")

    satisfied = ttft_ok and tpot_ok and fail_ok and res.parse_ok

    def margin(val_ms, lim_s):
        if lim_s < 0 or val_ms is None:
            return None
        v = val_ms / 1000.0
        return (lim_s - v) / lim_s            # 正数=达标且有余量, 负数=超标

    return {
        "satisfied":    satisfied,
        "ttft_margin":  margin(ttft_ms, ttft_lim),
        "tpot_margin":  margin(tpot_ms, tpot_lim),
        "reasons":      reasons,
        "check_ttft":   ttft_lim > 0,
        "check_tpot":   tpot_lim > 0,
        "latency_stat": stat.as_dict(),
    }


def throughput_score(res: BenchResult) -> float:
    """用于排序的吞吐分。

    ponly 输出长度压到 1（output token ≈ 请求数），Output Token Throughput 几乎等于
    req/s、区分不出优劣 —— 参考 p-only-sla.sh 的做法改用 **Input Token Throughput** 排序。
    其余模式优先 Output Token Throughput；缺失则退到 total / req。
    """
    if res.mode in (MODE_PONLY, MODE_PONLY_CACHED) and res.in_tok_thr is not None:
        return res.in_tok_thr
    if res.out_tok_thr is not None:
        return res.out_tok_thr
    if res.total_tok_thr is not None:
        return res.total_tok_thr
    return res.req_throughput or 0.0


def print_result_header(res: BenchResult, scene: dict):
    print(f"\n{'='*78}")
    print(f"  [{scene['name']}]  batch={res.batch}  rate={res.rate}  "
          f"num_prompts={res.num_prompts}  out_len={res.max_out_len}  "
          f"mode={res.mode}")
    print(f"{'='*78}")


def _slo_str(scene: dict, which: str, checked: bool = True) -> str:
    if not checked:
        return "本模式不考核"
    lim = scene["slo_ttft"] if which == "ttft" else scene["slo_tpot"]
    if lim is None or lim < 0:
        return "不限制"
    if which == "tpot":
        return f"{lim*1000:.1f} ms"
    return f"{lim} s"


def print_result_summary(res: BenchResult, scene: dict, slo: dict,
                         stat: Optional[StatCfg] = None):
    """
    stat: 判定用的统计量。不传就从 slo 里带回来的那份取（slo_check 会带上），
          这样调用方忘了传也不会显示成 P90。
    """
    stat = stat or entry_stat_of(slo)
    def fmt(v, unit):
        return f"{v:.4f}{unit}" if v is not None else "N/A"
    flag = "OK" if slo["satisfied"] else "FAIL"
    ttft_lbl = f"TTFT {stat_label(stat.ttft)}"
    tpot_lbl = f"TPOT {stat_label(stat.tpot)}"
    print(f"  → {ttft_lbl:<12} = {fmt(res.lat_ms('ttft', stat.ttft), ' ms')}   "
          f"(limit {_slo_str(scene, 'ttft', slo.get('check_ttft', True))})")
    print(f"  → {tpot_lbl:<12} = {fmt(res.lat_ms('tpot', stat.tpot), ' ms')}   "
          f"(limit {_slo_str(scene, 'tpot', slo.get('check_tpot', True))})")
    # 其余统计量也打出来 —— 判定用的是上面那一个，但回看时常常要对照别的分位
    print(f"  → 时延分布    TTFT: {res.lat_str('ttft')}")
    print(f"                TPOT: {res.lat_str('tpot')}")
    print(f"  → req/s      = {fmt(res.req_throughput, '')}")
    print(f"  → out tok/s  = {fmt(res.out_tok_thr, '')}    in tok/s = {fmt(res.in_tok_thr, '')}"
          f"    total tok/s = {fmt(res.total_tok_thr, '')}")
    print(f"  → failed     = {res.failed_requests}   success={res.success_requests}   "
          f"concurrency={res.max_concurrency}")
    print(f"  → SLO        = {flag}   ({'; '.join(slo['reasons']) or '到达'})   "
          f"耗时 {res.elapsed_s:.1f}s   日志 {Path(res.log_path).name}")
    print(f"{'='*78}")
