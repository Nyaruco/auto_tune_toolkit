#!/usr/bin/env python3
"""
自适应搜索 —— 找到本机最优参数组合（搜 rate: batch × rate；打满: 只有 batch）。

哪条路线不由模式定，由**跑法**定（见 runner.saturates_run）：
  · full 默认搜 rate，加 --no-rate-search 改成只搜 batch
  · donly/ponly 默认只搜 batch，加 --rate-search 改成也搜 rate

核心口径（所有模式一致）:
  · 吞吐是**目标** —— 集群 Output Token Throughput（ponly 改用 Input，见 throughput_score）
  · TTFT / TPOT 是**门槛** —— 不参与排序；但它们的**余量**决定还要不要继续搜
  · 合格 = SLO 达标；吞吐只用于择优和判断是否进入平台期
  · 余量还大（> SLO_FRONTIER_MARGIN）时不停：rate 上界继续扩、平台期继续放大并发，
    逼近 SLA 才收手；吞吐一次回落不会立即停止，连续 plateau_patience 档无实质提升才是平台期

两种搜索路线（都不再参考友商的 batch/rate —— 没有友商数据的场景照样能跑）:
  第一段（三种模式共用）—— rate 打满（--request-rate inf），只剩 batch 一个变量：
                 从 --sweep-start-batch（缺省 1）起倍增上探到第一个不合格点，再在
                 [最后合格, 首个不合格] 之间回搜（粒度 batch_step_min），收敛到最大
                 可用并发（起点自己不合格时先下探，见 _sweep_saturate）。
  第二段（搜 rate 的跑法才有）—— 从 SLO 最大并发继续按 batch_growth 放大 batch；
                 每档只按 SLO 找最大 rate，记录吞吐最高点。rate 上界和平台期都受
                 SLO 余量门控制，余量未耗尽时由 --max-rounds 保险丝兜底。
  打满的跑法没有第二段：rate 就是要打满，不看频率。
  打满上探不做 ±step 扰动（边界两侧都测过了）；第二段做（rate 二分有容差窗口）。

  场景没定义被考核那一轴的 SLO 时（agent 不限 TPOT、ultra_long 两轴都不限）：
  合格判据只剩吞吐一条，上探会一路顶到 batch 上界（开跑前会打 ⚠ 警告），结论取
  测过的点里吞吐最高的那个。

client 自身跑挂的点（解释器找不到 / 服务不可达 / dataset 下载失败）不算"不合格"：
只要出现在任何一个测点上，本场景本次就整体不作数、不写最优（见 ClientFailed）——
免得把 client 的问题当成"这个并发/频率不达标"而把可用点算小。

耗时控制（每次测试几十分钟，必须控制测试条数）:
  · rate 轴用**相对容差** rate_rel_tol（默认 5%）：二分区间宽度 < 已合格端×5% 即停，
    结论最多比真实边界低 5%（基准为什么取已合格端，见 _largest_slo_rate）
  · ±step 扰动只在步长 ≥ 容差时才做（落在容差窗口内，测了也不改变结论）
  · 同一 (batch, rate) 只测一次（结果进缓存）；跑挂的点不进缓存，续跑会重测
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

from runner import (
    BenchResult, run_one, slo_check, throughput_score,
    print_result_header, print_result_summary,
    MODE_FULL, MODE_DONLY, MODE_PONLY, MODE_PONLY_CACHED, MODE_FULL_CACHED,
    MODE_DESC,
    mode_missing_slo, mode_out_len,
    SATURATE_RATE, SATURATE_LABEL, mode_saturates_rate, saturates_run,
    client_for, mode_axes, cache_input_lens, is_cached_mode,
    BenchClientCfg, BENCH_RATE_INF, DEFAULT_PROMPTS_MULT, DEFAULT_RATE_TARGET_S,
    prompts_mult_for, is_long_input, DEFAULT_PROMPTS_MULT_LONG, DEFAULT_LONG_INPUT_LEN,
    CLIENT_AIS, CLIENT_BENCH,
    StatCfg, default_stat, stat_label, entry_lat_ms, entry_stat_of, LATENCY_STATS,
)

# batch 搜索上界（三种模式共用）。上探从 --sweep-start-batch（缺省 1）起倍增，
# 上界只决定"最多探到哪"；--max-batch 可覆盖。
FIXED_BATCH_CAP = 256

# 倍增上探的起点缺省值（--sweep-start-batch 可改）。这台机器扛得住多大并发事先不知道，
# 从 1 起翻倍到第一个不合格点，上探次数本来就只有几个；抬高它省的只是几次低并发测试。
SWEEP_START_BATCH = 1

# 输入长度 → 请求频率基准值：按「输入吞吐 ≈ 2 万 token/s」折算。
# 第二段的 rate 从这儿起搜（全量搜索不再拿友商 rate 当起点）。2 万这个量级是从
# dialog(4096→4.6)、knowledge(7168→3.32)、analysis_16k(16384→1.44) 三个场景的实测最优
# 反推出来的（rate × input_len ≈ 19k~24k，长文本场景另说）。它只是**起点**，准不准
# 无所谓 —— 后面二分上调到 SLO 保不住为止，落点由实测决定。
RATE_REF_IN_TOK_PER_S = 20000
RATE_MIN, RATE_MAX = 0.05, 500.0


# 搜索规则名（写进 state）——续跑时用来判断旧结果还算不算数。算法语义改了名字就
# 必须一起改，旧 state 自动作废重搜（已测点仍复用）。
# +slo_margin：第二段的停止/扩张判据纳入了 SLO 余量，旧的 rate_expand 结论不再适用。
RULE_RATE     = "slo_concurrency_sweep+rate_expand+plateau+slo_margin"
RULE_SATURATE = "slo_concurrency_sweep"
# full + --no-rate-search：跟 donly/ponly 一样只有 batch 一轴，但**不是**同一回事
# （mode 还是 full、client 还是场景配的那个、考核 TTFT+TPOT）。规则名必须跟上面两个
# 都不一样，否则 full 的两种跑法会互相把对方的 state 当成"已收敛"直接复用。
RULE_FULL_SATRATE = "slo_concurrency_sweep+no_rate"

# ── 回搜（撞线后在「已知合格点 lo」和「已知不合格点 hi」之间找最大合格值）的两种打法 ──
#   secant 割线(regula falsi)：拿两端实测的**数值**外推交点，比二分少打高并发点
#   bisect 二分：老办法，只问「合格吗」，每轮区间减半
# 两者收敛到的都是同一个边界，但沿途打点不同 ⇒ 结果可能差一两个 batch，
# 所以规则名里带上它，换打法时旧 state 作废（见 rule_name）。
BACKSEARCH_SECANT = "secant"
BACKSEARCH_BISECT = "bisect"
BACKSEARCH_MODES  = (BACKSEARCH_SECANT, BACKSEARCH_BISECT)

# 割线外推用幂律 v = A·b^α 而不是直线：TTFT 有固定开销（网络往返 + prefill 的定常
# 部分），直线会把那个截距吸收成斜率，从低并发两点外推时被带偏（实测在线性模型上
# 落到 63 而当正确答案是 64）。幂律只有一个参数 α，截距被吸进 A。
SECANT_FLAT_REL = 0.02   # 一根轴在区间里相对变化小于它就当"没在动"，不参与外推

# 吞吐尺子 (--thru-tol) 缺省值，**有符号**：0 = 不涨就算掉，正 = 要求再涨，负 = 允许回落。
# -0.02 跟改符号之前的 0.02 等价，不加参数的老命令行为不变（详见 accept_reason）。
DEFAULT_THRU_TOL = -0.02

# 第二段继续放大并发时使用的倍率，以及连续多少档吞吐没有实质提升后进入平台期。
DEFAULT_BATCH_GROWTH = 1.5
DEFAULT_PLATEAU_PATIENCE = 3
PLATEAU_IMPROVE_REL = 0.01

# SLO 余量安全带：已考核轴的**最小**余量 <= 它才算"已逼近 SLA"、可以停。0.30 留给
# 测量噪声（单点 TTFT 噪声 ±20-30%，见 _sweep_saturate）：实测余量 m 的最坏真实余量
# 约 1.3m-0.3，m=0.30 时仍有 +0.09；再往下搜只是把噪声当信号。
SLO_FRONTIER_MARGIN = 0.30
# 扩张要求 achieved 相对上一点至少再涨 2%，否则判容量墙（再加 rate 只是排队）。
RATE_FOLLOW_REL = 0.02
# rate 上界每次扩张的倍率（撞墙时最多多打 1 个探针）。
RATE_CAP_GROWTH = 1.5

# 割线停滞时强制改走一次二分的阈值（Illinois 修正）。割线在陡峭曲线上会让交点
# 反复贴着同一端，每轮只推进一点点，比二分还慢；连续这么多轮都落在同一侧就拽回来。
SECANT_STALL_LIMIT = 3

# rate 轴是四舍五入到 4 位小数的浮点，割线落点要往回夹一点才能保证落在开区间内
RATE_EPS = 1e-4


def rule_name(saturate: bool, backsearch: str = BACKSEARCH_SECANT,
              no_rate_search: bool = False) -> str:
    """写进 state 的规则名。回搜打法不同 ⇒ 结果可能不同 ⇒ 名字必须跟着变，
    否则拿割线跑出来的 state 会被二分的那次续跑直接当成"已收敛"复用。

    saturate 是**本次跑法**搜不搜 rate（见 runner.saturates_run）：搜 ⇒ RULE_RATE，
    不搜 ⇒ RULE_SATURATE。no_rate_search 单独标出"full 模式下打满"那种 —— 它跟
    donly/ponly 的打满不是一回事（考核轴/client/输出长度都不同），搜出来的"最优"是
    不同曲线上的点，所以自己占一档。"""
    if no_rate_search:
        rule = RULE_FULL_SATRATE
    else:
        rule = RULE_SATURATE if saturate else RULE_RATE
    return f"{rule}+{backsearch}"


# ──────────────────────────────────────────────────────────
# 工具：参数裁剪 / 度量
# ──────────────────────────────────────────────────────────
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def baseline_rate(scene: dict) -> float:
    """按输入长度折算一个请求频率基准值 —— 第二段 rate 搜索的起点。

    输入越长的场景，喂饱预取需要的 req/s 越小，所以取 rate ∝ 1/input_len。
    只是给二分一个量级正确的起点，不需要准（见 RATE_REF_IN_TOK_PER_S 的说明）。
    """
    in_len = int(scene.get("input_len") or 0)
    if in_len <= 0:
        return 1.0
    return round(clamp(RATE_REF_IN_TOK_PER_S / in_len, RATE_MIN, RATE_MAX), 4)


def save_state(state: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def scene_fingerprint(scene: dict, mode: str, *,
                      fail_tolerant: int, thru_tol: float) -> dict:
    """**判据类**参数：改了只需拿旧数字重新判一遍，不用重测。

    收紧或放宽 SLO、换输入长度、换数据集、调 --thru-tol / --fail-tolerant ——
    state 里存的 satisfied/margin 是按**当时**的判据算的，直接复用会得到
    "SLO 已经收紧到 20ms、它却还报 satisfied=True"这种结论。这些参数只影响
    "怎么读那批数字"，所以重判即可，测过的点照常复用。
    """
    fp = {
        "slo_ttft":      scene.get("slo_ttft"),
        "slo_tpot":      scene.get("slo_tpot"),
        "input_len":     scene.get("input_len"),
        "dataset":       scene.get("dataset"),
        # ponly 会把输出长度压到 1，所以记"本模式下实际用的那个值"
        "max_out_len":   mode_out_len(scene, mode),
        "fail_tolerant": fail_tolerant,
        # 键名从 thru_tol 改过来：同名同值会让老 state 里的 0.02 被按新语义（要求每点
        # 涨 2%）静默复用，换个键名逼它必然对不上（_fp_changes 比的是键的并集）
        "thru_need":     round(float(thru_tol), 6),
    }
    if is_cached_mode(mode):
        cached_len, noncached_len = cache_input_lens(scene)
        fp["cached_input_len"] = cached_len
        fp["non_cached_input_len"] = noncached_len
    return fp


def measure_fingerprint(*, prompts_mult: int) -> dict:
    """**测量类**参数：改了必须重测，光重判没有意义。

    --prompts-mult 决定每个点实际发多少个请求，它一变，旧点存的时延/吞吐是在
    另一个"共发多少条"下量出来的 —— 跟换 client 同理，数字本身不可比。
    （state 里每条测点都记了当时的 num_prompts，出表用的是那个值，所以报表不会写错。）

    --rate-target-s 故意**不在**里面：它只改限速点跑多久，不改负载形状（rate 固定，
    每秒发多少条一样；打满点根本不看它）。加进来会让老 state（反推不出这个值，见
    infer_measure_fingerprint）升级后无端重测一遍，代价远大于那点样本量差异。

    prompts_mult 收的是**本场景解析后的倍率**（长输入场景是长输入档那个值，见
    runner.prompts_mult_for）。所以改 --long-input-len / --long-input-mult 时，
    只有真正落到另一档的场景指纹会变、会重测，没跨档的场景不受影响 —— 记原始的两个
    参数就做不到这一点，那会让所有场景一起作废。档位的解析结果跟 infer_measure_
    fingerprint 从 num_prompts/batch 反推出来的东西是同一个量。
    """
    return {"prompts_mult": prompts_mult}


def infer_measure_fingerprint(state: dict) -> Optional[dict]:
    """老 state（这一版之前跑的）没记过 measure_fp，从它自己的测点反推当时的倍数。

    只认**打满点**（rate 为 0 / 空的那些）：那里 num_prompts 恒等于 batch × 倍率，
    比值就是当时的 --prompts-mult。限速点不能拿来比 —— 它发的是
    min(batch × 倍率, max(rate × 目标秒数, 100))，是另一个量：一旦 100 条下限或
    "跑够 N 秒"那头收窄，比值就小于倍率，一套参数跑出来的 state 会被误判成
    "不是一套参数"（实测：60K 那份 ponly state 里有 b=129/num_prompts=100 的点，
    比值 0.775，混进来整体就反推不出来了）。

    打满点之间比值一致才敢认；有出入说明这 state 不是一套参数跑出来的，返回 None
    （没法确定的事就不下结论，宁可提醒，也不拿旧数字冒充新参数）。
    """
    ratios = {round(t["num_prompts"] / t["batch"], 6)
              for t in state.get("tests", [])
              if t.get("batch") and t.get("num_prompts") and not t.get("rate")}
    if len(ratios) != 1:
        return None
    m = ratios.pop()
    return {"prompts_mult": int(m) if float(m).is_integer() else m}


def _fp_changes(old: Optional[dict], new: dict) -> str:
    """人话描述指纹差异（只列变了的键）；old 为 None = 老 state 没记过"""
    if old is None:
        return "旧 state 没记过判据指纹"
    keys = sorted(set(old) | set(new))
    diff = [f"{k}: {old.get(k)} → {new.get(k)}"
            for k in keys if old.get(k) != new.get(k)]
    return "; ".join(diff)


def pct_margin(m: Optional[float]) -> str:
    if m is None:
        return "—"
    return f"{m*100:+.1f}%"


def _slo_frontier_margin(slo: dict) -> Optional[float]:
    """已考核轴的余量最小值；SLO 未定义（没有可考核的轴）返回 None。

    slo_check 对不考核的轴给 margin=None（donly 只看 TPOT、ponly 只看 TTFT），要过滤；
    satisfied 还含失败请求数/parse_ok，调用方必须同时看 satisfied。
    """
    margins = [m for m in (slo.get("ttft_margin"), slo.get("tpot_margin"))
               if m is not None]
    return min(margins) if margins else None


def _point_key(ctx: SearchContext, batch, rate) -> tuple[int, float]:
    """缓存键：batch 量化 + rate 留 4 位（与 SearchContext.test_at 一致）。"""
    return ctx.quantize_batch(batch), round(float(rate), 4)


def _is_report_excluded(ctx: SearchContext, batch, rate) -> bool:
    """是不是只用于下结论的扩张探针（见 ctx.report_excluded）。"""
    return _point_key(ctx, batch, rate) in ctx.report_excluded


def _exclude_from_report(ctx: SearchContext, batch, rate) -> None:
    """把扩张探针移出最终择优池，并写进 state（续跑时同样有效）。"""
    key = _point_key(ctx, batch, rate)
    if key in ctx.report_excluded:
        return
    ctx.report_excluded.add(key)
    state = getattr(ctx, "state", None)
    if state is not None:
        state.setdefault("report_excluded", []).append(list(key))
        save_state(state, ctx.state_path)


class ClientFailed(Exception):
    """某个测点 client **自身**没跑起来（解析失败 / 服务不可达 / loader 下载失败）。

    这跟"这个并发 / 频率不达标"是两回事：拿它当边界，可用点会被算小。所以本场景
    本次整体作废（不写最优、标 converged=False），续跑时重测（跑挂的点不进缓存）。
    统一在 SearchContext.test_at 里抛出 —— 所有测点都从那儿过，两段搜索一条政策；
    search() 一个 try 兜住，落到 drop_scene()。
    """
    def __init__(self, res: BenchResult):
        super().__init__(f"batch={res.batch} rate={res.rate}: {res.error}")
        self.res = res


# ──────────────────────────────────────────────────────────
# 测试缓存：避免同场景下重复测试相同 (batch, rate) — 每次测试都很慢
# ──────────────────────────────────────────────────────────
class SearchContext:
    def __init__(self, scene, host_ip, host_port,
                 log_dir: Path, state: dict, state_path: Path, fail_tolerant: int = 0,
                 mode: str = MODE_FULL, bench_cfg: Optional[BenchClientCfg] = None,
                 stat: Optional[StatCfg] = None,
                 prompts_mult: int = DEFAULT_PROMPTS_MULT,
                 rate_target_s: float = DEFAULT_RATE_TARGET_S,
                 no_rate_search: bool = False,
                 rate_search: bool = False,
                 thru_tol: float = DEFAULT_THRU_TOL,
                 drop_cache: bool = False):
        self.scene           = scene
        self.stat            = default_stat(stat)     # 判定用哪个统计量 (--latency-stat)
        self.host_ip         = host_ip
        self.host_port       = host_port
        self.log_dir         = log_dir
        self.state           = state       # 持久化字典（断点续跑）
        self.state_path      = state_path
        self.fail_tolerant   = fail_tolerant
        self.mode            = mode
        self.bench_cfg       = bench_cfg or BenchClientCfg()   # bench_serving 的路径配置
        # 本次是否走"rate 打满"这条线（donly/ponly 默认，或 full --no-rate-search）。
        # 注意它**只**决定 rate 搜不搜，别拿它反推历史数据的 client —— 见下面 implied_client。
        self.saturate        = saturates_run(mode, no_rate_search=no_rate_search,
                                             rate_search=rate_search)
        self.client          = client_for(scene, mode)         # 压测 client（按场景配）
        # 打满点: num_prompts = batch × 它。传进来的是 search() 解析过的倍率（长输入
        # 场景是长输入档的值），这里不再判输入长度。
        self.prompts_mult    = prompts_mult
        self.rate_target_s   = rate_target_s                   # 限速点: 按这个时长给条数
        self.thru_tol        = thru_tol    # 吞吐尺子，有符号，见 accept_reason
        self.peak_thru       = 0.0         # 已合格点的最高吞吐（日志用）
        self.peak_prev       = 0.0         # **排除本点**后的同一峰值 = 判据基准，_judge 每次刷新
        self.accepted        = {}          # (batch, rate) → 合格点的吞吐，基准从它算
        # rate 轴的上界：由 search() 先给个占位、第二段按实测吞吐收紧（_tune_rate 写）
        self.max_rate:  Optional[float] = None
        # 搜 rate 的跑法定不出有限 rate、退回打满点时为 True（结论要如实标成"打满"）
        self.rate_fallback: bool = False
        # (batch, rate) → BenchResult cache，避免重复测试
        self.cache: dict[tuple[int, float], BenchResult] = {}
        # 扩张探针（跨线/容量墙）只用于下结论，不参与最终择优；缓存恢复后再填，见下。
        self.report_excluded: set[tuple[int, float]] = set()
        # 从 state 恢复缓存
        # drop_cache: 测量类参数（--prompts-mult）变了 —— 旧点的数字是在另一个
        # 请求数下量的，全都不作数，本次一切重测。
        check_ttft, check_tpot = mode_axes(mode)
        # 老 state 没有 per-point client 时的反推值（见下面缓存过滤处的说明）。
        # 按**模式**反推，不看 --no-rate-search —— 那个开关只改本次 rate 搜不搜，
        # 改不了"历史上的 full 数据是 ais_bench 打的"这个事实。
        implied_client = (
            CLIENT_BENCH
            if mode_saturates_rate(mode) or is_cached_mode(mode)
            else CLIENT_AIS
        )
        for t in ([] if drop_cache else state.get("tests", [])):
            # 客户端自身跑挂的点（解释器找不到 / random loader 去 HF 下载失败 /
            # 服务不可达）不代表"这个 batch 不达标" —— 不进缓存，续跑时重测。
            # 否则续跑会直接复读那个 FAIL，把搜索误判成"起点就不达标"提前收敛。
            if not t.get("parse_ok", True):
                continue
            # 判定要用的统计量这个点里没有（老 state 只存了 P90，或者那个版本 client
            # 没打这个分位）—— 复用它会把它误判成"指标缺失 = 不达标"。不进缓存，重测。
            if check_ttft and entry_lat_ms(t, "ttft", self.stat.ttft) is None:
                continue
            if check_tpot and entry_lat_ms(t, "tpot", self.stat.tpot) is None:
                continue
            # 这个点是别的 client 打出来的 —— 同一个 (batch, rate) 换 client 后
            # 数字不可比（ais_bench 读场景 jsonl，bench_serving 是 random 数据集，
            # 连并发语义都不一样）。不进缓存，续跑时按本次 client 重测。
            # 老 state 没记 per-point client：donly/ponly 的 client 恒为 bench_serving，
            # full 历史上是 ais_bench，所以按 mode 反推即可。
            if (t.get("client") or implied_client) != self.client:
                continue
            key = (t["batch"], t["rate"])
            # BenchResult.from_dict 会兼容老 state（只有 *_p90_ms 平铺字段）
            r = BenchResult.from_dict(
                t,
                scene_key=scene["key"],
                num_prompts=t.get("num_prompts", t["batch"] * 8),
                mode=t.get("mode", mode),
                max_out_len=t.get("max_out_len", mode_out_len(scene, mode)),
                host_ip=host_ip, host_port=host_port, log_path=t.get("log", ""),
                elapsed_s=t.get("elapsed_s", 0.0), parse_ok=t.get("parse_ok", True),
            )
            self.cache[key] = r

        # 续跑时恢复扩张探针的排除标记；只留缓存里还在的点 —— 换 client / 测量参数后
        # 旧点会重测，那时的 exclusion 不再适用。
        if not drop_cache:
            for b, r in state.get("report_excluded", []):
                key = (int(b), round(float(r), 4))
                if key in self.cache:
                    self.report_excluded.add(key)

    def _persist_test(self, res: BenchResult, slo: dict):
        self.state.setdefault("tests", []).append({
            "batch":           res.batch,
            "rate":            res.rate,
            "mode":            res.mode,
            "client":          self.client,      # 换 client 后这个点不可比（见缓存过滤）
            "max_out_len":     res.max_out_len,
            "num_prompts":     res.num_prompts,
            "max_concurrency": res.max_concurrency,
            "seed":            res.seed,
            # 时延按统计量存全，日志里有什么就存什么（判定用的是 latency_stat 指定的那个）
            "ttft_ms":         res.ttft_ms,
            "tpot_ms":         res.tpot_ms,
            "e2el_ms":         res.e2el_ms,
            "latency_stat":    self.stat.as_dict(),
            "req_throughput":  res.req_throughput,
            "out_tok_thr":    res.out_tok_thr,
            "in_tok_thr":     res.in_tok_thr,
            "total_tok_thr":  res.total_tok_thr,
            "failed_requests": res.failed_requests,
            "success_requests": res.success_requests,
            "elapsed_s":       res.elapsed_s,
            "log":             res.log_path,
            "parse_ok":        res.parse_ok,
            "error":           res.error,
            "satisfied":       slo["satisfied"],
            "ttft_margin":     slo["ttft_margin"],
            "tpot_margin":     slo["tpot_margin"],
            "reject_reason":   "",      # 被淘汰的原因（由 mark_reject 回头补写）
        })
        save_state(self.state, self.state_path)

    def _refresh_test(self, res: BenchResult, slo: dict):
        """缓存命中时按**本次**口径重新判定后，把新结论写回 state 里那条记录。

        换 --latency-stat 后复用的老测点，判定是按新口径做的，但 state 里存的
        satisfied / latency_stat 还是上一次的 —— 不刷新的话，全测点表会出现
        「判定统计量写着 P90、满足要求列却是按 Mean 判的」这种自相矛盾的行。
        """
        new = {"satisfied": slo["satisfied"], "ttft_margin": slo["ttft_margin"],
               "tpot_margin": slo["tpot_margin"], "latency_stat": self.stat.as_dict()}
        for t in reversed(self.state.get("tests", [])):
            if t.get("batch") == res.batch and abs(t.get("rate", 0.0) - res.rate) < 1e-9:
                if all(t.get(k) == v for k, v in new.items()):
                    return          # 口径没变，别白写一次 state
                t.update(new)
                save_state(self.state, self.state_path)
                return

    def mark_reject(self, res: BenchResult, reason: str):
        """把这个测试点被淘汰的原因记进 state —— 回看数据时才知道它为什么没被选。

        原因由 ctx.accept 判出来（SLO 不达标 / 吞吐回落），在上探判完的那一刻补写。
        三种模式的上探都是这一条判据，所以都能写（坐标下降补测的点不写，留空）。
        """
        if not reason:
            return
        for t in reversed(self.state.get("tests", [])):
            if t.get("batch") == res.batch and abs(t.get("rate", 0.0) - res.rate) < 1e-9:
                t["reject_reason"] = reason
                save_state(self.state, self.state_path)
                return

    def quantize_batch(self, b) -> int:
        """batch(=并发数) 的取值规则。

        三种模式统一：从 1 起倍增上探 → 允许任意整数（含奇数），下界 1；
        边界回搜的粒度由 batch_step_min 决定，不在这里卡偶数。
        （老 full 沿用友商的偶数约定，现在不参考友商了，跟 donly/ponly 一致。）
        """
        return max(1, int(round(b)))

    def _judge(self, res: BenchResult) -> dict:
        """判 SLO，并把合格点的吞吐记进基准。

        基准 = 其它已合格点的峰值，按**点**排除本点（同 key 就跳过），不是按"时间上在本点
        之前"—— 同一个点从缓存里重读时（test_at 的 cache 分支）它早就在基准里了，按时间算
        就成了本点跟自己比：--thru-tol >= 0 时连峰值那个点都会被判成"没涨"。
        """
        key = (self.quantize_batch(res.batch), round(res.rate, 4))
        self.peak_prev = max((v for k, v in self.accepted.items() if k != key), default=0.0)
        slo = slo_check(res, self.scene, fail_tolerant=self.fail_tolerant,
                        mode=self.mode, stat=self.stat)
        if res.parse_ok and slo["satisfied"]:
            # 不合格的点吞吐再高也不能抬高基准，否则后面每个点都会被误判成"吞吐回落"
            self.accepted[key] = throughput_score(res)
        else:
            self.accepted.pop(key, None)     # 重判后不再达标的点要从基准里撤掉
        self.peak_thru = max(self.accepted.values(), default=0.0)
        return slo

    def reset_thru_base(self):
        """清空吞吐基准（第二段换了搜索轴要重立，见 _tune_rate）。"""
        self.accepted.clear()
        self.peak_thru = self.peak_prev = 0.0

    def accept(self, res: BenchResult, slo: dict) -> tuple[bool, str]:
        """本点算不算合格 —— (是否合格, 不合格原因)。判据见 accept_reason。

        基准取 peak_prev（_judge 里按点排除本点后算好）。调用点必须紧跟本点的 test_at，
        别在中间插别的测点。
        """
        why = accept_reason(res, slo, self.peak_prev, self.thru_tol)
        return (not why), why

    def collect_passing(self) -> list:
        """缓存里所有 SLO 达标点，按 throughput 从高到低排序。

        **只按 SLO 过滤，不套吞吐判据** —— 吞吐判据管的是「还往不往上探」，不是「哪些点
        能当结论」。tol <= 0（含缺省）时两者结果相同（峰值点自己永远合格，且排第一）；
        tol > 0 时会分岔：排第一的可能正是被搜索判了「涨不动」而停下的那个，照样交出来。
        """
        out = []
        for (b, r), res in self.cache.items():
            if (b, r) in self.report_excluded:      # 扩张探针只用于下结论
                continue
            slo = slo_check(res, self.scene, fail_tolerant=self.fail_tolerant,
                            mode=self.mode, stat=self.stat)
            if slo["satisfied"]:
                out.append((throughput_score(res), b, r, res, slo))
        out.sort(key=lambda x: x[0], reverse=True)
        return out

    def test_at(self, batch: int, rate: float) -> tuple[BenchResult, dict]:
        """测一个 (batch, rate) 点（缓存命中就直接返回判好的结果）。

        client 自身跑挂时**抛 ClientFailed**（见该异常）：一处抛出，全局一个政策。
        """
        batch = self.quantize_batch(batch)
        rate = round(rate, 4)
        key = (batch, rate)
        if key in self.cache:
            res = self.cache[key]
            slo = self._judge(res)
            self._refresh_test(res, slo)
            _ok, why = self.accept(res, slo)
            print(f"  [cache] batch={batch} rate={rate}: {why or '合格'}  "
                  f"thru={throughput_score(res):.2f}")
            return res, slo

        res = run_one(
            self.scene, batch, rate,
            host_ip=self.host_ip, host_port=self.host_port,
            log_dir=self.log_dir,
            mode=self.mode,
            bench_cfg=self.bench_cfg,
            stat=self.stat,
            prompts_mult=self.prompts_mult,
            rate_target_s=self.rate_target_s,
            # 本次跑法搜不搜 rate 由 ctx 说了算，别再让 run_one 按 mode 默认去猜 ——
            # donly/ponly + --rate-search 下猜出来的正好是反的（限速点被翻成 inf）。
            saturate=self.saturate,
        )
        slo = self._judge(res)
        print_result_header(res, self.scene)
        print_result_summary(res, self.scene, slo, stat=self.stat)
        if not res.parse_ok:
            # 留痕（state 里能看到这次没跑起来、error 是什么）但**不进缓存** ——
            # 进了缓存续跑时就会直接复读这个 FAIL，把它当成"这个点不达标"。
            self._persist_test(res, slo)
            raise ClientFailed(res)
        self.cache[key] = res
        self._persist_test(res, slo)
        return res, slo


# ──────────────────────────────────────────────────────────
# 主搜索算法
# ──────────────────────────────────────────────────────────
def search(scene: dict, *,
           host_ip: str, host_port: str,
           log_dir: Path, state_path: Path,
           max_rounds: int = 8,
           batch_step_min: int = 1,
           rate_step_min: float = 0.05,
           rate_rel_tol: float = 0.0,
           thru_tol: float = DEFAULT_THRU_TOL,
           fail_tolerant: int = 0,
           mode: str = MODE_FULL,
           max_batch_override: Optional[int] = None,
           bench_cfg: Optional[BenchClientCfg] = None,
           model_name: Optional[str] = None,
           stat: Optional[StatCfg] = None,
           prompts_mult: int = DEFAULT_PROMPTS_MULT,
           prompts_mult_long: int = DEFAULT_PROMPTS_MULT_LONG,
           long_input_len: int = DEFAULT_LONG_INPUT_LEN,
           rate_target_s: float = DEFAULT_RATE_TARGET_S,
           no_rate_search: bool = False,
           rate_search: bool = False,
           batch_growth: float = DEFAULT_BATCH_GROWTH,
           plateau_patience: int = DEFAULT_PLATEAU_PATIENCE,
           sweep_start_batch: int = SWEEP_START_BATCH,
           backsearch: str = BACKSEARCH_SECANT) -> dict:
    """对一个场景执行自适应搜索。返回 state dict（同时持久化）。

    mode: full / donly / ponly —— 决定考核哪一轴、以及 ponly 下输出长度压到 1
          （donly 还加 --fake-prefill）。用哪个 client 由场景的 client 字段决定
          （见 runner.client_for），跟 mode 是两件独立的事。
          donly/ponly **默认**把 request rate 直接打满（--request-rate inf），rate 轴
          完全退出搜索，只剩 batch(=并发数) 一个变量；加 rate_search 就反过来搜它。

    no_rate_search: full 专用开关（--no-rate-search）—— 改成把 rate 打满，不搜第二段。
          考核轴、client、输出长度都还是 full 的，只是放弃"请求频率"这一维：
          推荐给不需要在表里报 request rate 的场合。少搜一维**不等于**结论一样 ——
          打满曲线的峰值点和限速曲线的峰值点是两个点（前者并发小、后者能拉更高并发），
          所以规则名单独一档（RULE_FULL_SATRATE），两种跑法的 state 互不复用。

    rate_search: donly/ponly 反过来用的开关（--rate-search）—— 也去搜 request rate，
          规则跟 full 第二段**完全一样**（同一个 _tune_rate、同样的
          SLO-only rate 搜索和平台期停止规则）。
          考核轴、client、输出长度还是 donly/ponly 的：donly 照样 --fake-prefill、
          ponly 照样把输出压到 1。搜出来的点落在同一份 state 里，所以规则名跟打满那档
          必须不同（用 RULE_RATE）—— 两种跑法互相作废重搜的意思，跟 full 那边同理。

          什么时候用：SLO 里带了"请求频率"这一维（表要报 req/s）、或者想知道多高的
          请求频率还扛得住。代价是第二段的十来个测点，每个几十分钟。

    搜索规则（TTFT/TPOT 是硬门槛，**余量**决定还搜不搜，吞吐只用于择优/平台期）:
      第一段（三种模式共用）—— rate 打满，只按 SLO 找**最大可用并发**，
                    不再被吞吐回落提前截断。
      第二段      —— 从第一段得到的最大 SLO 并发继续往外扩：每档并发都搜索
                    “仍满足 SLO 的最大 rate”，记录吞吐最高点；吞吐下降时至少继续
                    探索 plateau_patience 档，避免一次波动就中止。rate 上界和平台期
                    都受 SLO 余量门控制：余量还大就继续扩/继续搜，贴近 SLA 才停。
                    full 默认走，donly/ponly 加 rate_search 才走。
      打满跑法    —— 没有第二段：donly/ponly 默认，或 full + --no-rate-search。

    sweep_start_batch: 第一段倍增上探的起点（--sweep-start-batch，缺省 1），只改"从哪
          开始"不改结论 —— 详见 _sweep_saturate。

    任何一个测点 client 自身跑挂都抛 ClientFailed，本场景本次整体作废（见 ClientFailed）。

    max_batch_override: 覆盖 batch 搜索上界（三种模式默认都是 FIXED_BATCH_CAP=256）。

    bench_cfg: bench_serving 的路径配置（模型路径 / 解释器 / 可选数据集）。

    prompts_mult: 打满点的 num_prompts = batch × 它（--prompts-mult，缺省 5）。
          输入长度 > long_input_len 的场景改用 prompts_mult_long（--long-input-mult，
          缺省 2）—— 长输入的单条请求慢，小一点的倍数就够，三种模式都一样。
          解析在这里做（开头一次），底下只管用。

    rate_target_s: 限速点按"跑够这么多秒"给 num_prompts（--rate-target-s）。限速下
          时长 = num_prompts / rate，按 batch×倍率 给的话并发越大跑越久，而限速点的
          并发只要够接住这个 rate 就不是瓶颈了，多跑的时间买不到新信息。取与
          batch×倍率的小值，只可能更短（见 runner.num_prompts_for）。

    model_name: 本次被测模型的短名（报表「模型」列 / logs/<模型>/<mode>/）。
          换模型重跑时，同目录下旧模型的 state 会被识别为"不是本模型"而跳过，
          避免拿别的模型的 converged 状态把本模型的场景直接判成"已收敛"。

    stat: 判定 SLO 用哪个统计量（--latency-stat / --ttft-stat / --tpot-stat）。
          和 state 里记的口径不一样时，旧结论不算数 → 重搜（已测过的点会复用，
          因为日志里各统计量都存下来了，换口径只是重新判定，不用重跑）。

    backsearch: 撞线后怎么回搜边界 —— secant(割线, 默认) / bisect(二分, 老办法)。
          两者收敛到同一个边界，但沿途打点不同。**打法写进规则名**，换打法旧 state
          作废重搜（见 rule_name），否则拿一种打法跑出的结论会被另一种直接复用。

    batch_growth: 第二段每次并发放大的倍率（默认 1.5）。
    plateau_patience: 第二段连续多少档并发吞吐没有实质提升后进入平台期（默认 3）。
          只有最优余量已 <= SLO_FRONTIER_MARGIN 才真正停，否则继续放大并发。
    max_rounds: 余量仍大时平台期的保险丝（默认 8），防止在容量墙上无限烧测试。
    """
    stat = default_stat(stat)
    backsearch = _parse_backsearch(backsearch)
    if batch_growth <= 1.0:
        raise ValueError("batch_growth 必须大于 1")
    plateau_patience = max(1, int(plateau_patience))

    # num_prompts 倍率分长短两档，**在这里解析一次**（见 runner.prompts_mult_for）：
    # 底下 run_one / SearchContext / 测量指纹拿到的都是解析后的值，不用各自再判一遍。
    prompts_mult = prompts_mult_for(scene, prompts_mult, long_mult=prompts_mult_long,
                                    long_input_len=long_input_len)
    long_tier = is_long_input(scene, long_input_len)

    # 加载 / 初始化 state
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        old_mode = state.get("mode", MODE_FULL)
        if old_mode != mode:
            print(f"\n  ! state 文件模式不匹配: {state_path.name} 记录的是 {old_mode}, "
                  f"本次跑 {mode}。为避免把两种模式的结果混在一起, 跳过该场景。"
                  f"\n    处理: 删掉该文件, 或用 --restart 重新搜索。")
            return state
        old_model = state.get("model_name")
        if model_name and old_model and old_model != model_name:
            print(f"\n  ! state 文件模型不匹配: {state_path.name} 记录的是 {old_model}, "
                  f"本次跑 {model_name}。旧模型的搜索结果对新模型没有意义, 跳过该场景。"
                  f"\n    处理: 删掉该文件 / 用 --restart, 或换 --state-dir 按模型分目录。")
            return state
    else:
        state = {"scene": scene["key"], "mode": mode, "tests": [],
                 "optimal": None, "converged": False}
    state["mode"] = mode
    # 下面这些都是"上次用的是什么"，必须在覆盖 state 字段**之前**取，
    # 否则拿到的永远是本次的值，换了就检测不出来
    old_stat = entry_stat_of(state)             # 老 state 没这字段 → p90
    mode_sat = mode_saturates_rate(mode)        # 模式默认就打满（donly/ponly）
    # 两个开关各只对一个方向有意义，落在"本来就这样"的那一侧就是空操作。不归一化的话
    # 空操作也会把规则名换掉 —— 两边 state 互相作废、来回重搜。
    no_rate_search = bool(no_rate_search) and not mode_sat   # 只对 full 有意义
    rate_search    = bool(rate_search) and mode_sat          # 只对 donly/ponly 有意义
    # 归一化后两开关不会同时为真；真同时给了（绕过 CLI 直接调）rate_search 赢，见
    # runner.saturates_run。
    saturate = saturates_run(mode, no_rate_search=no_rate_search,
                             rate_search=rate_search)   # 本次搜不搜 rate，用法见下
    client   = client_for(scene, mode)
    # 老 state 没记 client：donly/ponly 的 client 恒为 bench_serving（不随场景变），
    # 不用比；full 历史上是 ais_bench，所以缺省按它比 —— 场景改成 bench_serving 后
    # 旧结论会自动作废重搜。
    old_client = state.get("client") or (
        CLIENT_BENCH if is_cached_mode(mode) else CLIENT_AIS
    )
    # 按 mode_sat 判"要不要比 client"，不是 saturate：full + --no-rate-search 的 client
    # 还是场景配的那个（full 缺省 ais_bench），得照常比。
    client_changed = (not mode_sat) and old_client != client
    state["latency_stat"] = stat.as_dict()      # 判定口径，随 state 一起持久化
    state["client"] = client                    # 压测 client（换 client 旧点不可比）
    if model_name:
        state["model_name"] = model_name

    # —— batch 上界 ——
    # 三种模式一样：从 1 起倍增上探，上界只决定"最多探到哪"，--max-batch 可覆盖。
    # （full 原来拿友商 seed_batch×1.5 当上界，现在不参考友商了。）
    if max_batch_override:
        max_batch = int(max_batch_override)
        cap_note = "（--max-batch 指定）"
    else:
        max_batch = FIXED_BATCH_CAP
        cap_note = "（固定上界）"
    max_batch = max(1, max_batch)

    # 已收敛则直接返回。下面三种情况旧结论不算数，重搜一遍（已测过的 (batch, rate)
    # 点仍从 state 复用，所以通常不会真的多跑测试，只是重新判定一遍）：
    #   ① 搜索规则换过（算法本身改了）
    #   ② 压测 client 变了（同一个 (batch, rate) 换 client 后的数字不可比）
    #   ③ 判定口径变了（--latency-stat 换过），同一批数字的结论可能完全不同
    #   ④ 判据/负载形状变了（SLO 数值、输入长度、数据集、容差）——
    #      存下来的 satisfied/margin 是按旧 SLO 算的，直接复用会给出错误结论
    # ①②③④ 都是「只需重判，旧点照用」；测量类参数 --prompts-mult 另算，见下。
    fp = scene_fingerprint(scene, mode, fail_tolerant=fail_tolerant,
                           thru_tol=thru_tol)
    old_fp = state.get("scene_fp")
    # 测量类参数变了要重测（不是重判）—— 旧点的数字是在另一个 --prompts-mult 下
    # 量出来的，跟换 client 同理，不可比。老 state 没记过这个指纹，就从它自己的
    # 测点反推当时的倍数（反推不出来就不强制重测，见 infer_measure_fingerprint）。
    mfp = measure_fingerprint(prompts_mult=prompts_mult)
    old_mfp = state.get("measure_fp") or infer_measure_fingerprint(state)
    measure_changed = old_mfp is not None and old_mfp != mfp
    if measure_changed:
        print(f"\n  [{scene['name']}] 测量参数变了（{_fp_changes(old_mfp, mfp)}），"
              f"旧测点不可比，重新测一遍。")
    if state.get("converged") and state.get("optimal"):
        rule = rule_name(saturate, backsearch, no_rate_search)
        if state.get("search_rule") != rule:
            print(f"\n  [{scene['name']}] 结果来自旧搜索规则，重新搜一遍"
                  f"（已测过的点会复用）。")
            state["converged"] = False
        elif client_changed:
            print(f"\n  [{scene['name']}] 压测 client 变了（{old_client} → {client}），"
                  f"旧结论不算数，重新搜一遍（已测过的点会复用）。")
            state["converged"] = False
        elif old_stat != stat:
            print(f"\n  [{scene['name']}] 判定统计量变了 "
                  f"({old_stat.label()} → {stat.label()})，旧结论不算数，重新判定"
                  f"（已测过的点会复用，只是换个口径重新比）。")
            state["converged"] = False
        elif measure_changed:
            state["converged"] = False
        elif old_fp != fp:
            print(f"\n  [{scene['name']}] 场景判据/负载参数变了"
                  f"（{_fp_changes(old_fp, fp)}），旧结论不算数，重新判定"
                  f"（已测过的点会复用，只是按新判据重新比）。")
            state["converged"] = False
        else:
            print(f"\n  [{scene['name']}] 已收敛跳过: b={state['optimal']['batch']} "
                  f"r={state['optimal']['rate']}")
            # 老 state 反推出来的倍数跟本次一致（才走到这儿），顺手把指纹补记上：
            # 补记的是从测点读出来的事实，不是替它猜一个值。下次再改 --prompts-mult
            # 就能正常识别出来了，而这次一次测试都不用多跑。
            if state.get("measure_fp") is None and old_mfp is not None:
                state["measure_fp"] = old_mfp
                save_state(state, state_path)
            return state

    state_path.parent.mkdir(parents=True, exist_ok=True)
    ctx = SearchContext(scene, host_ip, host_port,
                        log_dir, state, state_path, fail_tolerant=fail_tolerant,
                        mode=mode, bench_cfg=bench_cfg, stat=stat,
                        prompts_mult=prompts_mult, thru_tol=thru_tol,
                        rate_target_s=rate_target_s, no_rate_search=no_rate_search,
                        rate_search=rate_search, drop_cache=measure_changed)

    min_batch = batch_step_min
    min_rate  = rate_step_min
    # rate 基准值：第二段 rate 搜索的起点（输入长度折算，与友商无关）
    rate_base = baseline_rate(scene)
    # rate 上界要到上探结束才能定（用实测吞吐反推，见 _tune_rate）；
    # 这里先给个占位，保证 ctx 上永远是"已设过"的状态。
    max_rate  = RATE_MAX
    ctx.max_rate = max_rate

    check_ttft, check_tpot = mode_axes(mode)
    print(f"\n{'#'*78}")
    print(f"# 场景: {scene['name']}")
    print(f"# 模式: {mode} — {MODE_DESC.get(mode, mode)}   "
          f"(输出长度 {mode_out_len(scene, mode)}{' ← 压到 1' if mode == MODE_PONLY else ''})")
    print(f"# 考核轴: TTFT {'✓' if check_ttft else '✗ 不考核'}   "
          f"TPOT {'✓' if check_tpot else '✗ 不考核'}")
    print(f"# 统计量: TTFT {stat_label(stat.ttft)}{' ✓考核' if check_ttft else ' (本模式不考核)'}"
          f"   TPOT {stat_label(stat.tpot)}{' ✓考核' if check_tpot else ' (本模式不考核)'}"
          f"   (--latency-stat，可选 {', '.join(LATENCY_STATS)})")
    print(f"# rate 基准: {rate_base}"
          f"  (按输入长度 {scene.get('input_len')} 折算, 与友商无关"
          f"{'; rate 打满不用它' if saturate else '; 第二段从它起搜'})")
    # num_prompts 两种给法（见 runner.num_prompts_for）：打满点按时长≈倍率×e2e 给；
    # 限速点按目标时长给，并发越大越省 —— 这里如实报出本次用的是哪条。
    np_desc = f"打满点 batch × {prompts_mult}"
    if long_tier:       # 说清楚这个倍率是长输入档给的，不然跟 --prompts-mult 对不上
        np_desc += (f" (长输入档: 输入 {scene.get('input_len')} > {long_input_len}, "
                    f"--long-input-mult)")
    if not saturate:
        np_desc += f"；限速点 min(那个, rate × {rate_target_s:g}s)"
    print(f"# client:    {client}"
          f"{' — random 数据集, 不读场景 jsonl' if client == CLIENT_BENCH else ' — 读场景 jsonl'}"
          f"{'  --fake-prefill' if mode == MODE_DONLY else ''}   "
          f"num_prompts: {np_desc}")
    print(f"# 合格判据: SLO 达标；吞吐只用于最终择优和平台期判定")
    if not saturate:
        print(f"#            --thru-tol {thru_tol:+g} 保留兼容，不再用于中止第二阶段")
    print(f"# 搜索规则: 第一段 并发从 {sweep_start_batch} 起倍增上探 (--sweep-start-batch) "
          f"到第一个不合格的点, "
          f"再在 [最后合格, 首个不合格] 之间回搜 (粒度 batch_step={batch_step_min}); "
          f"这里只看 SLO，不被吞吐回落截断")
    if mode == MODE_PONLY:
        print(f"# 排序口径: Input Token Throughput (输出长度=1, out tok/s 没有区分度)")
    elif mode == MODE_PONLY_CACHED:
        print(f"# 排序口径: Input Token Throughput (cache 命中 prefill 场景)")
    if saturate:
        print(f"# rate 策略: {SATURATE_LABEL} (--request-rate {BENCH_RATE_INF} —— 请求一次性发出), "
              f"rate 轴不参与搜索, 负载全由并发数 = batch 决定")
        if no_rate_search:
            print(f"#            (--no-rate-search: 显式关掉第二段。考核轴/client/输出长度都还是 "
                  f"full 的，表里「请求频率」列写 {SATURATE_LABEL})")
        print(f"# 搜索上界: batch={max_batch}{cap_note}   (rate 固定={BENCH_RATE_INF})")
    else:
        print(f"# 搜索规则: 第二段 从 SLO 最大并发继续按 {batch_growth:g}x 放大 batch, "
              f"每档搜最大合格 rate (起点 {rate_base}); "
              f"连续 {plateau_patience} 档无实质吞吐提升且 SLO 余量 "
              f"<= {SLO_FRONTIER_MARGIN:.0%} 才停（余量还大时由 --max-rounds 兜底）")
        print(f"# 搜索上界: batch={max_batch}{cap_note}   rate={RATE_MAX}(探测上界, 实际按实测吞吐收紧)")
        # 容差基准是"已合格端 × rate_rel_tol"（不是上界），上界的绝对值不影响结论精度
        print(f"# rate 容差: 相对 {rate_rel_tol:.0%}(乘已合格端)、下限 {rate_step_min:g}   "
              f"batch 步长: {batch_step_min}")
    print(f"# SLO:      ttft<{scene['slo_ttft']}s  tpot<{scene['slo_tpot']}s"
          f"   （按 {stat.label()} 判定）")
    missing = mode_missing_slo(scene, mode)
    if missing:
        print(f"# ⚠ 警告: {mode} 模式只考核 {missing.upper()}, 但本场景没定义它的 SLO ——")
        print(f"#         合格判据只剩吞吐一条, 上探会一路顶到 batch 上界, "
              f"结论取测过的点里吞吐最高的那个。")
    print(f"{'#'*78}")

    # ========================================================
    # 搜索主循环
    #   第一段（三种模式共用）: rate 打满，只按 SLO 搜出最大并发。
    #   第二段（搜 rate 的跑法）: 从最大 SLO 并发继续放大 batch；每档只按 SLO
    #                           搜最大 rate，吞吐用于择优，连续无提升才停。
    # ========================================================
    def drop_scene(fail: BenchResult):
        """client 自身跑挂 —— 本场景本次不作数（见 ClientFailed）。

        已经跑通的点照常留在 state 里，排查后重跑即可，不会白跑。
        """
        print(f"\n  ! (batch={fail.batch}, rate={fail.rate}) 这个点没跑起来: {fail.error}")
        print(f"    这不是 SLO/吞吐不合格，是客户端或服务侧的问题 —— 本场景本次不作数。")
        print(f"    排查后重跑即可（已经跑通的点会从 state 里复用，不会白跑）。")
        state.pop("boundary_batch", None)   # 上一轮可能留下的边界，别让出表读到旧的
        state["converged"] = False
        save_state(state, state_path)
        return state

    # 两段搜索 + 收尾复测的每一个测点都从这里出去：一处兜住所有 ClientFailed。
    try:
        lo_b, _, best = _sweep_saturate(
            ctx, scene, max_batch=max_batch, batch_step_min=batch_step_min,
            start_batch=sweep_start_batch, backsearch=backsearch)
        state["boundary_batch"] = lo_b
        if best is not None:
            if saturate:
                # 打满跑法没有第二段，仍按吞吐峰值作为结论。
                best_res, best_slo = best
                best_score = throughput_score(best_res)
            else:
                # 搜 rate 的跑法必须从 **SLO 最大并发**出发；不能从打满时吞吐最高
                # 的小并发点出发，否则第二段永远看不到更高并发是否能带来更大吞吐。
                best_res, best_slo = ctx.test_at(lo_b, SATURATE_RATE)
                best_score = throughput_score(best_res)
                if best_res.batch != lo_b:
                    print(f"  · SLO 最大并发 batch={lo_b}，从这里进入第二段")
        else:
            # best 是 None ⇔ 连 batch=1 打满都不合格（_sweep_saturate 的下探保证了这一点）
            # —— 打满时没有可用点，就记这个点。它刚测过，这里命中缓存不会再跑一次。
            best_res, best_slo = ctx.test_at(lo_b or 1, SATURATE_RATE)
            best_score = -1.0

        b_hi = max_batch
        if not saturate:
            best_res, best_slo, best_score = _tune_rate(
                ctx, start_res=best_res, start_slo=best_slo,
                start_score=best_score, rate_base=rate_base,
                max_batch=b_hi, batch_step_min=batch_step_min,
                rate_step_min=rate_step_min, rate_rel_tol=rate_rel_tol,
                max_rounds=max_rounds, min_rate=min_rate,
                batch_growth=batch_growth, plateau_patience=plateau_patience,
            )
            max_rate = ctx.max_rate      # _tune_rate 里按实测吞吐收紧过，打印用
            # 上界变了，绝对容差跟着变（Step 4 用它判断 rate 扰动值不值得测）
            rate_tol = max(float(rate_step_min), round(max_rate * rate_rel_tol, 4))

        # ========================================================
        # Step 3: 复测最优确认 (cache 命中直接返回；否则补测一次)
        #   结论用 best_res 那一份就够 —— 这一趟的意义是把 state 里那行按**本次**
        #   口径重判一遍（换 --latency-stat 时靠它刷新），返回值不用。
        # ========================================================
        ctx.test_at(best_res.batch, best_res.rate)

        # ========================================================
        # Step 4: 局部 ±step 扰动确认（寻找可能被跳过的局部更优）
        #   第二段专用：打满跑法的倍增上探 + 二分回搜已经把边界两侧都测了
        #   （lo_b 合格 / hi_b 不合格），再探邻居只会多跑几次几十分钟的测试。
        # ========================================================
        def probe(nb, nr):
            """测一个邻居点；缓存里出现了更优候选就换掉 best_*

            候选池必须跟第二段搜索一致：只认**限速点**（rate > 0）。否则上探阶段的
            打满点也能把结论顶掉 —— 限速点撞到容量时
            两者本来就是同一个工况，吞吐差在噪声里，「请求频率」列会随机在"有限 rate"
            和"inf"之间跳。
            """
            nonlocal best_res, best_slo, best_score
            ctx.test_at(nb, nr)
            passing = ctx.collect_passing()
            if not saturate:
                passing = [p for p in passing if p[2] > 0]
            if passing and passing[0][0] > best_score:
                best_res, best_slo, best_score = passing[0][3], passing[0][4], passing[0][0]

        if saturate:
            reason = ("--no-rate-search 关掉了第二段" if no_rate_search
                      else f"本跑法 rate {SATURATE_LABEL}，没有第二段")
            print(f"  [skip] batch ±{batch_step_min} 扰动 / rate 扰动: "
                  f"上探/回搜已覆盖边界邻域，{reason}")
        else:
            for db in (+batch_step_min, -batch_step_min):
                nb = ctx.quantize_batch(best_res.batch + db)
                # 上界跟第二段一致（b_hi）—— 邻居探到第二段根本没去过的区间没有意义
                if nb != best_res.batch and min_batch <= nb <= b_hi:
                    probe(nb, best_res.rate)
            # rate 扰动步长比搜索容差还小 ⇒ 两个邻居都落在"放宽窗口"内，测了也不会
            # 改变结论，直接跳过（每次测试几十分钟，不值得）。
            if rate_step_min < rate_tol:
                print(f"  [skip] rate ±{rate_step_min:g} 扰动 < 容差 {rate_tol:g}，跳过")
            else:
                for dr in (+rate_step_min, -rate_step_min):
                    nr = round(best_res.rate + dr, 4)
                    if abs(nr - best_res.rate) > 1e-6 and min_rate <= nr <= max_rate:
                        probe(best_res.batch, nr)

    except ClientFailed as cf:
        return drop_scene(cf.res)

    # ========================================================
    # 最终化
    # ========================================================
    # 报出去的这一点是不是打满跑的？打满跑法当然是；搜 rate 的跑法还有一条路：rate 段一个
    # 合格点都没有、退回打满点（ctx.rate_fallback）。那条路上的 rate 是 0.0，
    # 只标 saturate 的话出表会写成「实测频率 0」—— 它是 inf，两个字面量同一个意思。
    rep_sat = saturate or ctx.rate_fallback
    state["optimal"] = {
        "batch":          best_res.batch,
        "rate":           best_res.rate,
        # 给人看的（state 里那行「请求频率」）：打满/退回打满 都写「打满」
        "rate_label":     SATURATE_LABEL if rep_sat else None,
        "rate_saturated": rep_sat,
        "mode":           mode,
        "max_out_len":    best_res.max_out_len,
        "num_prompts":    best_res.num_prompts,
        "max_concurrency": best_res.max_concurrency,
        "seed":          best_res.seed,
        "ttft_ms":        best_res.ttft_ms,
        "tpot_ms":        best_res.tpot_ms,
        "e2el_ms":        best_res.e2el_ms,
        "latency_stat":   stat.as_dict(),
        "req_throughput": best_res.req_throughput,
        "out_tok_thr":    best_res.out_tok_thr,
        "in_tok_thr":     best_res.in_tok_thr,
        "total_tok_thr":  best_res.total_tok_thr,
        "failed_requests": best_res.failed_requests,
        "ttft_margin":    best_slo["ttft_margin"],
        "tpot_margin":    best_slo["tpot_margin"],
        "throughput_score": best_score,
        "satisfied":      best_slo["satisfied"],
    }
    state["rate_saturated"] = rep_sat
    state["max_batch"]      = max_batch
    # rate 轴的上界（第二段按实测吞吐收紧后的值；打满跑法不搜 rate，记 0）
    state["max_rate"]       = 0.0 if saturate else max_rate
    # 记下用的哪条搜索规则 —— 续跑时用它判断旧结果还算不算数
    state["search_rule"]    = rule_name(saturate, backsearch, no_rate_search)
    # 记下本次的判据/负载形状 —— 改 SLO、输入长度、容差等都要让旧结论作废
    state["scene_fp"]       = fp
    # 测量类参数（改了要重测，不是重判）
    state["measure_fp"]     = mfp
    state["converged"] = True
    save_state(state, state_path)

    print(f"\n{'*'*78}")
    print(f"  ★ [{scene['name']}] 最优参数:  (mode={mode}, "
          f"out_len={best_res.max_out_len}, "
          f"考核 {'TTFT' if check_ttft else ''}{'/' if check_ttft and check_tpot else ''}"
          f"{'TPOT' if check_tpot else ''})")
    # 按 rep_sat 而不是 saturate 说：第二段全军覆没退回来的那个点也是打满跑的，
    # 它的 rate=0.0 是哨兵（客户端收到 inf）。跟 state / CSV 用同一个口径。
    rate_desc = (f"{SATURATE_LABEL} (--request-rate {BENCH_RATE_INF})" if rep_sat
                 else f"{best_res.rate}")
    print(f"     batch={best_res.batch}  rate={rate_desc}  num_prompts={best_res.num_prompts}")
    print(f"     TTFT {stat_label(stat.ttft)}={best_res.lat_ms('ttft', stat.ttft)} ms"
          f"  margin={pct_margin(best_slo['ttft_margin'])}")
    print(f"     TPOT {stat_label(stat.tpot)}={best_res.lat_ms('tpot', stat.tpot)} ms"
          f"  margin={pct_margin(best_slo['tpot_margin'])}")
    print(f"     集群 Output Tok/s = {best_res.out_tok_thr}    req/s = {best_res.req_throughput}")
    print(f"     满足 SLO = {'是' if best_slo['satisfied'] else '否'}")
    bnd = state.get("boundary_batch")
    bnd_desc = bnd if bnd is not None else '无（batch=1 就不合格）'
    if rep_sat:
        print(f"     可用并发上界: batch={bnd_desc}   搜索上界 batch={max_batch}")
    else:
        # 边界是"打满"时测的；限速后队列短、时延低，并发还能再往上走，
        # 所以两个数不一样，分开说清楚。
        print(f"     可用并发上界: batch={bnd_desc} (打满时测的)   "
              f"按最优 rate 跑可以到 batch={best_res.batch}")
        print(f"     请求频率: 实测 {best_res.req_throughput} req/s   搜索上界 rate={max_rate}")
    print(f"{'*'*78}\n")
    return state


def _largest_slo_rate(ctx: SearchContext, *, batch: int,
                      initial_rate: float, min_rate: float, max_rate: float,
                      min_step: float, rel_tol: float):
    """固定 batch，找仍然满足 SLO 的最大 rate。

    这里刻意只看 SLO，不看吞吐回落。吞吐是用来比较不同 batch 的目标，
    不能在 rate 轴搜索时充当停止条件，否则新并发下的短暂回落会被误判成
    “这个并发不行”，从而错过后面的更优点。

    跑挂的点到不了这里（ctx.test_at 抛 ClientFailed），所以只看 slo["satisfied"]：
    **不合格**就是"这个频率保不住 SLO"。

    上界合格时交给 _expand_slo_frontier 继续往 SLA 方向扩。report_excluded 里的点
    （上次扩张留下的跨线/容量墙探针）按不合格处理，免得续跑时被"上界合格就返回"
    原样交出去。返回 (res, slo)；一个可当结论的合格点都没有时返回 (None, None)。
    """
    rate = round(clamp(initial_rate, min_rate, max_rate), 4)
    res_lo, slo_lo = None, None

    # 先保证低端合格。低于 min_rate 仍不合格时，这一档 batch 没有可用点。
    for _ in range(12):
        res, slo = ctx.test_at(batch, rate)
        if slo["satisfied"] and not _is_report_excluded(ctx, batch, rate):
            res_lo, slo_lo = res, slo
            break
        if rate <= min_rate + RATE_EPS:
            break
        rate = round(max(min_rate, rate * 0.7), 4)
    if res_lo is None:
        return None, None

    lo = float(res_lo.rate)
    hi = round(max(max_rate, lo), 4)
    if hi <= lo:
        return res_lo, slo_lo

    res_hi, slo_hi = ctx.test_at(batch, hi)
    if slo_hi["satisfied"]:
        if _is_report_excluded(ctx, batch, hi):
            # 续跑：上界是上次的扩张探针 → 退回上一环（hi/1.5，必在缓存里）继续扩，
            # 结论与不中断时一致；找不到就按二分老路走。
            prev_rate = round(hi / RATE_CAP_GROWTH, 4)
            prev_key = _point_key(ctx, batch, prev_rate)
            if prev_key in ctx.cache:
                res_prev, slo_prev = ctx.test_at(batch, prev_rate)
                if (slo_prev["satisfied"]
                        and not _is_report_excluded(ctx, batch, prev_rate)):
                    return _expand_slo_frontier(ctx, batch=batch,
                                                prev=res_prev, prev_slo=slo_prev)
        else:
            # 上界合格 ⇒ 当前 cap 可能压在真实边界下面，继续往 SLA 方向扩。
            return _expand_slo_frontier(ctx, batch=batch, prev=res_hi, prev_slo=slo_hi)

    # 容差基准取 lo（已知合格端）而不是 hi（上界，可能远高于真实边界）：lo ≤ 真实边界，
    # 所以"结论最多比真实边界低 rel_tol"这条承诺成立；拿 hi 当基准，容差会被上界放大成
    # rel_tol × hi/边界。每轮按**当前** lo 重算 —— lo 只往上走，容差只放宽不变窄（收敛
    # 只快不慢），而任何一轮的 lo 都 ≤ 真实边界，承诺不受影响。
    while True:
        tol = max(float(min_step), round(lo * rel_tol, 4))
        if hi - lo <= tol:
            break
        mid = round((lo + hi) / 2.0, 4)
        if mid <= lo or mid >= hi:
            break
        res_mid, slo_mid = ctx.test_at(batch, mid)
        if slo_mid["satisfied"] and not _is_report_excluded(ctx, batch, mid):
            lo, res_lo, slo_lo = mid, res_mid, slo_mid
        else:
            hi = mid
    return res_lo, slo_lo


def _expand_slo_frontier(ctx: SearchContext, *, batch: int,
                         prev: BenchResult, prev_slo: dict):
    """上界点合格时把 rate 往 SLA 方向扩 cap。

    hi 是吞吐导出的（1.5×实测最高 achieved），合格不代表离 SLA 近。只要余量还
    > SLO_FRONTIER_MARGIN、achieved 还随 rate 涨，就按 RATE_CAP_GROWTH 继续探。

    只返回**最后一个被接受的点**。跨线 / 撞墙 / 吞吐持平的那一探只用于下结论：
    排除出合格池（否则 Step 4 的 probe 会把它顶成最优），也不写进 ctx.max_rate
    （否则下一档 batch 从墙外起步，cap 一路棘轮）。探针跑挂沿用全局 ClientFailed 政策。
    """
    cap = float(prev.rate)
    m0 = _slo_frontier_margin(prev_slo)
    if m0 is None or m0 <= SLO_FRONTIER_MARGIN:
        return prev, prev_slo                      # 上界点自己已逼近 SLA，没有余量可扩
    while True:
        next_cap = round(min(RATE_MAX, cap * RATE_CAP_GROWTH), 4)
        if next_cap <= cap + RATE_EPS:
            break
        res, slo = ctx.test_at(batch, next_cap)
        if not slo["satisfied"]:
            break                                  # 撞 SLO 墙（本来就不进合格池）
        m = _slo_frontier_margin(slo)
        if m is None or m <= SLO_FRONTIER_MARGIN:
            _exclude_from_report(ctx, batch, next_cap)
            break                                  # 逼近 SLA
        if achieved_rate(res) <= achieved_rate(prev) * (1.0 + RATE_FOLLOW_REL):
            _exclude_from_report(ctx, batch, next_cap)
            break                                  # 容量墙：吞吐不再跟随 rate
        cap, prev, prev_slo = next_cap, res, slo
        ctx.max_rate = cap                         # 只把**接受**的 cap 交给下一档 / 容差
    return prev, prev_slo


def _tune_rate(ctx: SearchContext, *,
               start_res: BenchResult, start_slo: dict, start_score: float,
               rate_base: float,
               max_batch: int, batch_step_min: int, rate_step_min: float,
               rate_rel_tol: float, max_rounds: int, min_rate: float,
               batch_growth: float = DEFAULT_BATCH_GROWTH,
               plateau_patience: int = DEFAULT_PLATEAU_PATIENCE):
    """第二段：从 SLO 最大并发继续放大 batch，每档找最大合格 rate。

    与旧坐标下降的差别：
      · rate 搜索只看 SLO，不因吞吐回落中止；
      · batch 不是“等到 rate 逼着它上涨”，而是从第一段的最大 SLO 并发主动递增；
      · 某一档吞吐下降时继续至少 plateau_patience 档，最终取全局吞吐最高点。

    停止条件带 **SLO 余量门**：平台期只在最优点余量 <= SLO_FRONTIER_MARGIN
    （已逼近 SLA）时才停；余量还大就继续放大并发，由 --max-rounds 保险丝兜底。
    `--thru-tol` 保留为兼容参数，不再参与第二段的中止判断。
    """
    r_cap = max((achieved_rate(p[3]) for p in ctx.collect_passing()), default=0.0)
    max_rate = round(
        clamp(
            max(r_cap * 1.5, rate_base * 2.0, min_rate * 4.0),
            min_rate,
            RATE_MAX,
        ),
        4,
    )
    ctx.max_rate = max_rate
    print(f"\n  第二段：并发从 SLO 边界继续递增 "
          f"(每档约 {batch_growth:g}x, 上界 batch={max_batch})")
    print(f"          rate 搜索只看 SLO，上界 {max_rate} "
          f"(实测合格点最高 {r_cap:.2f} 的 1.5x，起点 {rate_base})")
    print(f"          余量仍 > {SLO_FRONTIER_MARGIN:.0%} 时上界按 {RATE_CAP_GROWTH:g}x "
          f"继续扩张，逼近 SLA 才收手")

    # 兼容旧调用；新逻辑不再拿吞吐基准否决测点。
    ctx.reset_thru_base()

    best_res, best_slo = start_res, start_slo
    best_score = start_score
    cur_b = ctx.quantize_batch(start_res.batch)
    cur_r = round(
        clamp(start_res.rate if start_res.rate > 0 else rate_base, min_rate, max_rate),
        4,
    )
    no_gain = 0

    while True:
        if cur_b > max_batch:
            break
        # 上一档扩张可能抬高了 cap，本档直接拿它当 hi 起点
        max_rate = ctx.max_rate
        print(f"\n  [batch={cur_b}] 搜索最大 SLO 合格 rate (起点 {cur_r})")
        res, slo = _largest_slo_rate(
            ctx,
            batch=cur_b,
            initial_rate=cur_r,
            min_rate=min_rate,
            max_rate=max_rate,
            min_step=rate_step_min,
            rel_tol=rate_rel_tol,
        )

        if res is None:
            no_gain += 1
            print(f"  [batch={cur_b}] 没有 SLO 合格 rate（无提升 {no_gain}/{plateau_patience}）")
        else:
            score = throughput_score(res)
            meaningful = score > best_score * (1.0 + PLATEAU_IMPROVE_REL)
            if score > best_score:
                best_res, best_slo, best_score = res, slo, score
                print(f"  ↑ 更新最优: batch={res.batch} rate={res.rate} "
                      f"thru={score:.2f} ttft={res.lat_ms('ttft', ctx.stat.ttft)}ms")
            if meaningful:
                no_gain = 0
            else:
                no_gain += 1
            cur_r = float(res.rate)
            print(f"  [batch={cur_b}] 最大合格 rate={cur_r} "
                  f"thru={score:.2f}（无实质提升 {no_gain}/{plateau_patience}）")

        if no_gain >= plateau_patience:
            m = _slo_frontier_margin(best_slo)
            if m is None or m <= SLO_FRONTIER_MARGIN:
                print(f"  → 连续 {plateau_patience} 档并发无实质吞吐提升，"
                      f"且最优已逼近 SLA (min margin {pct_margin(m)})，进入平台期")
                break
            if no_gain >= max_rounds:
                print(f"  → 平台期已持续 {no_gain} 档但 SLO 余量仍大 "
                      f"(min margin {pct_margin(m)}) —— --max-rounds 保险丝，停止")
                break
            print(f"  → 平台期 {no_gain} 档，但 SLO 余量仍大 (min margin {pct_margin(m)} "
                  f"> {SLO_FRONTIER_MARGIN:.0%})，继续放大并发")
        if cur_b >= max_batch:
            break
        next_b = min(
            max_batch,
            max(
                cur_b + batch_step_min,
                int(math.ceil(cur_b * batch_growth)),
            ),
        )
        if next_b <= cur_b:
            break
        print(f"  ↑ batch {cur_b} → {next_b}，继续尝试更高并发")
        cur_b = ctx.quantize_batch(next_b)

    if best_res is start_res and start_res.rate == SATURATE_RATE:
        ctx.rate_fallback = True
        print("  ! 第二阶段没有找到优于打满点的有限 rate 结果，保留打满点")
    return best_res, best_slo, best_score


def accept_reason(res: BenchResult, slo: dict, base: float, thru_tol: float) -> str:
    """本点不合格的原因（返回 "" = 合格）。两种模式共用的判据：

    ① SLO 不达标 —— TTFT / TPOT 是**门槛**；
    ② 吞吐没过线 —— 要求 `thru >= base × (1 + thru_tol)`。并发加上去吞吐不涨了，说明
       已过吞吐拐点，再往上搜没意义（场景没定义 TTFT/TPOT 时②是唯一判据，否则会顶到上界）。

    --thru-tol 是**有符号**的尺子，一个式子覆盖三种说法：

        0     不涨就算掉（**持平也掉**，严格超过基准）     <= 是特例，见下
        0.05  要求比基准再涨 5%（涨不够就掉，边界值放行）
       -0.05  允许比基准回落 5%（落到 95% 以上算合格，边界值放行）

    0 特判成 `<=`：另两档的边界都在及格线上可以放行，0 这档说"不涨"，持平就是没涨。
    不特判的话持平会算成"增长 0% ≥ 0%"而放行，正好相反。

    本条还约定 base 的取法（由调用方保证）：只统计**已合格**的点；**排除本点自己**（含了的
    话 tol>=0 时本点跟自己比，同一个点重判两次会给出两个结论）；base 还是 0 时直接放行
    （没有可比的对象，否则 tol=0 会把"0 对 0"判掉，跟 tol<0 的结论相反）。
    """
    if not slo["satisfied"]:
        why = "  ".join(f"{k.upper()} margin={pct_margin(slo[f'{k}_margin'])}"
                        for k in ("ttft", "tpot") if slo.get(f"check_{k}"))
        return f"SLO 不达标 ({why or '见日志'})"
    thru = throughput_score(res)
    if base <= 0:
        return ""                      # 还没有基准可言
    if thru_tol == 0:
        bad = thru <= base             # 持平也算不涨
        need = f"要求 > 基准 {base:.2f}"
    else:
        bad = thru < base * (1 + thru_tol)
        need = f"要求 >= 基准 {base:.2f} 的 {100 + thru_tol * 100:.0f}%"
    if bad:
        return f"吞吐回落 (thru={thru:.2f}，{need})"
    return ""


def achieved_rate(res: BenchResult) -> float:
    """实测 req/s；读不到（None）按 0 —— 就是"一条都没发出去"的意思。

    日志里吞吐那一段没解析出来时是 None，直接拿去格式化/比较会炸，所以统一从这儿取。
    """
    return res.req_throughput or 0.0


def _axis_lat_ms(res, axis: str, stat) -> Optional[float]:
    """取某根轴用于判 SLO 的时延(ms)——与 slo_check 取的是同一个数。

    取不到（日志里没这个统计量 / 解析失败）返回 None，调用方据此放弃外推。
    """
    if res is None:
        return None
    try:
        return res.lat_ms(axis, stat.ttft if axis == "ttft" else stat.tpot)
    except (AttributeError, KeyError):
        return None


def _power_crossing(x1, v1, x2, v2, lim_ms: float) -> Optional[float]:
    """幂律外推：解 v(x) = lim 得到 x。解不出来返回 None。

    会返回 None 的四种情况，每一种都是"这两个点不足以外推"，不是错误：
      · 平坦 —— 相对变化小于 SECANT_FLAT_REL。TPOT 在边界前后常年只差百分之几，
        拿这点抖动当斜率会解出一个荒唐的交点（实测 α≈0.05 → 交点 7000+）。
      · 反向 —— v2 <= v1。实测曲线噪声能到 ±20~30%（有曲线并发 64 的 TTFT 反而
        比 32 低），此时拟合无意义。
      · 已越界 —— v1 >= lim，起点就已经贴着/超过 SLO 了。
      · α 退化到 0 —— 除零保护。
    """
    if x2 <= x1 or v1 is None or v2 is None or lim_ms <= 0:
        return None
    if v1 <= 0 or v2 <= 0:
        return None
    if abs(v2 - v1) < SECANT_FLAT_REL * max(abs(v1), abs(v2)):
        return None
    if v2 <= v1 or v1 >= lim_ms:
        return None
    alpha = math.log(v2 / v1) / math.log(x2 / x1)
    if alpha <= 1e-6:
        return None
    return x1 * (lim_ms / v1) ** (1.0 / alpha)


def _secant_mid(*, lo, hi, res_lo, res_hi, scene: dict, stat, is_batch: bool):
    """割线（regula falsi）给出的下一个测点，**保证落在开区间 (lo, hi) 内**。

    给不出落点就返回 None，调用方退回二分 —— 这是本方法唯一的安全网：
    割线只是在区间内部挑一个"更聪明的中点"，挑不出来就退化成分半，不会比老办法慢。

    两根轴各解一个交点，取小的：哪个轴先撞 SLO 就在哪儿停。

    注意夹取顺序（踩过的坑）：解出 31.999999999999996 时 round() 给 32 == hi，
    落点跑到区间外，调用方一看"没夹在中间"就收工 —— 边界直接丢了。所以先按浮点解
    筛，四舍五入之后再夹进 [lo+1, hi-1]（浮点轴夹 RATE_EPS）。
    """
    cands = []
    for axis in ("ttft", "tpot"):
        lim_s = scene.get(f"slo_{axis}", -1)
        if not lim_s or lim_s <= 0:
            continue
        x = _power_crossing(lo, _axis_lat_ms(res_lo, axis, stat),
                            hi, _axis_lat_ms(res_hi, axis, stat), lim_s * 1000.0)
        if x is not None and lo < x < hi:
            cands.append(x)
    if not cands:
        return None
    x = min(cands)

    if is_batch:
        if hi - lo < 2:                 # 区间里已经没有整数了
            return None
        return int(min(max(round(x), lo + 1), hi - 1))
    lo_in, hi_in = lo + RATE_EPS, hi - RATE_EPS
    if lo_in >= hi_in:
        return None
    return round(min(max(x, lo_in), hi_in), 4)


def _parse_backsearch(mode: str) -> str:
    m = (mode or "").strip().lower()
    if m not in BACKSEARCH_MODES:
        raise ValueError(f"未知的回搜方式: {mode!r}（可选 {', '.join(BACKSEARCH_MODES)}）")
    return m


def _sweep_saturate(ctx: SearchContext, scene: dict, *,
                    max_batch: int, batch_step_min: int,
                    start_batch: int = SWEEP_START_BATCH,
                    backsearch: str = BACKSEARCH_SECANT):
    """第一段搜索（三种模式共用）：rate 打满，只按 SLO 找最大并发。

    在 rate 打满下做 —— 负载全由并发决定，batch 是唯一变量，上探/回搜才有单调
    的边界可找。搜 rate 的跑法拿到边界后再去第二段定 rate（见 _tune_rate）。

    这里**只看 SLO**，不套吞吐闸门：吞吐回落只说明打满曲线过峰，不代表更大并发
    在限速后不能达到更高吞吐。

    start_batch 是倍增上探的起点（--sweep-start-batch，缺省 1）。抬高它省几个低并发点，
    但起点自己就可能已经不合格（边界在它下面），这时**下探** batch=1 把边界找回来
    —— 不这么做的话，抬高起点会把真实边界整段跳过去。

    返回 (lo_b, hi_b, best)：
      lo_b  已验证合格的最大并发（None = 连 batch=1 都不合格）
      hi_b  第一个不合格的并发（None = 上界内全部合格）
      best  合格点里吞吐最高的 (res, slo)（None = 没有合格点）

    取 best 而不是"缓存里 SLO 达标点中吞吐最高的"：口径必须跟合格判据一致，
    否则可能把一个被淘汰的点当成最优。
    """
    lo_b = hi_b = None
    best_thru, best = -1.0, None
    b = max(1, ctx.quantize_batch(start_batch))
    if b > max_batch:
        print(f"  ! --sweep-start-batch={b} 超过搜索上界，收到 {max_batch}")
        b = max_batch
    print(f"\n  上探: 并发从 {b} 起，每次翻倍（{b} → {b*2} → {b*4} …），"
          f"直到 SLO 不达标或到上界 batch={max_batch}")

    res_of = {}                     # batch → BenchResult，供割线取两端的实测数值

    def visit(batch):
        """测一个点。返回 "" = 合格、不合格原因 = 不合格"""
        nonlocal best_thru, best
        res, slo = ctx.test_at(batch, SATURATE_RATE)
        res_of[ctx.quantize_batch(batch)] = res
        ok = bool(slo["satisfied"])      # parse_ok 已含在其中（跑挂的点根本到不了这儿）
        why = "" if ok else ("SLO 不达标: " + "; ".join(slo.get("reasons") or ["见日志"]))
        if ok:
            thru = throughput_score(res)
            if thru > best_thru:
                best_thru, best = thru, (res, slo)
        else:
            # 把"为什么这个点不算数"记进 state —— 出表时要标出来
            ctx.mark_reject(res, why)
        return why

    while True:
        bad = visit(b)
        if bad:
            hi_b = b
            print(f"  ↓ batch={b} {bad} —— 回搜边界")
            break
        lo_b = b
        if b >= max_batch:
            print(f"  ↑ batch={b} 合格，且已到上界 batch={max_batch} —— 停止上探")
            break
        nb = min(b * 2, max_batch)
        print(f"  ↑ batch={b} 合格 (thru={ctx.peak_thru:.2f}) → 翻倍到 {nb}")
        b = nb

    if lo_b is None:
        # 起点自己就不合格 ⇒ 边界在它下面。起点是 1 就到此为止（batch=1 打满都不合格 =
        # 打满时没有可用点）；起点 > 1 得**下探**：先测 batch=1，合格就把 [1, 起点)
        # 交给同一套回搜打法 —— 否则 --sweep-start-batch 会把真实边界整段跳过去。
        why1 = visit(1) if hi_b > 1 else "起点就是 batch=1"
        if why1:
            # 只说"打满时"没有可用点：搜 rate 的跑法接着还会去第二段找限速点，
            # 限速下 batch=1 完全可能是合格的（所以别写成"这个场景没有可用点"）。
            print(f"  ! batch=1 也不合格（{why1}）: 打满时没有可用点")
            return lo_b, hi_b, best
        lo_b = 1
        print(f"  ↑ batch=1 合格 —— 边界落在 [1, {hi_b}) 之间，回搜")

    if hi_b is None:
        print(f"  ↩ 上界内 (batch≤{max_batch}) 全部合格: 最大可用并发 batch={lo_b}")
        return lo_b, hi_b, best

    how = "割线" if backsearch == BACKSEARCH_SECANT else "二分"
    print(f"  回搜: 在 batch=({lo_b}, {hi_b}) 之间{how}，找最大可用并发")
    keep_lo = keep_hi = 0        # 某一端被连续保留几次（割线停滞的信号）
    while hi_b - lo_b > batch_step_min:
        mid = None
        # 连续 SECANT_STALL_LIMIT 轮都落在同一侧 ⇒ 割线贴着那一端走（陡崖上必现），
        # 退化成一步一步爬，这时比二分还慢 —— 强制走一次中点把它拽回来。
        # 这就是 regula falsi 的 Illinois 修正。
        if backsearch == BACKSEARCH_SECANT and \
                keep_lo < SECANT_STALL_LIMIT and keep_hi < SECANT_STALL_LIMIT:
            mid = _secant_mid(lo=lo_b, hi=hi_b,
                              res_lo=res_of.get(lo_b), res_hi=res_of.get(hi_b),
                              scene=scene, stat=ctx.stat, is_batch=True)
        if mid is None:
            mid = (lo_b + hi_b) // 2
        if mid <= lo_b or mid >= hi_b:
            break
        bad = visit(mid)
        if bad:
            hi_b, keep_hi, keep_lo = mid, 0, keep_lo + 1
            print(f"    ↓ batch={mid} 不合格（{bad}）")
        else:
            lo_b, keep_lo, keep_hi = mid, 0, keep_hi + 1
            print(f"    ↑ batch={mid} 合格")
    print(f"  ↩ 回搜结束: 最大可用并发 batch={lo_b}（第一个不合格 {hi_b}）")
    return lo_b, hi_b, best
