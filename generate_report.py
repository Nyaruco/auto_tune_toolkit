#!/usr/bin/env python3
"""
generate_report.py — 把 auto_tune 的最优结果生成能与友商对齐的 CSV 报表

输入:
  - report.json      (auto_tune.py 输出，含每场景的最优 batch/rate/吞吐/TTFT/TPOT)
  - state_<场景>.json (auto_tune.py 的断点续跑文件，含**每一次**测试的原始数字)

输出（两张表）:
  1) metax_summary_<模型>.csv  结论表 —— 每行一个「场景 × 模式」的最优结果，
                               列结构与友商对齐，用来对标
  2) metax_report_<模型>.csv   全测点表 —— 每个真跑过的测试点一行（含没达标的），
                               带「满足要求」「最优」「备注」标记，用来看趋势/找拐点

CSV 列结构
  结论表（20 列 / 单模式 19 列 —— 只保留实测值）:
    [模式] | 模型 | 测试用例 | 输入/输出 | SLO 要求 | precision | 输入总请求数
    请求频率 | 最大并发
    <时延> 首token平均时延(s) | Mean … | P99 …
    <时延> decode 平均时延(s)  | Mean … | P99 …
    Mean 端到端时延(s) | P99 …
    QPS(req/s) | 集群 Output Token Throughput | 集群 Total Token Throughput
  全测点表（25 列）:
    模式 | 模型 | 测试用例 | 输入/输出 | SLO 要求 | precision | 判定统计量
    请求频率 | 最大并发 | 输入总请求数 | QPS(req/s)
    TTFT(s)  + Mean / P99 | TPOT(ms) + Mean / P99 | Mean / P99 E2E(s)
    输入吞吐 | 输出吞吐 | 总吞吐 | 满足要求 | 最优 | 备注
  E2E 只有两列：它不参与 SLO 判定，没有"判定统计量"，也就没有判定列 —— 两个 client
  都打不出 P95/P50 的 E2E（bench_serving 的 E2E 块只打 Mean/Median/P90/P99），
  硬给一列跟着 TTFT 口径走的话，--latency-stat p95 时它整列是 "—"。

  两张表都是**单行表头**（第一行就是列名）。结论表原来在列名之上还有一行合并的
  分组行（测试环境 / AIS_bench 参数 / 时延 / 吞吐），已经去掉：`输入总请求数` 底下
  只辖一列，`precision`、`模式` 挂在"测试环境"下也名不副实，而 Excel/pandas 打开
  平白多一行要绕。分组信息本来就在列名里（"首token平均时延"、"集群 Total Token
  Throughput"），不需要再来一行。

  「测试用例」原来是一格塞三行（场景名 / TTFT SLO / TPOT SLO），现在拆成
  「测试用例」（场景名，长度括号去掉）+「输入/输出」+「SLO 要求」（两条要求用「；」
  连成一行）—— 单元格里不再有换行，Excel 里能直接排序/筛选。
  「输入/输出」从场景的 input_len / max_out_len 现算（1024→1K、512→512），不解析
  场景名，所以 scenes_custom.py 那种没有长度括号的名字也对得上。写的是**场景配置**
  的长度，不是本次实际发的（ponly 的输出会被压到 1 个 token）。

  时延每个轴占三列：判定口径那一列（见 auto_tune.py --latency-stat，默认 P90）
  + 并列的 Mean / P99。结论表的判定列**表头跟着实际口径变**，不会拿 P90 的表头装
  Mean 的数字；判定口径本身就是 Mean/P99 时跳过重复的并列列（免得两列同名同值）。
  全测点表的 Mean/P99 是固定列，宽表更适合一列一列比对。

  口径是从数据里读的（每条 entry 的 latency_stat），不是这个脚本的参数 —— 所以
  跑测换了口径，出表自动跟上，不用两边对齐。**一条结果都没搜到时也照读**：没搜到
  往往正是因为 --latency-stat p95 取不到值（client 没打 P95 补丁），这时若兜底成
  P90，表头写「P90 首token平均时延（s）」、同一行的「SLO 要求」却写 ttft_p95 < 3s，
  读表的人会以为只是 P90 那列缺数，实际是一个点都没跑成。数据里真的没有 latency_stat
  （老 report）才退回 P90。零结果的时延格子留空 —— 空 = 没测到，"—" = 本模式不考核。

  被删掉的列（机器数量 / 单机卡数量 / P、D 实例资源数与并行配置）：这些不是实测值，
  是部署方手填的元数据；"单卡吞吐" = 集群吞吐 ÷ (机器数量 × 单机卡数量)，依赖的正是
  那几个手填值，所以一并删掉。要留档就记在报告之外（部署文档 / 机房台账）。

  本模式不考核的那一轴写 "—"（两张表同一套规则）：donly 的 TTFT 是 --fake-prefill
  下的数（prefill 被跳过），ponly 输出只有 1 个 token、请求内没有 ITL（TPOT 恒 0，
  E2E 也就等于 TTFT）—— 写数字会被误读成"这台机器这个轴很快"。E2E 只有在
  TTFT 和 TPOT **都考核**的 full 模式下才写数值。

  文件用 utf-8-sig（带 BOM）写，Excel 双击打开不会把中文显示成乱码。

额外需要的信息 (只影响报表两列元数据，不进算法，全部走 CLI):
  - model              报表「模型」列，缺省 scenes.MODEL_NAME（本机实际模型），--model-name 覆盖
  - precision          报表 precision 列，缺省 "INT8"（真实部署精度，自己填对）

用法:
    python3 generate_report.py --all-modes                # 两张表（最常用）
    python3 generate_report.py --all-points               # 只要全测点表
    python3 generate_report.py --all-modes --print-table  # 同时打印 human-readable 表
    python3 generate_report.py                            # 单模式（只读 report.json）
    python3 generate_report.py --report other.json --out x.csv
    python3 generate_report.py --precision INT8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

# --scenes-file 换场景配置 —— 出表也得跟跑测用同一套场景（场景顺序、key、
# 报表里的场景名都从它来），所以同样要在 import scenes 之前换掉。
import scene_file

SCENES_FILE = scene_file.bootstrap(sys.argv[1:], Path(__file__).parent)

from scenes import SCENES, SCENE_BY_KEY, MODEL, MODEL_NAME
from runner import (MODE_FULL, MODE_DONLY, MODE_PONLY, MODE_PONLY_CACHED,
                    MODE_FULL_CACHED, CACHED_MODES, is_cached_mode,
                    mode_axes, mode_saturates_rate, entry_lat_ms, entry_stat_of, stat_label,
                    DEFAULT_LATENCY_STAT, DEFAULT_PROMPTS_MULT,
                    CLIENT_AIS, CLIENT_BENCH)


# 报表「模型」列 = scenes.MODEL_NAME (本机实际跑的模型)，不再硬编码友商模型名。
# 可用 CLI --model-name 覆盖。
#
# 两种出表方式:
#   1) 单模式：--report report_<mode>.json（默认 report.json）
#   2) 三模式汇总（--all-modes）：把 auto_tune.py 各模式跑出的
#      report.json / report_donly.json / report_ponly.json 合成一张结论表，
#      每行一个「场景 × 模式」，多一列「模式」，默认输出 metax_summary_<模型>.csv
#      （同一趟还会顺手出全测点表 metax_report_<模型>.csv）

# CSV 列（友商 18 列做了删减：只留"真实测出来的"，拓扑那几列删掉）
#   删掉了部署拓扑那几列 ——
#   机器数量/单机卡数量/P、D 实例资源数与并行配置是手填的元数据，不是实测值；
#   单卡吞吐 = 集群吞吐 ÷ (机器数量×单机卡数量)，依赖的正是那两个手填值，
#   所以一并删掉（原先靠一份 pd_config.json 填，那份配置也一起去掉了）。
#
# 时延列**每个轴占好几列**：判定口径那一列（--latency-stat，默认 P90）+ 并列的
# Mean / P99。判定口径表头得跟着变，和它重复的 Mean/P99 又得跳过 —— 列数不固定，
# 所以表头和数据行都从 summary_cols() 这一份列定义生成。原来按下标去改表头
# （row2[6] = …），列一变就整体左移一格。
EXTRA_STATS = (("mean", "Mean"), ("p99", "P99"))


def _latency_cols(ttft_stat: str, tpot_stat: str) -> list[tuple]:
    """结论表「时延」组的列 —— [(列名, 分组, 取值键), …]。

    取值键是 (轴, 统计量)，取值见 _summary_values。判定口径排在各轴最前，
    后面固定跟 Mean / P99；和判定口径重复的跳过，免得同表出现两列同名同值
    （--latency-stat p99 时就会撞上）。E2E 不参与判定，只列 Mean / P99。
    """
    out = []
    for axis, judged, name in (("ttft", ttft_stat, "首token平均时延"),
                               ("tpot", tpot_stat, "decode 平均时延"),
                               ("e2el", "",        "端到端时延")):
        if judged:
            out.append((f"{stat_label(judged)} {name}（s）", (axis, judged)))
        for stat, lbl in EXTRA_STATS:
            if stat == judged:
                continue
            out.append((f"{lbl} {name}（s）", (axis, stat)))
    return out


def summary_cols(merged: bool, ttft_stat: str, tpot_stat: str) -> list[tuple]:
    """结论表的列定义 —— [(列名, 取值键), …]，表头和数据行的唯一来源。

    merged=True 时在「模型」后插一列「模式」（三模式汇总表用）。
    「测试用例」原来是一格里塞三行（场景名 / TTFT SLO / TPOT SLO），现在拆成
    场景名 + 输入/输出 + SLO 要求三列，单元格里不再有换行。
    """
    cols = [("模型", "model")]
    if merged:
        cols.append(("模式", "mode"))
    cols += [
        ("测试用例",     "case"),
        ("输入/输出",    "io"),
        ("SLO 要求",     "slo"),
        ("precision",    "precision"),
        ("输入总请求数", "num_prompts"),
        ("请求频率",     "rate"),
        ("最大并发",     "conc"),
    ]
    cols += _latency_cols(ttft_stat, tpot_stat)
    cols += [
        ("QPS（req/s）",                            "qps"),
        ("集群 Output Token Throughput(tokens/s)", "out_tok"),
        ("集群 Total Token Throughput(tokens/s)",  "total_tok"),
    ]
    return cols


# ── 全测点表（metax_report_<模型>.csv）─────────────────────────────────
# 一行 = 一个真跑过的测试点（场景 × 模式 × 并发点），达没达标都留着 ——
# 结论表只留最好的那一行，看趋势、找吞吐拐点、解释"为什么只选到 6 并发"要看这张。
#   TTFT/TPOT 两列取「判定统计量」列指的那个分位（默认 P90），不是固定 P90；
#   后面跟的 Mean / P99 是**固定**的，口径混着也算得出来，不随判定统计量变
#   （这张表的列宽固定才好一列一列对比，所以这里不做结论表那种"重复就跳过"）。
#   E2E 不参与判定，只列 Mean / P99，单位是秒（几百上千个 token 的端到端是几十秒）。
ALL_POINTS_HEADER = [
    "模式", "模型", "测试用例", "输入/输出", "SLO 要求", "precision", "判定统计量",
    "请求频率", "最大并发", "输入总请求数", "QPS（req/s）",
    "TTFT（s）", "Mean TTFT（s）", "P99 TTFT（s）",
    "TPOT（ms）", "Mean TPOT（ms）", "P99 TPOT（ms）",
    # E2E 只有 Mean / P99 两列 —— 它不参与 SLO 判定，没有自己的「判定统计量」。
    # 原来前面还有一列跟着 TTFT 口径走的 E2E，但两个 client 都打不出 P95/P50 的 E2E
    # （bench_serving 的 E2E 块只打 Mean/Median/P90/P99），--latency-stat p95 时那列
    # 整列是 "—"，旁边 Mean/P99 却有值 —— 表头又不带统计量名，读表的人分不清是
    # 「缺 P95」还是「没测」。跟结论表保持一致，干脆不要这一列。
    "Mean E2E（s）", "P99 E2E（s）",
    "输入吞吐(tokens/s)", "输出吞吐(tokens/s)", "总吞吐(tokens/s)",
    "满足要求", "最优", "备注",
]

# 模式 → auto_tune.py 写出的 report 文件名（full 保持 report.json 不变）
MODE_REPORT_FILES = {
    MODE_FULL:  "report.json",
    MODE_DONLY: "report_donly.json",
    MODE_PONLY: "report_ponly.json",
    MODE_PONLY_CACHED: "report_ponly_cached.json",
    MODE_FULL_CACHED: "report_full_cached.json",
}
MODE_ORDER = (MODE_FULL, MODE_DONLY, MODE_PONLY, MODE_PONLY_CACHED,
              MODE_FULL_CACHED)


# ──────────────────────────────────────────────────────────
# 报表元数据：只有「模型」和 precision 两列
# ──────────────────────────────────────────────────────────
DEFAULT_PRECISION = "INT8"


def report_meta(*, model_name: str | None = None,
                precision: str | None = None) -> dict:
    """报表里要写的两列元数据。

    部署拓扑（机器数 / 卡数 / P、D 实例与并行配置）**不在这里** —— 那些列已经从
    CSV 删掉了（它们不是实测值，摆进表里会被当成这台机器的成绩），所以整套
    pd_config.json 也一并去掉：留一个只为填俩字段的配置文件不值当。
    """
    return {
        "model":     model_name or MODEL_NAME,
        "precision": precision or DEFAULT_PRECISION,
    }


# ──────────────────────────────────────────────────────────
# 把 report.json 的结果索引成 {scene_key -> optimal_dict}
# 同时也会调和 state_<scene>.json 里的数据：
#   - 优先使用 report.json 的 optimal（该是搜索收敛后的结果）
#   - state 里 optimal 亦可作备兑给出中途查看。
# 在搜索中途查看时 report.json 还不存在 —— 这时从 state<scene>.json 里的 tests 内
# 随机取最近 SLO 达标的点作为 ``当前最佳``，让 CSV 至少有数值。
# ──────────────────────────────────────────────────────────
def _point_client(t: dict, mode: str) -> str:
    """这条测点记录是哪个 client 打出来的。

    老 state 没有 per-point client：donly/ponly 的 client 恒为 bench_serving
    （不随场景变），full 历史上是 ais_bench —— 按 mode 反推。
    注意用 mode 反推而不是 state["client"]：那个字段每次 search() 都会被刷成
    本次的值，拿它过滤会让旧点"看起来"也是新 client 打的。
    """
    return t.get("client") or (
        CLIENT_BENCH
        if mode_saturates_rate(mode) or is_cached_mode(mode)
        else CLIENT_AIS
    )


def _pick_best_in_state(state: dict) -> dict | None:
    """从 state 里挑一个「当前最佳」点，让搜索中途也能出数。

    已收敛 → 直接用 state.optimal；没收敛 → 取达标点里吞吐最高的那个
    （ponly 输出长度=1，out tok/s 没区分度，跟搜索一样看 input token throughput）。
    一个达标的都没有就返回 None，调用方按「没有达标点」处理。
    """
    if state.get("converged") and state.get("optimal"):
        return state["optimal"]
    best = None
    best_score = -1.0
    mode = state.get("mode", MODE_FULL)
    # 换过 client 之后 state 里会同时留着两种 client 的测点 —— 同一个 (batch, rate)
    # 两边的数字不可比，只在本 client 的点里选（跟 search 的缓存过滤同一套规则）。
    cur_client = state.get("client")
    # ponly 输出长度=1, out tok/s 没区分度 → 跟搜索一样用 input token throughput
    key = "in_tok_thr" if mode in (MODE_PONLY, MODE_PONLY_CACHED) else "out_tok_thr"
    for t in state.get("tests", []):
        if not t.get("satisfied"):
            continue
        if cur_client and _point_client(t, mode) != cur_client:
            continue
        score = (t.get(key) or t.get("total_tok_thr") or 0)
        if score > best_score and score > 0:
            best_score = score
            best = t
    return best


def load_mode_states(state_dir: Path, mode: str, model_name: str | None = None) -> dict:
    """没有 report_<mode>.json 时的退路：直接从 state_<场景>[_<mode>].json 里取结论。

    搜索被中断、或者只手工存了 state 的目录（比如 runs/<模型>/ 下只有一堆
    state_*_ponly.json），report 根本不会生成 —— 但每次测试的数字都在 state 里，
    结论照样拿得出来。取的规则跟 _pick_best_in_state 一致：收敛了用 optimal，
    没收敛就用已达标的点里吞吐最高的那个。
    """
    out: dict[str, dict] = {}
    for sc in SCENES:
        sfx = "" if mode == MODE_FULL else f"_{mode}"
        p = state_dir / f"state_{sc['key']}{sfx}.json"
        if not p.exists():
            continue
        try:
            st = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[警告] state 读取失败 {p}: {e}", file=sys.stderr)
            continue
        if st.get("mode", MODE_FULL) != mode:
            continue
        if model_name and st.get("model_name") and st["model_name"] != model_name:
            continue
        opt = _pick_best_in_state(st)
        if not opt:
            continue
        out[sc["key"]] = {
            **opt,
            "scene_name":  sc["name"],
            "mode":         mode,
            # 打满模式的「请求频率」列要写 inf（state 里记着，缺了按模式推）
            "rate_saturated": st.get("rate_saturated", mode_saturates_rate(mode)),
            "latency_stat": st.get("latency_stat") or opt.get("latency_stat"),
            "converged":   st.get("converged", False),
            "num_tests":   len(st.get("tests", [])),
            "_source":     "state",
        }
    return out


def report_mode(path: Path) -> str:
    """这份 report 是哪个模式跑出来的 —— 读它顶层的 `mode`。

    单模式出表（不带 --all-modes）以前不认模式，把 `--fake-prefill` 的 TTFT 当真实
    能力写进表、「SLO 要求」列也不改文案；而同一份 state 走 --all-modes 却全是 "—"。
    有了这个，两条路出一张表的规则才一致。

    老 report 没这个字段（那时只有 full），认不出的值也一律当 full —— 跟没带
    mode 时的老行为完全一样。
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            mode = (json.load(f) or {}).get("mode")
    except Exception:
        return MODE_FULL
    return mode if mode in MODE_ORDER else MODE_FULL


def load_report(path: Path, state_dir: Path, use_states: bool = True,
                mode: str = MODE_FULL) -> dict:
    """读 report.json 与可选的 state_<scene>[_<mode>].json，以 scene_key->optimal 返回。

    state 文件名带模式后缀，跟 load_mode_states / load_all_tests 一套规则 ——
    否则 `--report report_donly.json` 会去读 full 的 state_<场景>.json，把上一个模式的
    行混进 donly 的表里（再被按 donly 的规则屏蔽掉，等于拿错数还不显示）。
    """
    res: dict[str, dict] = {}
    # 1. 先加 state_<scene>[_<mode>].json 里的趁手值
    if use_states:
        sfx = "" if mode == MODE_FULL else f"_{mode}"
        for sc in SCENES:
            p = state_dir / f"state_{sc['key']}{sfx}.json"
            if not p.exists():
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    st = json.load(f)
            except Exception as e:
                print(f"[警告] state 读取错误 {p}: {e}", file=sys.stderr)
                continue
            opt = _pick_best_in_state(st)
            if opt:
                res[sc["key"]] = {
                    **opt,
                    "scene_name": sc["name"],
                    "converged": st.get("converged", False),
                    "num_tests": len(st.get("tests", [])),
                    "_source":    "state",
                }
            else:
                # 没有 satisfied 点，仅记录元数据（表格在表格里会显示“未优化”）。
                # latency_stat 照样带上 —— 一个场景都没达标时，全表就靠它把表头
                # 的判定统计量写对（见 _summary_stats 的 fallback 说明）。
                res[sc["key"]] = {
                    "latency_stat": st.get("latency_stat"),
                    "converged": False,
                    "num_tests": len(st.get("tests", [])),
                    "_source":   "state",
                }
    # 2. report.json 里的 optimal 优先覆盖
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            rep = json.load(f)
        # 整份 report 的口径（每条 result 上也有，取不到时用它兜底）
        rep_stat = rep.get("latency_stat")
        for r in rep.get("results", []):
            opt = r.get("optimal") or {}
            old = res.get(r["scene_key"], {})
            res[r["scene_key"]] = {
                **opt,
                "scene_name": r.get("scene_name", r["scene_key"]),
                # 没搜到达标点时 optimal 是空的，但这条 result 顶层照样记着当初的
                # 判定口径 —— 出表时一条结果都没有就靠它写表头（见 _summary_stats）
                "latency_stat": opt.get("latency_stat") or r.get("latency_stat") or rep_stat,
                "converged":  r.get("converged", False),
                "num_tests":  r.get("num_tests", len(old.get("tests", [])))
                                or old.get("num_tests"),
                "_source":    "report",
            }
    elif not res:
        print(f"[警告] report.json 不存在且未发现 state_*.json: {path}",
              file=sys.stderr)
    return res


# ──────────────────────────────────────────────────────────
# 构造一行 CSV 数据（友商格式）
# ──────────────────────────────────────────────────────────
def _fmt_ms_to_s(val_ms) -> str:
    """None → ''; 否则 ms 除以 1000，保留 4 位小数。"""
    if val_ms is None:
        return ""
    return f"{val_ms / 1000.0:.4f}"


def _fmt_rate(rate_raw) -> str:
    """请求频率列：友商例子是「5」「2.43」「0.22」这样的干净写法，避免结尾的「5.0」。"""
    if rate_raw is None:
        return ""
    if isinstance(rate_raw, float):
        return str(int(rate_raw)) if rate_raw.is_integer() else str(round(rate_raw, 4))
    return str(rate_raw)


def _fmt_thr(val) -> str:
    if val is None:
        return ""
    # 友商示例精度 4-7 位，这里保留 7 位让用户后续自取
    return f"{val:.7f}" if isinstance(val, float) else f"{val:.4f}"


def _fmt_qps(val) -> str:
    """QPS 列 = client 实测的请求吞吐（req/s）。"""
    if val is None:
        return ""
    return f"{val:.4f}" if isinstance(val, float) else str(val)


# 场景名尾部的长度括号，如 "对话场景(4K/1K)" —— 长度已经单独成列（见 _io_cell），
# 名字里再留一份就重复了。只认「数字/数字」形式，免得误伤将来带括号的场景名。
_LEN_PAREN_RE = re.compile(r"\s*[(（]\s*\d+K?\s*/\s*\d+K?\s*[)）]\s*$")


def _scene_short_name(scene: dict) -> str:
    """「测试用例」列 = 场景名去掉尾部的长度括号。

    scenes_custom.py 那套名字里没有括号（"1K ISL / 1K OSL"），原样保留即可。
    """
    name = str(scene.get("name") or scene.get("key") or "")
    return _LEN_PAREN_RE.sub("", name).strip()


def _fmt_len(n) -> str:
    """长度写成友商那种紧凑写法：4096→4K、7168→7K；除不尽就写原值（512→512、200→200）。"""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return ""
    return f"{n // 1024}K" if n >= 1024 and n % 1024 == 0 else str(n)


def _io_cell(scene: dict) -> str:
    """「输入/输出」列 —— 从场景字段现算，不解析场景名。

    场景名的括号是手写的，scenes_custom.py 那套根本没有括号，按字段算两边都对得上。
    写的是**场景配置**的长度，不是本次实际发出的：ponly 的输出会被压到 1 个 token，
    但这一列描述的是测试用例本身（跟原来「测试用例」列里的 (4K/1K) 是同一个数）。
    """
    i, o = _fmt_len(scene.get("input_len")), _fmt_len(scene.get("max_out_len"))
    return f"{i}/{o}" if (i and o) else ""


def _slo_cell(scene: dict, mode: str | None = None) -> str:
    """「SLO 要求」列 = TTFT / TPOT 两条要求拼成**一行**（原来是一个格子里换行）。

    mode 为 donly/ponly 时，把**不考核的那一轴**的 SLO 文本换成说明，
    避免 P/D only 表里出现"看着像超标其实不判"的误导。
    """
    ttft_txt = scene.get("slo_ttft_text", "")
    tpot_txt = scene.get("slo_tpot_text", "")
    if mode and mode != MODE_FULL:
        check_ttft, check_tpot = mode_axes(mode)
        if not check_ttft:
            ttft_txt = f"（{mode} 模式不考核 TTFT）"
        if not check_tpot:
            tpot_txt = f"（{mode} 模式不考核 TPOT）"
    return "；".join(t for t in (ttft_txt, tpot_txt) if t)


def _summary_values(cols: list[tuple], scene: dict, meta: dict,
                    opt: dict | None, mode: str | None) -> dict:
    """结论表一行里各列的值 —— 键就是 summary_cols() 里的取值键。

    时延列的取值键是 (轴, 统计量)，**按表头给的那个统计量取数**，不是按这一行
    当初判定用的口径 —— 两处不一致时（--latency-stat 换过又续跑），表头写 P90
    而格子填 P95 更糟，`_summary_stats` 会另外打警告。
    """
    has_opt = bool(opt and opt.get("batch") is not None)

    # 本模式不考核的那一轴写 "—"：bench 打出来的数字是测试脚手架的产物，不是这台机器的能力
    #   · donly 加了 --fake-prefill，TTFT 反映的是"跳过 prefill"后的排队时间
    #   · ponly 输出只压到 1 个 token，请求内没有 ITL，TPOT 恒 0、E2E 也就等于 TTFT
    # 照抄进表里会被读成"这个轴很快"，所以和「SLO 要求」列的（不考核 XX）标注保持一致。
    check_ttft, check_tpot = mode_axes(mode) if mode else (True, True)
    checked = {"ttft": check_ttft, "tpot": check_tpot,
               "e2el": check_ttft and check_tpot}

    v: dict = {
        "model":     meta["model"],
        "mode":      mode or "",
        "case":      _scene_short_name(scene),
        "io":        _io_cell(scene),
        "slo":       _slo_cell(scene, mode),
        "precision": meta["precision"],
    }

    # 时延列：这一行要用到哪些 (轴, 统计量) 由列定义决定，逐个取数
    lat_keys = [k for _, k in cols if isinstance(k, tuple)]
    if not has_opt:
        for k in lat_keys:
            v[k] = ""
        return {**v, "num_prompts": "", "rate": "", "conc": "",
                "qps": "", "out_tok": "", "total_tok": ""}

    batch    = opt.get("batch", "")
    rate_raw = opt.get("rate")
    # donly/ponly 的 rate 是打满：client 传的是 --request-rate inf，
    # 表里就写 inf（不再写「打满」，免得跟真实数值混在一起看不清）。
    # full 也一样：rate=0 就是"不限速"那些点（第一段上探、或退回打满的兜底），
    # 写 0 会被当成"实测频率 0"，写 inf 才是它真正的含义。
    v["rate"] = "inf" if (opt.get("rate_saturated") or not rate_raw) else _fmt_rate(rate_raw)
    v["num_prompts"] = opt.get("num_prompts") or (
        batch * DEFAULT_PROMPTS_MULT if isinstance(batch, int) else "")
    max_conc = opt.get("max_concurrency")
    # ais_bench 报表中"最大并发"即为 AIS_BATCH_SIZE (客户端一次拋 inflight 请求数)
    # — 该列友商总是整数，本脚本以 ais_bench 测出的 Max Concurrency 优先，缺省回退 batch.
    if not max_conc and isinstance(batch, int):
        max_conc = batch
    if max_conc is None:
        max_conc = ""
    elif isinstance(max_conc, float):
        max_conc = int(max_conc) if max_conc.is_integer() else max_conc
    v["conc"]     = max_conc
    v["qps"]      = _fmt_qps(opt.get("req_throughput"))
    v["out_tok"]  = _fmt_thr(opt.get("out_tok_thr"))
    v["total_tok"] = _fmt_thr(opt.get("total_tok_thr"))

    for key in lat_keys:
        axis, stat = key
        v[key] = ("—" if not checked.get(axis, True)
                  else _fmt_ms_to_s(entry_lat_ms(opt, axis, stat)))

    # 这一行是"本轮最优"，但它自己就没达标时得说清楚 —— 否则一张表里
    # 摆着一个超 SLO 的数字，看着像达标结果。全部测点见 metax_report_<模型>.csv
    if opt.get("satisfied") is False:
        v["slo"] += "（本场景没有达标的点，此行是已测点里吞吐最高的）"
    return v


def build_row(cols: list[tuple], scene: dict, meta: dict,
              opt: dict | None, mode: str | None = None) -> list:
    """按 summary_cols() 的列定义产出一条数据行（列数/顺序与表头一一对应）。"""
    v = _summary_values(cols, scene, meta, opt, mode)
    return [v.get(k, "") for _, k in cols]


# ──────────────────────────────────────────────────────────
# 写 CSV
# ──────────────────────────────────────────────────────────
def write_csv(path: Path, meta: dict, report_by_scene: dict, scenes_order: list,
              mode: str = MODE_FULL):
    """单模式结论表。mode 决定「不考核的那一轴写 —」和 SLO 列的文案 —— 由调用方从
    report 顶层读出来传进来（见 report_mode），默认 full 就是老行为。"""
    ttft_stat, tpot_stat = _summary_stats([o for o in report_by_scene.values() if o])
    cols = summary_cols(merged=False, ttft_stat=ttft_stat, tpot_stat=tpot_stat)
    # utf-8-sig: 带 BOM，Excel 直接双击打开才不会把中文显示成乱码
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        # 单行表头（原来多一行合并的分组行，现在只有列名 —— 跟全测点表一致，
        # Excel/pandas 打开不用绕开第 2 行）
        w.writerow([c[0] for c in cols])
        first = True
        for sc in scenes_order:
            opt = report_by_scene.get(sc["key"])
            row = build_row(cols, sc, meta, opt, mode=mode)
            # 友商报表里第一行带 "模型" 名；其余行首留空 (合并单元格效果)
            if not first:
                row[0] = ""
            w.writerow(row)
            first = False
    print(f"\n  CSV 写出: {path}")
    print(f"  共 {len(scenes_order)} 条场景数据")


def _summary_stats(opts: list, fallback: list | None = None) -> tuple[str, str]:
    """结论表判定列用哪两个统计量**键**（不是标签）。口径一致就照它写；混了警告并用第一条。

    混用时的后果比原来轻：表头写的是 P90，每一行也都按 P90 取数（见 _summary_values），
    不会再出现"表头 P90、格子填 P95"。但**当初判定 SLO 用的口径**确实不一样，
    「SLO 要求」列的对照关系得自己心里有数，所以照样警告一声。

    fallback 只在 opts 里**一个能问的都没有**时才查（opts 为空，或全是连 latency_stat
    都没记的老数据）。以前这里直接兜底 P90 —— 于是「一条结果都没搜到」时表头写
    「P90 首token平均时延（s）」、同一行的「SLO 要求」却写 `ttft_p95 < 3s`，自相矛盾；
    而没搜到恰恰常常是因为 --latency-stat p95 取不到值（client 没打 P95 补丁），
    兜底方向正好反了。fallback 传同一批数据里**没结果的那些 entry**，它们虽然没
    batch，latency_stat 是照实记的，拿它兜底才能跟运行侧的「判定统计量 P95」对上。
    """
    stats = [entry_stat_of(o) for o in opts if o]
    if not stats:
        stats = [entry_stat_of(o) for o in (fallback or []) if o]
    if not stats:
        return DEFAULT_LATENCY_STAT, DEFAULT_LATENCY_STAT
    ttfts = {s.ttft for s in stats}
    tpots = {s.tpot for s in stats}
    if len(ttfts) > 1 or len(tpots) > 1:
        print(f"[警告] 表里的判定统计量不一致 (TTFT {sorted(ttfts)} / TPOT {sorted(tpots)})"
              f" —— 表头和取值都按第一条的 "
              f"{stat_label(stats[0].ttft)}/{stat_label(stats[0].tpot)} 走，"
              f"但这些结果当初不是同一个口径判出来的", file=sys.stderr)
    return stats[0].ttft, stats[0].tpot


def write_csv_multi_mode(path: Path, meta: dict, by_mode: dict,
                         scenes_order: list, modes: list):
    """结论表：每行一个「场景 × 模式」的最优结果，多一列「模式」。

    只有真正跑出结果 (有 optimal.batch) 的 场景×模式 组合才出行 —— 没跑的模式不占位，
    这样表里出现的每一行都是真实测过的数据。全量测点在 metax_report_<模型>.csv。
    """
    pairs = []
    for sc in scenes_order:
        for mode in modes:
            opt = (by_mode.get(mode) or {}).get(sc["key"])
            if not opt or opt.get("batch") is None:
                continue
            pairs.append((sc, mode, opt))

    # 列定义得先有才能出行（时延列的取值键取决于判定统计量），所以先定列再拼行。
    # pairs 只收 batch 不为 None 的，全没搜到时它是空的 —— 再拿一遍没结果的那些
    # entry 兜底，否则表头会兜底成 P90，跟同一行的「SLO 要求」打架（见 _summary_stats）
    all_opts = [(by_mode.get(m) or {}).get(sc["key"])
                for sc in scenes_order for m in modes]
    ttft_stat, tpot_stat = _summary_stats([o for _, _, o in pairs], fallback=all_opts)
    cols = summary_cols(merged=True, ttft_stat=ttft_stat, tpot_stat=tpot_stat)
    rows = [build_row(cols, sc, meta, opt, mode=mode) for sc, mode, opt in pairs]

    # utf-8-sig: 带 BOM，Excel 直接双击打开才不会把中文显示成乱码
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([c[0] for c in cols])   # 单行表头，见 write_csv 里的说明
        for r in rows:
            w.writerow(r)          # 结论表每行都带模型名（方便筛选/透视），不空首行

    print(f"\n  CSV 写出 (结论表): {path}")
    print(f"  共 {len(rows)} 条数据行 = 场景 × 模式   判定统计量: "
          f"TTFT {stat_label(ttft_stat)} / TPOT {stat_label(tpot_stat)}")
    for mode in modes:
        cov = sum(1 for sc in scenes_order
                  if ((by_mode.get(mode) or {}).get(sc["key"]) or {}).get("batch") is not None)
        print(f"    {mode:<6}: {cov}/{len(scenes_order)} 个场景有结果")
    return rows


def aggregate_all_modes(state_dir: Path,
                        out_path: Path | None = None,
                        model_name: str | None = None,
                        all_points_path: Path | None = None,
                        precision: str | None = None) -> Path | None:
    """跑完一个模式后自动出表：同时写「结论表」和「全测点表」。

    结论表   metax_summary_<模型>.csv —— 每行一个「场景 × 模式」的最优结果，跟友商对齐
    全测点表 metax_report_<模型>.csv  —— 每个真跑过的测试点一行，带「满足要求」标记

    各模式读 report.json / report_donly.json / report_ponly.json 里的最优结果；
    没有 report（搜索被中断 / 只手工存了 state）就退回从 state 里取。
    全测点表一律从 state_*.json 读（report 里只有最优那一行）。
    模型短名以数据里记的为准（detect_model_name），没数据时才用 scenes.py 的默认值。
    一个模式都没跑过 → 返回 None（不写文件）。
    """
    model_name = model_name or detect_model_name(state_dir)
    meta = report_meta(model_name=model_name, precision=precision)

    by_mode: dict[str, dict] = {}
    stale: list[str] = []
    from_state: list[str] = []
    for mode in MODE_ORDER:
        p = state_dir / MODE_REPORT_FILES[mode]
        rep = {}
        if p.exists():
            # 同目录换了模型跑时，旧模型的 report_<mode>.json 还在 —— 别把它的行
            # 并进新模型的汇总表（「模型」列会张冠李戴）。模型列对不上就跳过。
            rec_model = None
            if model_name:
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        rec_model = (json.load(f) or {}).get("model")
                except Exception:
                    rec_model = None
            if rec_model and rec_model != model_name:
                stale.append(f"{p.name}({rec_model})")
                continue
            rep = load_report(p, state_dir, use_states=False, mode=mode)
        else:
            # 没有 report（搜索被中断 / 只手工存了 state）→ 从 state 里取结论
            rep = load_mode_states(state_dir, mode, model_name)
            if rep:
                from_state.append(mode)
        if rep:
            by_mode[mode] = rep
    if stale:
        print(f"[警告] 跳过其它模型的 report: {', '.join(stale)}"
              f"（本次模型 {model_name}）—— 换模型重跑时请删掉这些文件，"
              f"或用 --state-dir 按模型分目录", file=sys.stderr)
    if from_state:
        print(f"[提示] 没有 report_<模式>.json，{', '.join(from_state)} 的结论读自 "
              f"state_*.json（可能是搜索被中断；未收敛的场景取的是已达标的点里吞吐最高的）",
              file=sys.stderr)

    name = model_name or MODEL_NAME

    # 全测点表：从 state 读，独立于 report —— 搜索还没收敛（report 未生成）时也有表看
    all_points = load_all_tests(state_dir, model_name)
    if all_points:
        ap_out = all_points_path or (state_dir / f"metax_report_{name}.csv")
        write_all_points_csv(ap_out, meta, all_points)

    if not by_mode:
        return None

    out = out_path or (state_dir / f"metax_summary_{name}.csv")
    write_csv_multi_mode(out, meta, by_mode, SCENES,
                         [m for m in MODE_ORDER if m in by_mode])
    return out




def print_table(meta: dict, report_by_scene: dict, scenes_order: list,
                title: str = "METAX 最优结果报表", mode: str | None = None):
    """在 stdout 打印人类可读的形式，便于快速检视。

    mode 不为 None 时，不考核的那一轴（donly 的 TTFT / ponly 的 TPOT）显示 "—"，
    与 CSV 一致 —— 那几个数不是这台机器在该轴上的能力。
    """
    print("\n" + "=" * 110)
    print(f"  {title}")
    print("=" * 110)
    print(f"{'场景':<26} {'precision':<10} {'batch':<6} {'rate':<6} "
          f"{'TTFT_P90(ms)':<13} {'TPOT_P90(ms)':<13} {'集群 Out tok/s':<15} SLO")
    print("-" * 110)
    for sc in scenes_order:
        opt = report_by_scene.get(sc["key"], {})
        # 判断是否真正拿到最优结果（optimal 字段存在且非空）
        has_opt = bool(opt and opt.get("batch") is not None)
        if not has_opt:
            print(f"  {sc['name'][:24]:<26} "
                  f"{str(meta['precision']):<10} "
                  f"{'—':<6} {'—':<6}          (未优化)")
            continue
        sat = opt.get("satisfied")
        sat_str = "✓" if sat else ("✗（未达 SLO）" if sat is False else "")

        # 本模式不考核的轴 → "—"（数字不可比：donly 的 TTFT 是 fake-prefill 下的，
        # ponly 的输出只有 1 个 token，TPOT 恒为 0）
        check_ttft, check_tpot = mode_axes(mode) if mode else (True, True)

        def _v(key):        # None / 缺失 都显示 "—"
            v = opt.get(key)
            return "—" if v is None else v

        def _lat_v(axis, stat):
            """None 才显示 "—"。不能用 `or "—"` —— 真实的 0.0 会被当成缺失，
            控制台写 "—" 而 CSV 写 "0.0000"，两边对不上（而且 "—" 现在是「不考核」
            的意思，撞在一起更说不清）。"""
            ms = entry_lat_ms(opt, axis, stat)
            return "—" if ms is None else ms

        # 跟 CSV 那条（build_report 的 v["rate"]）同一套判据：rate=0.0 也是 inf。
        # 老 state/report 里"full 退回打满点"那条路 rate_saturated 记的还是 False，
        # 只看它会把控制台写成「0.0」而 CSV 写「inf」。
        rate = "inf" if (opt.get("rate_saturated") or not opt.get("rate")) else _v("rate")
        # 取该结果判定时用的统计量（老数据没有这字段 → P90），别再写死 p90 键名
        opt_stat = entry_stat_of(opt)
        ttft_v = "—" if not check_ttft else _lat_v("ttft", opt_stat.ttft)
        tpot_v = "—" if not check_tpot else _lat_v("tpot", opt_stat.tpot)
        row = (f"  {sc['name'][:24]:<26} "
               f"{str(meta['precision']):<10} "
               f"{str(_v('batch')):<6} "
               f"{str(rate):<6} "
               f"{str(ttft_v):<13} "
               f"{str(tpot_v):<13} "
               f"{str(_v('out_tok_thr')):<15} "
               f"{sat_str}")
        print(row)
    print("=" * 110)


# ──────────────────────────────────────────────────────────
# 全测点表：把每个真跑过的测试点都列出来
# ──────────────────────────────────────────────────────────
def detect_model_name(state_dir: Path) -> str | None:
    """从 state/report 文件里认出这批数据是哪个模型跑的。

    报表「模型」列不能靠 scenes.py 里的默认值猜 —— 换模型跑过、或者数据是从别的
    机器拷过来的目录，scenes.py 的 MODEL_NAME 跟数据根本不是一回事。report/state
    里都记了跑的时候用的短名，以它为准（都没有才退回 scenes.py 的默认值）。
    """
    for mode in MODE_ORDER:
        p = state_dir / MODE_REPORT_FILES[mode]
        if not p.exists():
            continue
        try:
            nm = (json.loads(p.read_text(encoding="utf-8")) or {}).get("model")
        except Exception:
            nm = None
        if nm:
            return nm
    for p in sorted(state_dir.glob("state_*.json")):
        try:
            nm = (json.loads(p.read_text(encoding="utf-8")) or {}).get("model_name")
        except Exception:
            nm = None
        if nm:
            return nm
    return None


def load_all_tests(state_dir: Path, model_name: str | None = None) -> list:
    """扫 state_<场景>[_<模式>].json，返回全部测试点。

    返回 [{mode, scene, state, test}, …]，按「场景顺序 → 模式 → 并发」排好。
    只读 state（report 里只有最优那一行，全量表要的是每一次测试）。
    模型对不上的 state 跳过 —— 同目录换过模型时，别把两个模型的数据并进一张表。
    """
    by_scene_mode: dict[tuple[str, str], list] = {}
    for sc in SCENES:
        for mode in MODE_ORDER:
            sfx = "" if mode == MODE_FULL else f"_{mode}"
            p = state_dir / f"state_{sc['key']}{sfx}.json"
            if not p.exists():
                continue
            try:
                st = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[警告] state 读取失败 {p}: {e}", file=sys.stderr)
                continue
            # 文件名只是线索，以 state 自己记的为准（换模式跑过 / 手工改过名都不怕）
            st_mode = st.get("mode", MODE_FULL)
            if st_mode != mode:
                continue
            if model_name and st.get("model_name") and st["model_name"] != model_name:
                continue
            tests = [t for t in st.get("tests", []) if t.get("batch") is not None]
            if tests:
                by_scene_mode[(sc["key"], mode)] = [st, tests]

    out = []
    for sc in SCENES:                                  # 场景顺序跟 scenes.py 一致
        for mode in MODE_ORDER:
            hit = by_scene_mode.get((sc["key"], mode))
            if not hit:
                continue
            st, tests = hit
            for t in sorted(tests, key=lambda x: (x.get("batch", 0), x.get("rate", 0))):
                out.append({"mode": mode, "scene": sc, "state": st, "test": t})
    return out


def _is_optimal(state: dict, test: dict) -> bool:
    opt = state.get("optimal") or {}
    if opt.get("batch") is None:
        return False
    return (opt.get("batch") == test.get("batch")
            and abs((opt.get("rate") or 0.0) - (test.get("rate") or 0.0)) < 1e-9)


def build_all_points_row(scene: dict, mode: str, state: dict, t: dict, meta: dict) -> list:
    """一行 = 一个测试点。时延取判定口径（老 state 没有 → P90）。

    口径以 **state 的**为准，不是每个测点自己记的那个：换过 --latency-stat 续跑时，
    复用的老测点会被按新口径重新判定（search._refresh_test），但没被复用的点还留着
    上一次的口径 —— 逐点取会让同一张表里混着 P90 和 Mean，跟「满足要求」列的判定
    口径对不上。state 的口径 = 这批结论真正用的那个。
    """
    stat = entry_stat_of(state) if state.get("latency_stat") else entry_stat_of(t)
    batch = t.get("batch", "")

    # rate=0 是"不限速"（rate 打满的跑法恒为 0；搜 rate 的跑法第一段上探、第二段退回
    # 打满的兜底也是 0），一律写 inf —— 写成 0 会被当成"实测频率 0"。
    # 判据用**这个点自己的** rate，不用 state 级的 rate_saturated：那是"结论是不是
    # 打满点"，而一个搜过 rate 的 state 里本来就躺着有限 rate 的测点 —— 第二段一个
    # 合格点都没找到、结论退回打满点时就同时有两种，拿 state 判会把那些点也写成 inf。
    rate = "inf" if not t.get("rate") else _fmt_rate(t.get("rate"))
    max_conc = t.get("max_concurrency")
    if max_conc is None:
        max_conc = batch
    elif isinstance(max_conc, float):
        max_conc = int(max_conc) if max_conc.is_integer() else max_conc

    # 本模式不考核的那一轴写 "—"，跟结论表同一套规则（donly 的 TTFT 是 fake-prefill
    # 下的排队时间，ponly 的 TPOT 恒 0、E2E 就等于 TTFT）—— 这张表是逐点明细，
    # 混进这些数字更容易被当成"这台机器在那个轴上很快"。
    check_ttft, check_tpot = mode_axes(mode)
    checked = {"ttft": check_ttft, "tpot": check_tpot,
               "e2el": check_ttft and check_tpot}

    def _lat(axis: str, st: str, fmt) -> str:
        if not checked[axis]:
            return "—"
        ms = entry_lat_ms(t, axis, st)
        return "—" if ms is None else fmt(ms)

    _s   = lambda ms: f"{ms / 1000:.4f}"      # TTFT / E2E 用秒
    _ms  = lambda ms: f"{ms:.2f}"             # TPOT 用毫秒

    if not t.get("parse_ok", True):
        verdict = "—（客户端没跑成）"
    elif t.get("satisfied"):
        verdict = "是"
    else:
        verdict = "否"

    notes = []
    if t.get("reject_reason"):
        notes.append(t["reject_reason"])
    if not t.get("parse_ok", True) and t.get("error"):
        notes.append(str(t["error"]).splitlines()[0])
    if not state.get("converged", False):
        notes.append("本场景搜索未完成（未收敛），此行是中途测点")
    if not notes and _is_optimal(state, t) and t.get("satisfied") is False:
        notes.append("本场景最优点也没达标（1 并发就超 SLO）")

    return [
        mode,
        meta["model"],
        _scene_short_name(scene),
        _io_cell(scene),
        _slo_cell(scene, mode),
        meta["precision"],
        stat.label(),
        rate,
        max_conc,
        t.get("num_prompts", ""),
        _fmt_qps(t.get("req_throughput")),
        # TTFT / TPOT 各三列：第一列取「判定统计量」列那个分位，Mean / P99 固定。
        # E2E 只有两列 —— 它不参与 SLO 判定，没有「判定统计量」，也就没有第一列
        # （见 ALL_POINTS_HEADER 上的说明）。
        _lat("ttft", stat.ttft, _s), _lat("ttft", "mean", _s), _lat("ttft", "p99", _s),
        _lat("tpot", stat.tpot, _ms), _lat("tpot", "mean", _ms), _lat("tpot", "p99", _ms),
        _lat("e2el", "mean", _s), _lat("e2el", "p99", _s),
        _fmt_thr(t.get("in_tok_thr")),
        _fmt_thr(t.get("out_tok_thr")),
        _fmt_thr(t.get("total_tok_thr")),
        verdict,
        "★" if _is_optimal(state, t) else "",
        "  ".join(notes),
    ]


def write_all_points_csv(path: Path, meta: dict, tests: list) -> list:
    """写全测点表；返回写出的行。utf-8-sig 同结论表（Excel 双击不乱码）。"""
    rows = [build_all_points_row(rec["scene"], rec["mode"], rec["state"],
                                 rec["test"], meta)
            for rec in tests]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(ALL_POINTS_HEADER)
        w.writerows(rows)
    print(f"\n  CSV 写出 (全测点): {path}")
    print(f"  共 {len(rows)} 行 = 每个真跑过的测试点一行")
    if rows:
        # 按表头找「满足要求」列，别写下标 —— 加列以后下标就变了
        vi = ALL_POINTS_HEADER.index("满足要求")
        ok = sum(1 for r in rows if r[vi] == "是")
        print(f"    达标 {ok} 行 / 未达标 {sum(1 for r in rows if r[vi] == '否')} 行"
              f" / 未跑成 {sum(1 for r in rows if r[vi].startswith('—'))} 行")
    return rows


def aggregate_all_points(state_dir: Path,
                         out_path: Path | None = None,
                         model_name: str | None = None,
                         precision: str | None = None) -> Path | None:
    """把该模型跑过的**所有**测试点汇总成一张表（metax_report_<模型>.csv）。"""
    model_name = model_name or detect_model_name(state_dir)
    meta = report_meta(model_name=model_name, precision=precision)
    tests = load_all_tests(state_dir, model_name)
    if not tests:
        return None
    name = model_name or MODEL_NAME
    out = out_path or (state_dir / f"metax_report_{name}.csv")
    write_all_points_csv(out, meta, tests)
    return out


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report",       default="report.json",
                   help="auto_tune.py 输出的 report.json 路径")
    p.add_argument("--out",          default=None,
                   help="结论表输出路径（缺省: 单模式 metax_report.csv; "
                        "--all-modes 为 metax_summary_<模型>.csv）")
    p.add_argument("--out-points",   default=None,
                   help="全测点表输出路径（缺省: metax_report_<模型>.csv）")
    p.add_argument("--all-modes",    action="store_true",
                   help="出两张表：结论表 metax_summary_<模型>.csv（每行一个「场景 × 模式」的"
                        "最优结果，多一列「模式」）+ 全测点表 metax_report_<模型>.csv"
                        "（每个跑过的测试点一行，带「满足要求」标记）")
    p.add_argument("--all-points",   action="store_true",
                   help="只出全测点表 metax_report_<模型>.csv（读 state_*.json 里的每一次测试）")
    p.add_argument("--state-dir",    default="",
                   help="查找 state_<scene>.json 的目录（默认 同目录，中途查看用）")
    p.add_argument("--no-states",    action="store_true",
                   help="忽略 state_*.json，仅读 report.json")
    p.add_argument("--precision",    default=None,
                   help=f"报表 precision 列 (e.g. INT8 / FP16 / W8A8)，缺省 {DEFAULT_PRECISION}")
    p.add_argument("--scenes-file", "--scenes", default="", metavar="FILE",
                   dest="scenes_file",
                   help="换一套场景配置（如 scenes_custom.py），缺省用同目录的 scenes.py。"
                        "跟跑测时给 auto_tune.py 的那个要一致，否则场景对不上、表会是空的")
    p.add_argument("--model-name",  default=None,
                   help=f"覆盖报表「模型」列 (默认取 scenes.MODEL 的 basename: {MODEL_NAME})")
    p.add_argument("--print-table",  action="store_true",
                   help="写 CSV 同时在 stdout 打印人类可读的对照表")
    p.add_argument("--scene",        default=None,
                   help="只输出指定 scene_key 对应的行；缺省全部")
    return p.parse_args()


def main():
    args = parse_args()
    here = Path(__file__).parent.resolve()

    state_dir = Path(args.state_dir) if args.state_dir else here

    # ── 全测点表（每个跑过的测试点一行）──────────────
    if args.all_points and not args.all_modes:
        out = Path(args.out or args.out_points) if (args.out or args.out_points) else None
        merged = aggregate_all_points(state_dir, out_path=out,
                                      model_name=args.model_name,
                                      precision=args.precision)
        if merged is None:
            print(f"[警告] 没找到任何 state_*.json（{state_dir}）—— 先跑 auto_tune.py",
                  file=sys.stderr)
        return

    # ── 两张表：结论表（每行一个 场景×模式）+ 全测点表 ──────────────
    if args.all_modes:
        out = Path(args.out) if args.out else None
        pts = Path(args.out_points) if args.out_points else None
        merged = aggregate_all_modes(state_dir, out_path=out,
                                     model_name=args.model_name,
                                     all_points_path=pts,
                                     precision=args.precision)
        if merged is None and not args.all_points:
            print("[警告] 没找到任何 report (report.json / report_donly.json / "
                  "report_ponly.json) —— 先跑 auto_tune.py", file=sys.stderr)
            return
        if args.print_table:
            args.model_name = args.model_name or detect_model_name(state_dir)
            meta = report_meta(model_name=args.model_name, precision=args.precision)
            scenes_order = [s for s in SCENES if not args.scene or s["key"] == args.scene]
            for mode in MODE_ORDER:
                p = state_dir / MODE_REPORT_FILES[mode]
                if not p.exists():
                    continue
                rep = load_report(p, state_dir, use_states=False, mode=mode)
                if rep:
                    print_table(meta, rep, scenes_order,
                                title=f"METAX 最优结果报表 — mode={mode}", mode=mode)
        return

    meta = report_meta(model_name=args.model_name, precision=args.precision)
    # 这份 report 是哪个模式的 —— 决定不考核的那一轴写 "—"、SLO 列的文案，
    # 以及去读哪个 state_<场景>[_<模式>].json
    mode = report_mode(here / args.report)
    rep_by_scene = load_report(here / args.report, state_dir,
                               use_states=not args.no_states, mode=mode)

    scenes_order = [s for s in SCENES if not args.scene or s["key"] == args.scene]

    out_path = here / (args.out or "metax_report.csv")
    write_csv(out_path, meta, rep_by_scene, scenes_order, mode=mode)

    if args.print_table:
        print_table(meta, rep_by_scene, scenes_order, mode=mode)

    if not rep_by_scene:
        print("\n[提示] report.json 与 state_*.json 都未发现 —— 表格的性能列留空。",
              file=sys.stderr)
        print("[提示] 先运行 auto_tune.py (...) 开始搜索，生成 state_<scene>.json 后可随时代重跑。",
              file=sys.stderr)
    elif not Path(here / args.report).exists():
        print("\n[提示] report.json 还未生成（搜索未完成）—— 表格读自 state_*.json 的 ``当前最佳``。",
              file=sys.stderr)
        print("[提示] 中途中随时查看友好：");
        print(f"         python3 generate_report.py --print-table")
        print("[提示] auto_tune.py 跑完后会生成 report.json，含最终 optimal。",
              file=sys.stderr)


if __name__ == "__main__":
    main()
