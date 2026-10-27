#!/usr/bin/env python3
"""
场景配置 —— 本工具唯一的事实来源。

改 SLO、输入/输出长度、用哪个压测 client、加新场景，都只改这个文件。

一个场景 = 一次 make_scene(...)，字段:
  key            短名（命令行 / 日志文件名 / state_<key>.json 都用它）
  name           长名（报表「测试用例」列第一行）
  input_len      输入长度（token）；client=bench_serving 时按它生成 random 输入
  max_out_len    输出最大 token 数（ponly 会被压到 1）
  cached_input_len / non_cached_input_len
                 ponly_cached / full_cached 专用：共享前缀命中长度和非命中长度；
                 命中率 = cached / (cached + non_cached)，且两者之和应等于 input_len。
                 `none_cached_input_len` 也兼容（历史拼写）。
  output_len     cache 模式专用（可选）：送进 --gsp-output-len；
                 不填时使用 max_out_len。
  slo_ttft       TTFT 上限（秒），-1 = 不限制
  slo_tpot       TPOT 上限（秒），-1 = 不限制
  client         压测 client: ais_bench / bench_serving（缺省 ais_bench）
  dataset        ais_bench 读的自定义数据集 jsonl（client=ais_bench 时必填）
  slo_ttft_text / slo_tpot_text
                 报表「测试用例」列里的 SLO 文本，缺省按数值自动生成
  seed_batch / seed_rate（可选，不填也行）
                 友商的 batch / 请求率 —— **仅供参考、不参与搜索**，搜索完全靠
                 自己上探（并发从 1 起倍增，rate 按输入长度折算基准值再二分）。
                 新场景没有友商数据就不用填。

新增场景: 照抄一条改 key / name / 长度 / SLO 即可，其余字段走缺省。
donly / ponly / ponly_cached / full_cached 模式固定用 bench_serving，不看这里的 client。
cache 模式使用 generated-shared-prefix 数据集，每次实际测试会生成唯一 seed。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# ── 压测 client ──────────────────────────────────────────────────────────
#   ais_bench            读场景 jsonl（--custom-dataset-path），rate 参与搜索
#   sglang.bench_serving 按 input_len / max_out_len 生成 random 请求
# 两者能打出的时延统计量不同，见 runner.CLIENT_LATENCY_STATS。
CLIENT_AIS   = "ais_bench"
CLIENT_BENCH = "sglang.bench_serving"
CLIENTS = (CLIENT_AIS, CLIENT_BENCH)

# 命令行 --client 的宽松写法 → 内部名
CLIENT_ALIASES = {
    "ais": CLIENT_AIS, "aisbench": CLIENT_AIS, CLIENT_AIS: CLIENT_AIS,
    "bench": CLIENT_BENCH, "bench_serving": CLIENT_BENCH,
    "sglang": CLIENT_BENCH, CLIENT_BENCH: CLIENT_BENCH,
}


def normalize_client(name: str) -> str:
    """把 --client 的写法归一化；不认识的保持原样，交给 validate_scenes 报错。"""
    return CLIENT_ALIASES.get((name or "").strip().lower(), (name or "").strip())


MODEL = "/path/to/MyModel/vllm_quant_model"

# 模型目录经常是 vllm_quant_model / vllm_model 这种没有区分度的占位名，
# 例如 /path/to/models/MyModel/vllm_quant_model
# —— 报表「模型」列 / 日志目录 logs/<模型>/<mode> 要的是能区分模型的短名，
# 所以遇到占位名就退一级取父目录名。
GENERIC_MODEL_DIRS = {"vllm_quant_model", "vllm_model", "quant_model",
                      "model", "models", "served_model", "hf"}


def derive_model_name(model_path: str) -> str:
    """从模型路径推一个能区分模型的短名（--model-name 可覆盖）。"""
    p = Path(str(model_path).rstrip("/"))
    name = p.name or str(model_path)
    if name in GENERIC_MODEL_DIRS and p.parent.name:
        return p.parent.name
    return name


# 短名：报表「模型」列 / 日志目录 logs/<模型>/<mode> 都用它
MODEL_NAME = derive_model_name(MODEL)

DATASET_DIR = "/path/to/datasets"

# bench_serving 虽然用 --dataset-name random 生成请求，但它的 random loader 在没给
# --dataset-path 时会去 HuggingFace 拉 ShareGPT_V3_unfiltered_cleaned_split.json
# （无外网的机器上直接 '[Errno 101] Network is unreachable' 挂掉），所以固定传本地这份。
BENCH_DATASET = "/path/to/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"


# ── 场景缺省值 ───────────────────────────────────────────────────────────
SCENE_DEFAULTS = {
    "client":       CLIENT_AIS,
    "dataset":      "",
    "slo_ttft":     -1,
    "slo_tpot":     -1,
    "slo_ttft_text": "",
    "slo_tpot_text": "",
    "cached_input_len":     None,
    "non_cached_input_len": None,
    "output_len":           None,
}


def _ttft_text(slo: float) -> str:
    return "ttft_p90 不限制" if slo < 0 else f"ttft_p90 < {slo:g}s"


def _tpot_text(slo: float) -> str:
    return "tpot_p90 不限制" if slo < 0 else f"tpot_p90 < {slo * 1000:g}ms"


def make_scene(key: str, name: str, *, input_len: int, max_out_len: int,
               seed_batch: Optional[int] = None,
               seed_rate: Optional[float] = None, **kw) -> dict:
    """建一个场景配置，缺省字段补齐、SLO 文本按数值生成。

    SLO 文本是给报表「测试用例」列用的（跟友商对齐），想写别的传
    slo_ttft_text / slo_tpot_text 覆盖即可。
    seed_batch / seed_rate 是友商数据，**不参与搜索**，没有就不填。
    """
    s = {**SCENE_DEFAULTS, "key": key, "name": name,
         "input_len": input_len, "max_out_len": max_out_len,
         "seed_batch": seed_batch, "seed_rate": seed_rate}
    s.update(kw)
    s["client"] = normalize_client(s["client"])
    if not s["slo_ttft_text"]:
        s["slo_ttft_text"] = _ttft_text(s["slo_ttft"])
    if not s["slo_tpot_text"]:
        s["slo_tpot_text"] = _tpot_text(s["slo_tpot"])
    return s


def _ds(fname: str) -> str:
    return f"{DATASET_DIR}/{fname}"


SCENES = [
    make_scene(
        "dialog", "对话场景(4K/1K)",
        input_len=4096, max_out_len=1024,
        slo_ttft=2.0, slo_tpot=0.035,
        seed_batch=100, seed_rate=4.6,
        dataset=_ds("2_4k_dialog_scene.jsonl"),
    ),
    make_scene(
        "knowledge", "知识库场景(7K/2K)",
        input_len=7168, max_out_len=2048,
        slo_ttft=4.0, slo_tpot=0.035,
        seed_batch=128, seed_rate=3.32,
        dataset=_ds("3_7k_knowledge_base_scene.jsonl"),
    ),
    make_scene(
        "content_512", "内容生成场景(512/8K)",
        input_len=512, max_out_len=8192,
        slo_ttft=2.0, slo_tpot=0.035,
        seed_batch=64, seed_rate=25,
        dataset=_ds("5_512_content_generation_scene.jsonl"),
    ),
    make_scene(
        "agent", "agent编程场景(40K/200)",
        input_len=40960, max_out_len=200,
        slo_ttft=8.0, slo_tpot=-1,              # TPOT 不限制
        seed_batch=8, seed_rate=2.1,
        dataset=_ds("1_40k_agent_programming_scene.jsonl"),
    ),
    make_scene(
        "analysis_16k", "内容分析场景(16K/2K)",
        input_len=16384, max_out_len=2048,
        slo_ttft=6.0, slo_tpot=0.035,
        seed_batch=64, seed_rate=1.44,
        dataset=_ds("4_16k_content_analysis_scene.jsonl"),
    ),
    make_scene(
        "analysis_32k", "内容分析场景(32K/200)",
        input_len=32768, max_out_len=200,
        slo_ttft=7.0, slo_tpot=0.035,
        seed_batch=30, seed_rate=8,
        dataset=_ds("7_32k_content_analysis_scene.jsonl"),
    ),
    make_scene(
        "ultra_long", "超长文本场景(128K/1K)",
        input_len=131072, max_out_len=1024,
        slo_ttft=-1, slo_tpot=-1,               # 两轴都不限制
        seed_batch=9, seed_rate=0.4,
        dataset=_ds("6_128k_ultra_long_text_scen.jsonl"),
    ),
]

SCENE_BY_KEY = {s["key"]: s for s in SCENES}


def validate_scenes(scenes: list | None = None):
    """开跑前把配置错误一次性报出来 —— 免得场景跑到一半才发现字段缺失。"""
    scenes = SCENES if scenes is None else scenes
    errs: list[str] = []
    seen: set[str] = set()
    for i, s in enumerate(scenes):
        key = s.get("key") or f"<第 {i + 1} 条没有 key>"
        if key in seen:
            errs.append(f"{key}: key 重复")
        seen.add(key)
        if s.get("client") not in CLIENTS:
            errs.append(f"{key}: client={s.get('client')!r} 不认识，"
                        f"可选 {' / '.join(CLIENTS)}")
        elif s["client"] == CLIENT_AIS and not s.get("dataset"):
            errs.append(f"{key}: client=ais_bench 必须给 dataset（场景 jsonl 路径）")
        # seed_batch / seed_rate 不在必填里 —— 它们只是友商参考值，不参与搜索，
        # 新场景没有友商数据也能加。
        for f in ("name", "input_len", "max_out_len"):
            if not s.get(f):
                errs.append(f"{key}: 缺字段 {f}")
        cached = s.get("cached_input_len")
        noncached = s.get("non_cached_input_len", s.get("none_cached_input_len"))
        if (cached is None) != (noncached is None):
            errs.append(
                f"{key}: cached_input_len / non_cached_input_len 必须同时配置"
            )
        elif cached is not None:
            try:
                total = int(cached) + int(noncached)
            except (TypeError, ValueError):
                errs.append(f"{key}: cache 长度必须是整数")
            else:
                if total <= 0:
                    errs.append(f"{key}: cache 长度之和必须大于 0")
                if int(s.get("input_len") or 0) != total:
                    errs.append(
                        f"{key}: input_len={s.get('input_len')} 与 "
                        f"cached + non_cached={total} 不一致"
                    )
    if errs:
        raise ValueError("scenes.py 配置有误:\n  " + "\n  ".join(errs))


validate_scenes()
