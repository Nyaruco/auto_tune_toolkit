#!/usr/bin/env python3
"""
自定义场景配置 —— 另一张 SLO 表（P95）对应的 5 个场景。

**不指就不会生效**，缺省跑的仍是 scenes.py：
    python3 auto_tune.py --mode full --scenes-file scenes_custom.py ...
（scenes_custom.sh 里已经把 --scenes-file 带上了。）

写法跟 scenes.py 一样，只是只留了这 5 个场景；make_scene / CLIENT_* / 各种缺省值
仍然 `from scenes import ...` 复用，不用再抄一遍。

两点跟 scenes.py 不一样的地方：

1. **判定统计量是 P95**，不是工具缺省的 P90 —— 跑的时候要带 `--latency-stat p95`。
   SLO 文本也按 P95 写死传进去了：`_ttft_text/_tpot_text` 只会生成 "p90" 的字样，
   不改它就得自己给文本，否则报表「测试用例」列会写成 ttft_p90 跟判定口径对不上。
2. **client 一律 bench_serving**：它按 input_len / max_out_len 现造 random 请求，
   不用为每个长度准备一份场景 jsonl；而且 TTFT/TPOT 的 P95 只有它打得出来
   （本目录的 bench_serving.py 已补过 P95，记得覆盖到跑压测那台机器上）。

没有友商数据：seed_batch / seed_rate 一律不填，并发和 rate 全靠工具自己上探
（batch 从 1 起倍增，rate 按输入长度折算基准值再二分）。
"""

from __future__ import annotations

# 复用 scenes.py 的构造器 / client 常量 / 校验（换成 --scenes-file 之后，这几个名字
# 也得在本模块里能 import 到 —— 工具是 from scenes import ... 拿的）
from scenes import (CLIENT_BENCH, CLIENTS, BENCH_DATASET, DATASET_DIR,  # noqa: F401
                    derive_model_name, make_scene, normalize_client, validate_scenes)

# 模型路径 / 短名由 scenes_custom.sh 的 --bench-model / --model-name 传进来；
# 这里留空，免得漏传时把 scenes.py 里的默认模型带去压测。
MODEL = ""
MODEL_NAME = ""


def _p95(ttft_s: float, tpot_ms: float) -> dict:
    """报表「测试用例」列里的 SLO 文本 —— 判定口径是 P95，文本就写 p95。"""
    return {
        "slo_ttft_text": "ttft_p95 不限制" if ttft_s < 0 else f"ttft_p95 < {ttft_s:g}s",
        "slo_tpot_text": "tpot_p95 不限制" if tpot_ms < 0 else f"tpot_p95 < {tpot_ms:g}ms",
    }

SCENES = [
    #make_scene(
    #    "isl1k_osl1k", "1K ISL / 1K OSL",
    #    input_len=1024, max_out_len=1024,
    #    slo_ttft=3.0, slo_tpot=0.070,
    #    seed_batch=64, seed_rate=14,
    #    client=CLIENT_BENCH, **_p95(3.0, 70),
    #),
    #make_scene(
    #    "isl4k_osl1k", "4K ISL / 1K OSL",
    #    input_len=4096, max_out_len=1024,
    #    slo_ttft=6.0, slo_tpot=0.070,
    #    seed_batch=64, seed_rate=6,
    #    client=CLIENT_BENCH, **_p95(6.0, 70),
    #),
    #make_scene(
    #    "isl8k_osl1k", "8K ISL / 1K OSL",
    #    input_len=8192, max_out_len=1024,
    #    slo_ttft=12.0, slo_tpot=0.070,
    #    client=CLIENT_BENCH, **_p95(12.0, 70),
    #),
    make_scene(
        "isl16k_osl1k", "16K ISL / 1K OSL",
        input_len=16384, max_out_len=1024,
        slo_ttft=15.0, slo_tpot=0.070,
        seed_batch=64, seed_rate=6,
        client=CLIENT_BENCH, **_p95(15.0, 70),
    ),
    make_scene(
        "isl32k_osl1k", "32K ISL / 1K OSL",
        input_len=32768, max_out_len=1024,
        slo_ttft=18.0, slo_tpot=0.080,
        seed_batch=64, seed_rate=3,
        client=CLIENT_BENCH, **_p95(18.0, 80),
    ),
    make_scene(
        "isl48k_osl1k", "48K ISL / 1K OSL",
        input_len=49152, max_out_len=1024,
        slo_ttft=18.0, slo_tpot=0.080,
        seed_batch=64, seed_rate=3,
        client=CLIENT_BENCH, **_p95(18.0, 80),
    ),
    make_scene(
        "isl60k_osl1k", "60K ISL / 1K OSL",
        input_len=61440, max_out_len=1024,
        slo_ttft=18.0, slo_tpot=0.080,
        seed_batch=64, seed_rate=3,
        client=CLIENT_BENCH, **_p95(18.0, 80),
    ),
    make_scene(
        "isl72k_osl1k", "72K ISL / 1K OSL",
        input_len=73728, max_out_len=1024,
        slo_ttft=24.0, slo_tpot=0.080,
        seed_batch=64, seed_rate=3,
        client=CLIENT_BENCH, **_p95(24.0, 80),
    ),
    make_scene(
        "isl84k_osl1k", "84K ISL / 1K OSL",
        input_len=85899, max_out_len=1024,
        slo_ttft=24.0, slo_tpot=0.080,
        seed_batch=64, seed_rate=1,
        client=CLIENT_BENCH, **_p95(24.0, 80),
    ),
    make_scene(
        "isl96k_osl1k", "96K ISL / 1K OSL",
        input_len=98304, max_out_len=1024,
        slo_ttft=24.0, slo_tpot=0.080,
        seed_batch=64, seed_rate=1,
        client=CLIENT_BENCH, **_p95(24.0, 80),
    ),
    make_scene(
        "isl128k_osl1k", "128K ISL / 1K OSL",
        input_len=131072, max_out_len=1024,
        slo_ttft=40.0, slo_tpot=0.080,
        seed_batch=64, seed_rate=1,
        client=CLIENT_BENCH, **_p95(40.0, 80),
    ),
    make_scene(
        "isl256k_osl1k", "256K ISL / 1K OSL",
        input_len=262144, max_out_len=1024,
        slo_ttft=60.0, slo_tpot=0.080,
        seed_batch=64, seed_rate=1,
        client=CLIENT_BENCH, **_p95(60.0, 80),
    ),
]

SCENE_BY_KEY = {s["key"]: s for s in SCENES}

"""

    make_scene(
        "isl384k_osl1k", "384K ISL / 1K OSL",
        input_len=393216, max_out_len=1024,
        slo_ttft=60.0, slo_tpot=0.080,
        seed_batch=64, seed_rate=0.2,
        client=CLIENT_BENCH, **_p95(60.0, 80),
    ),
    make_scene(
        "isl512k_osl1k", "512K ISL / 1K OSL",
        input_len=524288, max_out_len=1024,
        slo_ttft=90.0, slo_tpot=0.080,
        seed_batch=64, seed_rate=0.2,
        client=CLIENT_BENCH, **_p95(90.0, 80),
    ),
    make_scene(
        "isl768k_osl1k", "768K ISL / 1K OSL",
        input_len=786432, max_out_len=1024,
        slo_ttft=90.0, slo_tpot=0.080,
        client=CLIENT_BENCH, **_p95(90.0, 80),
    ),
    make_scene(
        "isl960k_osl1k", "960K ISL / 1K OSL",
        input_len=983040, max_out_len=1024,
        slo_ttft=90.0, slo_tpot=0.080,
        client=CLIENT_BENCH, **_p95(90.0, 80),
    ),

"""
# 显式传 SCENES —— validate_scenes() 不带参时校验的是 scenes.py 自己那份
validate_scenes(SCENES)
