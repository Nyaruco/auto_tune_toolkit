#!/usr/bin/env python3
"""Example scene file for the ponly_cached generated-shared-prefix mode."""

from __future__ import annotations

from scenes import (
    CLIENT_BENCH,
    make_scene,
    validate_scenes,
)

MODEL = ""
MODEL_NAME = ""


def _slo(ttft_s: float, tpot_ms: float) -> dict:
    return {
        "slo_ttft_text": "ttft_p95 不限制" if ttft_s < 0 else f"ttft_p95 < {ttft_s:g}s",
        "slo_tpot_text": "tpot_p95 不限制" if tpot_ms < 0 else f"tpot_p95 < {tpot_ms:g}ms",
    }


SCENES = [
    make_scene(
        "cached_16k",
        "Cached 16K / 1K OSL",
        input_len=16384,
        max_out_len=1024,
        cached_input_len=12288,
        non_cached_input_len=4096,
        output_len=1024,
        slo_ttft=15.0,
        slo_tpot=-1,
        client=CLIENT_BENCH,
        **_slo(15.0, 0),
    ),
]

SCENE_BY_KEY = {s["key"]: s for s in SCENES}
validate_scenes(SCENES)
