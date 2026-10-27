#!/usr/bin/env python3
"""Regression tests for the ponly_cached generated-shared-prefix mode."""

from __future__ import annotations

import unittest

from runner import (
    MODE_FULL_CACHED,
    MODE_PONLY_CACHED,
    BenchClientCfg,
    build_bench_serving_cmd,
    cache_hit_ratio,
    cache_input_lens,
    client_for,
    is_cached_mode,
    mode_axes,
    mode_missing_slo,
    mode_out_len,
    new_cache_seed,
    saturates_run,
)


SCENE = {
    "key": "cached_16k",
    "name": "Cached 16K / 1K",
    "client": "ais_bench",
    "input_len": 16384,
    "max_out_len": 1024,
    "cached_input_len": 12288,
    "non_cached_input_len": 4096,
}


class PonlyCachedTest(unittest.TestCase):
    def test_mode_contract(self) -> None:
        self.assertEqual(mode_axes(MODE_PONLY_CACHED), (True, False))
        self.assertEqual(mode_axes(MODE_FULL_CACHED), (True, True))
        self.assertIsNone(mode_missing_slo(SCENE, MODE_FULL_CACHED))
        self.assertFalse(saturates_run(MODE_PONLY_CACHED))
        self.assertFalse(saturates_run(MODE_FULL_CACHED))
        self.assertTrue(is_cached_mode(MODE_FULL_CACHED))
        self.assertEqual(client_for(SCENE, MODE_PONLY_CACHED),
                         "sglang.bench_serving")
        self.assertEqual(client_for(SCENE, MODE_FULL_CACHED),
                         "sglang.bench_serving")
        self.assertEqual(mode_out_len(SCENE, MODE_PONLY_CACHED), 1024)
        self.assertEqual(mode_out_len(SCENE, MODE_FULL_CACHED), 1024)
        self.assertEqual(cache_input_lens(SCENE), (12288, 4096))
        self.assertAlmostEqual(cache_hit_ratio(SCENE), 0.75)
        alias_scene = {
            **SCENE,
            "non_cached_input_len": None,
            "none_cached_input_len": 4096,
        }
        self.assertEqual(cache_input_lens(alias_scene), (12288, 4096))

    def test_command_uses_gsp_and_unique_seed(self) -> None:
        cfg = BenchClientCfg(model="/model", python="python3", dataset="/ignored")
        cmd = build_bench_serving_cmd(
            SCENE,
            batch=64,
            rate=3.5,
            mode=MODE_PONLY_CACHED,
            host_ip="127.0.0.1",
            host_port="8001",
            cfg=cfg,
            num_prompts=320,
            seed=12345,
        )
        joined = " ".join(cmd)
        self.assertIn("--dataset-name generated-shared-prefix", joined)
        self.assertIn("--gsp-num-groups 1", joined)
        self.assertIn("--gsp-prompts-per-group 320", joined)
        self.assertIn("--gsp-system-prompt-len 12288", joined)
        self.assertIn("--gsp-question-len 4096", joined)
        self.assertIn("--gsp-output-len 1024", joined)
        self.assertIn("--gsp-range-ratio 1.0", joined)
        self.assertIn("--seed 12345", joined)
        self.assertNotIn("--dataset-path", joined)
        self.assertNotIn("--random-input-len", joined)

        full_cmd = build_bench_serving_cmd(
            SCENE,
            batch=64,
            rate=3.5,
            mode=MODE_FULL_CACHED,
            host_ip="127.0.0.1",
            host_port="8001",
            cfg=cfg,
            num_prompts=320,
            seed=23456,
        )
        full_joined = " ".join(full_cmd)
        self.assertIn("--dataset-name generated-shared-prefix", full_joined)
        self.assertIn("--gsp-system-prompt-len 12288", full_joined)
        self.assertIn("--gsp-question-len 4096", full_joined)
        self.assertIn("--gsp-output-len 1024", full_joined)
        self.assertIn("--seed 23456", full_joined)

        seeds = {new_cache_seed() for _ in range(8)}
        self.assertGreater(len(seeds), 1)


if __name__ == "__main__":
    unittest.main()
