#!/usr/bin/env python3
"""Regression tests for the SLO-first concurrency/rate search."""

from __future__ import annotations

import unittest

from runner import BenchResult, StatCfg
from search import (
    BACKSEARCH_BISECT, BACKSEARCH_SECANT, SATURATE_RATE, SLO_FRONTIER_MARGIN,
    _largest_slo_rate, _sweep_saturate, _tune_rate, rule_name,
)


class FakeSearchContext:
    """合成机器：限速点 TTFT = 100 + slope×rate（SLO 线 1000 ⇒ rate 边界 = 900/slope），
    吞吐 = min(rate, capacity)。

    打满点（rate=SATURATE_RATE）单独建模：TTFT = saturate_ttft，achieved =
    capacity × saturate_degrade。`saturate_degrade` 是复现 rate-cap 陷阱的必要旋钮 ——
    真实机器高并发打满时 achieved 会退化（state 里 16.7 而限速下能到 18.5），
    只有退化后 r_cap 才低于 rate 边界，_tune_rate 的初始上界才会压在边界下面。
    全部新参数默认值保持老行为，旧测试的轨迹不变。
    """

    # 老测试用的默认容量表；传 capacities={} 可换成 capacity_default 的均匀容量。
    DEFAULT_CAPACITIES = {64: 10.0, 96: 12.0, 144: 14.0, 216: 13.0, 256: 13.0}

    def __init__(self, *, ttft_slope: float = 20.0, saturate_ttft: float = 900.0,
                 saturate_degrade: float = 1.0,
                 capacities: dict[int, float] | None = None,
                 capacity_default: float = 13.0) -> None:
        self.stat = StatCfg("p90", "p90")
        self.max_rate = None
        self.cache: dict[tuple[int, float], tuple[BenchResult, dict]] = {}
        self.report_excluded: set[tuple[int, float]] = set()
        self.tried_batches: list[int] = []
        self.ttft_slope = ttft_slope
        self.saturate_ttft = saturate_ttft
        self.saturate_degrade = saturate_degrade
        self.capacities = dict(
            self.DEFAULT_CAPACITIES if capacities is None else capacities)
        self.capacity_default = capacity_default

    @staticmethod
    def quantize_batch(batch) -> int:
        return max(1, int(round(batch)))

    @staticmethod
    def reset_thru_base() -> None:
        pass

    def _capacity(self, batch: int) -> float:
        return self.capacities.get(batch, self.capacity_default)

    def test_at(self, batch: int, rate: float):
        batch = self.quantize_batch(batch)
        rate = round(float(rate), 4)
        key = (batch, rate)
        if key in self.cache:
            return self.cache[key]
        if batch not in self.tried_batches:
            self.tried_batches.append(batch)

        capacity = self._capacity(batch)
        if rate == SATURATE_RATE:
            achieved = capacity * self.saturate_degrade
            ttft = self.saturate_ttft
        else:
            achieved = min(rate, capacity)
            ttft = 100.0 + self.ttft_slope * rate
        tpot = 40.0
        res = BenchResult(
            scene_key="fake",
            batch=batch,
            rate=rate,
            num_prompts=max(100, batch * 5),
            ttft_ms={"p90": ttft},
            tpot_ms={"p90": tpot},
            req_throughput=achieved,
            out_tok_thr=achieved * 1000.0,
            total_tok_thr=achieved * 2000.0,
            failed_requests=0,
            success_requests=max(100, batch * 5),
            max_concurrency=capacity,
            parse_ok=True,
        )
        slo = {
            "satisfied": res.parse_ok and ttft < 1000.0 and tpot < 80.0,
            "ttft_margin": (1000.0 - ttft) / 1000.0,
            "tpot_margin": (80.0 - tpot) / 80.0,
            "reasons": [],
            "check_ttft": True,
            "check_tpot": True,
        }
        self.cache[key] = (res, slo)
        return res, slo

    def collect_passing(self):
        rows = []
        for (batch, rate), (res, slo) in self.cache.items():
            if (batch, rate) in self.report_excluded:
                continue
            if slo["satisfied"]:
                rows.append((res.out_tok_thr, batch, rate, res, slo))
        rows.sort(key=lambda row: row[0], reverse=True)
        return rows


class SearchStrategyTest(unittest.TestCase):
    def test_saturation_phase_uses_slo_not_throughput_gate(self) -> None:
        class SweepContext:
            stat = StatCfg("p90", "p90")
            peak_thru = 0.0

            @staticmethod
            def quantize_batch(batch) -> int:
                return max(1, int(round(batch)))

            @staticmethod
            def mark_reject(_res, _reason) -> None:
                pass

            def test_at(self, batch: int, rate: float):
                batch = self.quantize_batch(batch)
                res = BenchResult(
                    scene_key="fake",
                    batch=batch,
                    rate=rate,
                    num_prompts=batch * 5,
                    ttft_ms={"p90": 50.0 + batch * 5.0},
                    tpot_ms={"p90": 40.0},
                    req_throughput=1.0,
                    out_tok_thr=10000.0 if batch <= 96 else 9000.0,
                    total_tok_thr=20000.0,
                    failed_requests=0,
                    success_requests=batch * 5,
                    max_concurrency=batch,
                    parse_ok=True,
                )
                slo = {
                    "satisfied": batch <= 160,
                    "ttft_margin": (1000.0 - (50.0 + batch * 5.0)) / 1000.0,
                    "tpot_margin": 0.5,
                    "reasons": [] if batch <= 160 else ["TTFT 超限"],
                    "check_ttft": True,
                    "check_tpot": True,
                }
                return res, slo

        lo_b, hi_b, _best = _sweep_saturate(
            SweepContext(),
            {"slo_ttft": 1.0, "slo_tpot": 0.080},
            max_batch=256,
            batch_step_min=2,
            start_batch=64,
            backsearch="bisect",
        )
        self.assertEqual(lo_b, 160)
        self.assertEqual(hi_b, 162)

    def test_keeps_expanding_after_first_throughput_drop(self) -> None:
        ctx = FakeSearchContext()
        start, start_slo = ctx.test_at(64, SATURATE_RATE)

        best, best_slo, best_score = _tune_rate(
            ctx,
            start_res=start,
            start_slo=start_slo,
            start_score=start.out_tok_thr,
            rate_base=10.0,
            max_batch=256,
            batch_step_min=1,
            rate_step_min=0.01,
            rate_rel_tol=0.0,
            max_rounds=6,
            min_rate=0.05,
            batch_growth=1.5,
            plateau_patience=3,
        )

        self.assertTrue(best_slo["satisfied"])
        self.assertEqual(best.batch, 144)
        self.assertEqual(best_score, 14000.0)
        self.assertIn(216, ctx.tried_batches)
        self.assertIn(256, ctx.tried_batches)

    # ── SLO 余量驱动的停止/扩张（RULE_RATE 带 +slo_margin）─────────────────

    def test_resume_returns_accepted_predecessor_not_crossing_probe(self) -> None:
        """续跑：上界正好落在上次的跨线探针上 ⇒ 退回上一环接受点，不报跨线点。"""
        ctx = FakeSearchContext(ttft_slope=10.0, saturate_degrade=0.05,
                                capacities={}, capacity_default=200.0)
        ctx.test_at(64, 67.5)                   # 上次跨线留下的（余量 0.225）
        ctx.report_excluded.add((64, 67.5))

        res, slo = _largest_slo_rate(
            ctx, batch=64, initial_rate=45.0, min_rate=0.05, max_rate=67.5,
            min_step=0.05, rel_tol=0.05)

        self.assertEqual(res.rate, 45.0)        # = 67.5 / 1.5，上一环接受点
        self.assertNotEqual(res.rate, 67.5)
        self.assertGreater(slo["ttft_margin"], SLO_FRONTIER_MARGIN)

    def test_rate_cap_expands_while_margin_is_large(self) -> None:
        """上界点合格且余量大 ⇒ cap 继续 1.5x 上探；跨越安全带的那一探不报出。"""
        # 限速点边界 rate=90（TTFT=100+10×rate），容量 200 ⇒ 吞吐全程跟随 rate。
        # 打满退化到 0.05 ⇒ r_cap=10 ⇒ 第二段初始上界只有 20，远低于真实边界。
        ctx = FakeSearchContext(ttft_slope=10.0, saturate_degrade=0.05,
                                capacities={}, capacity_default=200.0)
        start, start_slo = ctx.test_at(64, SATURATE_RATE)   # achieved=10, margin=0.1

        best, best_slo, _score = _tune_rate(
            ctx, start_res=start, start_slo=start_slo,
            start_score=start.out_tok_thr, rate_base=10.0,
            max_batch=64, batch_step_min=1, rate_step_min=0.05,
            rate_rel_tol=0.05, max_rounds=8, min_rate=0.05,
            batch_growth=1.5, plateau_patience=3)

        # 20 → 30 → 45 余量都 > 0.30；67.5 余量 0.225 越线，只用于下结论。
        self.assertEqual(best.rate, 45.0)
        self.assertEqual(ctx.max_rate, 45.0)                # 只有被接受的点才抬上界
        self.assertIn((64, 67.5), ctx.cache)                # 越线探针测过……
        self.assertIn((64, 67.5), ctx.report_excluded)      # ……但不进合格池
        self.assertGreater(best_slo["ttft_margin"], SLO_FRONTIER_MARGIN)
        self.assertEqual(best.req_throughput, 45.0)         # cap 扩张真的换到了吞吐

    def test_plateau_does_not_stop_while_margin_is_large(self) -> None:
        """容量墙：吞吐恒定但 TTFT 余量 90% ⇒ 平台期不停止，继续放大并发。"""
        ctx = FakeSearchContext(saturate_ttft=100.0, capacities={},
                                capacity_default=10.0)
        start, start_slo = ctx.test_at(64, SATURATE_RATE)   # margin=0.9

        best, best_slo, _score = _tune_rate(
            ctx, start_res=start, start_slo=start_slo,
            start_score=start.out_tok_thr, rate_base=10.0,
            max_batch=256, batch_step_min=1, rate_step_min=0.01,
            rate_rel_tol=0.0, max_rounds=8, min_rate=0.05,
            batch_growth=1.5, plateau_patience=3)

        # 旧逻辑 no_gain 到 3（batch=144）就停了；新逻辑余量门放行到上界。
        self.assertIn(216, ctx.tried_batches)
        self.assertIn(256, ctx.tried_batches)
        self.assertEqual(best.batch, 64)
        self.assertGreater(best_slo["ttft_margin"], SLO_FRONTIER_MARGIN)

    def test_max_rounds_is_fuse_when_plateau_persists_with_margin(self) -> None:
        """余量一直大、平台期一直持续 ⇒ max_rounds 保险丝兜底，不无限烧探针。"""
        ctx = FakeSearchContext(saturate_ttft=100.0, capacities={},
                                capacity_default=10.0)
        start, start_slo = ctx.test_at(64, SATURATE_RATE)

        _tune_rate(
            ctx, start_res=start, start_slo=start_slo,
            start_score=start.out_tok_thr, rate_base=10.0,
            max_batch=256, batch_step_min=1, rate_step_min=0.01,
            rate_rel_tol=0.0, max_rounds=4, min_rate=0.05,
            batch_growth=1.5, plateau_patience=3)

        self.assertEqual(ctx.tried_batches[-1], 216)        # no_gain=4 触发保险丝
        self.assertNotIn(256, ctx.tried_batches)

    def test_plateau_stops_normally_when_near_sla(self) -> None:
        """余量已 <= 30%（逼近 SLA）⇒ 平台期按 patience 正常停，不被保险丝推迟。"""
        # 限速点边界 rate=30（100+30×30=1000）⇒ 最优 rate≈30 时余量≈0。
        ctx = FakeSearchContext(ttft_slope=30.0,
                                capacities={64: 10.0, 96: 12.0, 144: 14.0},
                                capacity_default=13.0)
        start, start_slo = ctx.test_at(64, SATURATE_RATE)

        best, best_slo, best_score = _tune_rate(
            ctx, start_res=start, start_slo=start_slo,
            start_score=start.out_tok_thr, rate_base=10.0,
            max_batch=600, batch_step_min=1, rate_step_min=0.01,
            rate_rel_tol=0.0, max_rounds=8, min_rate=0.05,
            batch_growth=1.5, plateau_patience=3)

        self.assertIn(486, ctx.tried_batches)               # no_gain 攒到 3 的那一档
        self.assertNotIn(600, ctx.tried_batches)            # 余量门让平台期在 486 结束
        self.assertEqual(best.batch, 144)
        self.assertEqual(best_score, 14000.0)
        self.assertLessEqual(best_slo["ttft_margin"], SLO_FRONTIER_MARGIN)

    def test_rule_name_invalidates_old_rate_search_states(self) -> None:
        """搜 rate 的规则名带 +slo_margin（旧 state 作废重搜）；打满跑法不变。"""
        self.assertIn("+slo_margin", rule_name(False))
        self.assertNotIn("slo_margin", rule_name(True))
        self.assertNotIn("slo_margin", rule_name(True, no_rate_search=True))
        self.assertNotEqual(rule_name(False, BACKSEARCH_SECANT),
                            rule_name(False, BACKSEARCH_BISECT))


if __name__ == "__main__":
    unittest.main()
