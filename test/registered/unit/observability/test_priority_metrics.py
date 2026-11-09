"""Regression for #19979: changing client priorities must not grow metric labels.

Use actual collectors and registries: helper-only assertions miss retained series
in both the scheduler and tokenizer metrics, including PD queue gauges.
"""

import dataclasses
import os
import random
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import prometheus_client as prom

from sglang.srt.observability.label_transform import PRIORITY_TIERS
from sglang.srt.observability.metrics_collector import (
    QueueCount,
    SchedulerMetricsCollector,
    SchedulerStats,
    TokenizerMetricsCollector,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_QUEUE_FIELDS = (
    "num_running_reqs",
    "num_queue_reqs",
    "num_prefill_bootstrap_queue_reqs",
    "num_prefill_inflight_queue_reqs",
    "num_decode_prealloc_queue_reqs",
    "num_decode_transfer_queue_reqs",
    "num_decode_host_receive_queue_reqs",
)


def _reqs(priorities):
    return [SimpleNamespace(priority=p) for p in priorities]


def _collectors(registry, *, enabled=True, low_first=False):
    labels = {"model_name": "test", "moe_ep_rank": 1}
    if enabled:
        labels["priority"] = ""
    with ExitStack() as stack:
        for name in ("Gauge", "Counter", "Histogram"):
            stack.enter_context(
                patch.object(
                    prom, name, partial(getattr(prom, name), registry=registry)
                )
            )
        stack.enter_context(
            get_context().override_server_args(
                enable_priority_scheduling=enabled,
                schedule_low_priority_values_first=low_first,
            )
        )
        return (
            SchedulerMetricsCollector(labels=labels, server_args=None),
            TokenizerMetricsCollector(labels=labels),
        )


def _samples(registry, name):
    return [
        sample
        for family in registry.collect()
        for sample in family.samples
        if sample.name == name
    ]


def _observe_request(tokenizer, priority):
    labels = {
        **tokenizer.labels,
        "priority": str(priority) if priority is not None else "",
    }
    tokenizer.observe_one_finished_request(labels, 1, 2, 1, 0.01, True)
    tokenizer.observe_time_to_first_token(labels, 0.01, stream=False)
    tokenizer.observe_inter_token_latency(labels, 0.01, 1)
    tokenizer.observe_one_aborted_request(labels)


def _series_keys(registry):
    return {
        (sample.name, tuple(sorted(sample.labels.items())))
        for family in registry.collect()
        for sample in family.samples
    }


class TestPriorityMetrics(CustomTestCase):
    def test_boundaries_defaults_and_wire_roundtrip(self):
        priorities = [-sys.maxsize - 1, -1, 0, 9, 10, 29, 30, 49, 50, sys.maxsize, None]
        reqs = _reqs(priorities)
        count = QueueCount.from_reqs(reqs, True)
        self.assertEqual(count.tier_counts(), dict(zip(PRIORITY_TIERS, (4, 2, 2, 3))))
        self.assertIsNone(count.by_priority)
        self.assertEqual(count.worst_priority(False), -sys.maxsize - 1)
        self.assertEqual(count.worst_priority(True), sys.maxsize)
        self.assertEqual([r.priority for r in reqs], priorities)
        stats = SchedulerStats(num_running_reqs=count, fwd_occupancy=0.0)
        restored = msgspec.convert(
            msgspec.json.decode(msgspec.json.encode(stats)), type=SchedulerStats
        )
        self.assertEqual(dataclasses.asdict(restored), dataclasses.asdict(stats))
        self.assertEqual(QueueCount().tier_counts(), dict.fromkeys(PRIORITY_TIERS, 0))
        self.assertIsNone(QueueCount.from_reqs(_reqs(priorities)).by_tier)

    def test_missing_priorities_use_batch_tier_without_inventing_a_raw_value(self):
        registry = prom.CollectorRegistry()
        collector, tokenizer = _collectors(registry)
        collector.log_stats(
            SchedulerStats(num_running_reqs=QueueCount.from_reqs(_reqs([None]), True))
        )
        self.assertEqual(
            _samples(registry, "sglang:lowest_running_priority_value")[0].value, 0
        )
        self.assertEqual(
            {
                s.labels["tier"]: s.value
                for s in _samples(registry, "sglang:num_requests_running_by_tier")
            },
            dict(zip(PRIORITY_TIERS, (0, 0, 0, 1))),
        )
        _observe_request(tokenizer, None)
        self.assertEqual(
            {
                s.labels["priority"]
                for s in _samples(registry, "sglang:num_requests_total")
            },
            {"P3_Batch"},
        )

    def test_multiprocess_snapshots_are_latest_not_summed_and_counters_sum(self):
        from prometheus_client.multiprocess import MultiProcessCollector

        worker = """
import runpy
import sys
import prometheus_client as prom
helpers = runpy.run_path(sys.argv[1])
collector, _ = helpers['_collectors'](prom.CollectorRegistry())
count = helpers['QueueCount'].from_reqs(helpers['_reqs']([30] * int(sys.argv[2])), True)
collector.log_stats(helpers['SchedulerStats'](num_running_reqs=count))
collector.increment_priority_preemption(50, 0)
"""
        with tempfile.TemporaryDirectory() as directory:
            env = {**os.environ, "PROMETHEUS_MULTIPROC_DIR": directory}
            for size in (1, 2):
                subprocess.run(
                    [sys.executable, "-c", worker, __file__, str(size)],
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            registry = prom.CollectorRegistry()
            MultiProcessCollector(registry, path=directory)
            self.assertEqual(
                {
                    s.labels["tier"]: s.value
                    for s in _samples(registry, "sglang:num_requests_running_by_tier")
                },
                dict(zip(PRIORITY_TIERS, (0, 0, 2, 0))),
            )
            self.assertEqual(
                [
                    s.value
                    for s in _samples(registry, "sglang:lowest_running_priority_value")
                ],
                [30],
            )
            self.assertEqual(
                [s.value for s in _samples(registry, "sglang:preemptions_total")], [2]
            )

    def test_direction_and_empty_queues_reset_every_series(self):
        for low_first, expected in ((False, -20), (True, 80)):
            with self.subTest(low_first=low_first):
                registry = prom.CollectorRegistry()
                collector, _ = _collectors(registry, low_first=low_first)
                stats = SchedulerStats()
                for field in _QUEUE_FIELDS:
                    setattr(
                        stats,
                        field,
                        QueueCount.from_reqs(_reqs([-20, 10, 30, 80]), True),
                    )
                collector.log_stats(stats)
                self.assertEqual(
                    _samples(registry, "sglang:lowest_running_priority_value")[0].value,
                    expected,
                )
                # Same queue size, different tiers: an old tier must still clear.
                for field in _QUEUE_FIELDS:
                    setattr(stats, field, QueueCount.from_reqs(_reqs([30] * 4), True))
                collector.log_stats(stats)
                for name in (
                    "sglang:num_requests_running_by_tier",
                    "sglang:num_requests_waiting_by_tier",
                ):
                    self.assertEqual(
                        {s.labels["tier"]: s.value for s in _samples(registry, name)},
                        dict(zip(PRIORITY_TIERS, (0, 0, 4, 0))),
                    )
                collector.log_stats(SchedulerStats())
                for field in _QUEUE_FIELDS:
                    self.assertTrue(
                        all(s.value == 0 for s in _samples(registry, "sglang:" + field))
                    )
                self.assertEqual(
                    _samples(registry, "sglang:lowest_running_priority_value")[0].value,
                    0,
                )
                self.assertTrue(
                    all(
                        s.value == 0
                        for s in _samples(
                            registry, "sglang:num_requests_running_by_tier"
                        )
                    )
                )
                self.assertTrue(
                    all(
                        s.value == 0
                        for s in _samples(
                            registry, "sglang:num_requests_waiting_by_tier"
                        )
                    )
                )

    def test_legacy_producers_are_bucketed_and_fcfs_totals_unchanged(self):
        registry = prom.CollectorRegistry()
        collector, _ = _collectors(registry)
        legacy = QueueCount(4, {-999: 1, 20: 1, 60: 1, None: 1})
        collector.log_stats(SchedulerStats(num_running_reqs=legacy))
        samples = _samples(registry, "sglang:num_running_reqs")
        self.assertEqual(
            {s.labels["priority"]: s.value for s in samples},
            {"": 4, "P0_Critical": 1, "P1_High": 1, "P2_Normal": 0, "P3_Batch": 2},
        )
        registry = prom.CollectorRegistry()
        collector, _ = _collectors(registry, enabled=False)
        collector.log_stats(
            SchedulerStats(num_running_reqs=QueueCount.from_reqs(_reqs([1, 2])))
        )
        self.assertEqual(
            [s.value for s in _samples(registry, "sglang:num_running_reqs")], [2]
        )
        self.assertFalse(_samples(registry, "sglang:num_requests_running_by_tier"))

    def test_ten_thousand_distinct_priorities_bound_actual_registry(self):
        priorities = random.Random(19979).sample(range(-(10**9), 10**9), 10000)
        self.assertEqual(len(set(priorities)), 10000)
        registry = prom.CollectorRegistry()
        collector, tokenizer = _collectors(registry)
        # Materialize the entire legal pair space before measuring steady state.
        for victim in (0, 10, 30, 50):
            for trigger in (0, 10, 30, 50):
                collector.increment_priority_preemption(victim, trigger)
            _observe_request(tokenizer, victim)
        collector.log_stats(SchedulerStats())
        collector.increment_retracted_reqs(0, 0, 0)
        before = _series_keys(registry)
        stats = SchedulerStats()
        for start in range(0, len(priorities), 100):
            count = QueueCount.from_reqs(_reqs(priorities[start : start + 100]), True)
            self.assertLessEqual(len(count.by_tier), 4)
            self.assertIsNone(count.by_priority)
            for field in _QUEUE_FIELDS:
                setattr(stats, field, count)
            collector.log_stats(stats)
            for priority in priorities[start : start + 100]:
                collector.increment_priority_preemption(priority, -priority)
                _observe_request(tokenizer, priority)
            # Force repeated real exposition, not just helper evaluations.
            prom.generate_latest(registry)
        collector.increment_priority_preemption(50, None)
        collector.increment_priority_preemption(None, 0)
        collector.increment_retracted_reqs(2, 3, 4)
        pairs = _samples(registry, "sglang:preemptions_total")
        self.assertEqual(len(pairs), 16)
        self.assertEqual(sum(s.value for s in pairs), 10016)
        for _ in range(3):
            collector.log_stats(stats)
        self.assertEqual(
            sum(s.value for s in _samples(registry, "sglang:preemptions_total")), 10016
        )
        self.assertEqual(
            _samples(registry, "sglang:num_retracted_requests_total")[0].value, 2
        )
        for family in registry.collect():
            for sample in family.samples:
                for label in ("tier", "victim_tier", "trigger_tier"):
                    if label in sample.labels:
                        self.assertIn(sample.labels[label], PRIORITY_TIERS)
                if "priority" in sample.labels:
                    self.assertIn(sample.labels["priority"], ("", *PRIORITY_TIERS))
        for field in _QUEUE_FIELDS:
            self.assertEqual(len(_samples(registry, "sglang:" + field)), 5)
        self.assertEqual(_series_keys(registry), before)


if __name__ == "__main__":
    unittest.main()
