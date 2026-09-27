"""Tests for the microbenchmark, per-GPU verdict and cross-GPU summary
(US-006, US-007).

@spec:AC-012 @spec:AC-013 @spec:AC-014 @spec:AC-023
@principle:P-006 @principle:P-007 @principle:P-009 @principle:P-011
"""

from __future__ import annotations

import pytest

from lws.adapters.filesystem.report_store import FilesystemReportStore
from lws.application.gpu.run_microbenchmark import (
    KernelVariant,
    RejectedOutcome,
    RunnableVariant,
    TimedOutcome,
    benchmark_variants,
    decide_shape_verdict,
    faster_variant,
)
from lws.application.gpu.summarize_reports import summarize_reports
from lws.domain.benchmark.statistics import (
    LatencySamples,
    Microseconds,
    RunCounts,
    check_run_counts,
    compute_latency_statistics,
    effective_bandwidth,
)
from lws.domain.benchmark.summary import ReportMetadata, check_metadata, classify_reports
from lws.domain.benchmark.verdict import (
    ShapeVerdicts,
    Verdict,
    aggregate_verdict,
    baseline_is_bandwidth_bound,
    speed_ratio,
    verdict_for_shape,
)
from lws.domain.device.properties import Bytes, GigabytesPerSecond
from tests.fakes import FakeBenchmarkTimer, FakeReportStore


def test_latency_statistics_median_and_percentiles():
    """@spec:AC-012 @principle:P-006"""
    samples = LatencySamples(values=tuple([100.0] * 50 + [90.0] * 40 + [200.0] * 10))
    statistics = compute_latency_statistics(samples)
    assert statistics.median.value == pytest.approx(100.0)
    percentiles = statistics.percentiles
    assert percentiles.p10.value == pytest.approx(90.0)
    assert percentiles.p90.value >= 100.0


def test_effective_bandwidth_is_bytes_over_median_time():
    """@spec:AC-012"""
    samples = LatencySamples(values=(1000.0,) * 10)  # 1000 us = 1 ms
    statistics = compute_latency_statistics(samples)
    bandwidth = effective_bandwidth(Bytes(10**9), statistics)  # 1 GB in 1 ms -> 1000 GB/s
    assert bandwidth.value == pytest.approx(1000.0, rel=1e-6)


def test_run_counts_meet_the_minimum_required_by_principle_p006():
    """@principle:P-006 — warmup >= 25, timed runs >= 100."""
    assert check_run_counts(RunCounts(warmup=25, timed=100)).meets_minimum is True
    assert check_run_counts(RunCounts(warmup=24, timed=100)).meets_minimum is False
    assert check_run_counts(RunCounts(warmup=25, timed=99)).meets_minimum is False


def test_verdict_is_go_when_fused_is_at_least_5_percent_faster_and_baseline_is_bandwidth_bound():
    """@spec:AC-014"""
    ratio = speed_ratio(Microseconds(90.0), Microseconds(100.0))
    bandwidth_check = baseline_is_bandwidth_bound(GigabytesPerSecond(750.0), GigabytesPerSecond(900.0))
    result = verdict_for_shape(ratio, bandwidth_check)
    assert result.verdict is Verdict.GO


def test_verdict_is_no_go_when_fused_is_not_5_percent_faster():
    """@spec:AC-014"""
    ratio = speed_ratio(Microseconds(99.0), Microseconds(100.0))
    bandwidth_check = baseline_is_bandwidth_bound(GigabytesPerSecond(750.0), GigabytesPerSecond(900.0))
    result = verdict_for_shape(ratio, bandwidth_check)
    assert result.verdict is Verdict.NO_GO


def test_verdict_is_inconclusive_when_baseline_is_not_bandwidth_bound():
    """@spec:AC-013"""
    ratio = speed_ratio(Microseconds(90.0), Microseconds(100.0))
    bandwidth_check = baseline_is_bandwidth_bound(GigabytesPerSecond(500.0), GigabytesPerSecond(900.0))  # 55% of peak
    result = verdict_for_shape(ratio, bandwidth_check)
    assert result.verdict is Verdict.INCONCLUSIVE


def test_aggregate_verdict_needs_every_shape_to_be_go():
    """@spec:AC-014 — GO only when every shape is GO."""
    go = verdict_for_shape(speed_ratio(Microseconds(90.0), Microseconds(100.0)), baseline_is_bandwidth_bound(GigabytesPerSecond(750.0), GigabytesPerSecond(900.0)))
    no_go = verdict_for_shape(speed_ratio(Microseconds(99.0), Microseconds(100.0)), baseline_is_bandwidth_bound(GigabytesPerSecond(750.0), GigabytesPerSecond(900.0)))
    inconclusive = verdict_for_shape(speed_ratio(Microseconds(90.0), Microseconds(100.0)), baseline_is_bandwidth_bound(GigabytesPerSecond(500.0), GigabytesPerSecond(900.0)))

    assert aggregate_verdict(ShapeVerdicts(items=(go, go))) is Verdict.GO
    assert aggregate_verdict(ShapeVerdicts(items=(go, no_go))) is Verdict.NO_GO
    assert aggregate_verdict(ShapeVerdicts(items=(go, inconclusive))) is Verdict.INCONCLUSIVE


def test_a_variant_that_fails_accuracy_is_rejected_and_never_timed():
    """@principle:P-007 — a fast wrong kernel is never reported."""
    timer = FakeBenchmarkTimer(samples=LatencySamples(values=(1.0,)), peak_bandwidth=GigabytesPerSecond(900.0))
    variant = KernelVariant(name="broken", runnable=RunnableVariant(accuracy_passed=False, operation=lambda: (_ for _ in ()).throw(AssertionError("must not run"))))

    results = benchmark_variants((variant,), Bytes(1024), RunCounts(warmup=25, timed=100), timer)

    assert len(results) == 1
    result = results[0]
    assert isinstance(result.outcome, RejectedOutcome)
    assert timer.timed_operations == []  # never called


def test_a_variant_that_passes_accuracy_is_timed_and_gets_a_bandwidth():
    """@spec:AC-012"""
    timer = FakeBenchmarkTimer(samples=LatencySamples(values=(100.0,) * 100), peak_bandwidth=GigabytesPerSecond(900.0))
    calls = []
    variant = KernelVariant(name="ok", runnable=RunnableVariant(accuracy_passed=True, operation=lambda: calls.append(1)))

    results = benchmark_variants((variant,), Bytes(10**7), RunCounts(warmup=25, timed=100), timer)

    result = results[0]
    assert isinstance(result.outcome, TimedOutcome)
    assert len(timer.timed_operations) == 1
    assert calls == [1]  # the operation itself DID run, exactly once through the fake timer


def test_faster_variant_prefers_the_smaller_median_and_handles_rejection():
    """@principle:P-009 — the verdict uses the faster of the lossless baselines."""
    timer = FakeBenchmarkTimer(samples=LatencySamples(values=(50.0,) * 10), peak_bandwidth=GigabytesPerSecond(900.0))
    fast = KernelVariant(name="fast", runnable=RunnableVariant(accuracy_passed=True, operation=lambda: None))
    counts = RunCounts(warmup=1, timed=10)
    fast_result = benchmark_variants((fast,), Bytes(1024), counts, timer)[0]

    slow_timer = FakeBenchmarkTimer(samples=LatencySamples(values=(500.0,) * 10), peak_bandwidth=GigabytesPerSecond(900.0))
    slow = KernelVariant(name="slow", runnable=RunnableVariant(accuracy_passed=True, operation=lambda: None))
    slow_result = benchmark_variants((slow,), Bytes(1024), counts, slow_timer)[0]

    rejected = KernelVariant(name="rejected", runnable=RunnableVariant(accuracy_passed=False, operation=lambda: None))
    rejected_result = benchmark_variants((rejected,), Bytes(1024), counts, timer)[0]

    assert faster_variant(fast_result, slow_result) is fast_result
    assert faster_variant(rejected_result, fast_result) is fast_result
    assert faster_variant(fast_result, rejected_result) is fast_result


def test_decide_shape_verdict_is_none_when_either_variant_was_rejected():
    """@spec:AC-014 — a verdict needs both variants timed."""
    timer = FakeBenchmarkTimer(samples=LatencySamples(values=(50.0,) * 10), peak_bandwidth=GigabytesPerSecond(900.0))
    counts = RunCounts(warmup=1, timed=10)
    ok = KernelVariant(name="ok", runnable=RunnableVariant(accuracy_passed=True, operation=lambda: None))
    rejected = KernelVariant(name="rejected", runnable=RunnableVariant(accuracy_passed=False, operation=lambda: None))
    ok_result, rejected_result = benchmark_variants((ok, rejected), Bytes(1024), counts, timer)

    result = decide_shape_verdict(ok_result, rejected_result, GigabytesPerSecond(900.0))

    assert result is None


def test_check_metadata_flags_missing_principle_p006_fields():
    """@spec:AC-023 @principle:P-006"""
    complete = {name: 0 for name in __import__("lws.domain.benchmark.summary", fromlist=["REQUIRED_METADATA_FIELDS"]).REQUIRED_METADATA_FIELDS}
    check = check_metadata(ReportMetadata(fields=complete))
    assert check.is_valid is True

    incomplete = dict(complete)
    del incomplete["driver_version"]
    del incomplete["nvrtc_version"]
    check = check_metadata(ReportMetadata(fields=incomplete))
    assert check.is_valid is False
    assert set(check.missing.names) == {"driver_version", "nvrtc_version"}


def test_classify_reports_separates_valid_from_rejected():
    """@spec:AC-023"""
    from lws.domain.benchmark.summary import REQUIRED_METADATA_FIELDS

    complete = {name: 0 for name in REQUIRED_METADATA_FIELDS}
    incomplete = {"gpu_name": "Tesla T4"}
    classified = classify_reports((ReportMetadata(fields=complete), ReportMetadata(fields=incomplete)))
    assert len(classified.valid.items) == 1
    assert len(classified.rejected.items) == 1
    assert "gpu_name" not in classified.rejected.items[0].missing.names  # it WAS present
    assert "driver_version" in classified.rejected.items[0].missing.names


def test_summarize_reports_use_case_with_a_fake_store():
    """@spec:AC-023"""
    from lws.domain.benchmark.summary import REQUIRED_METADATA_FIELDS
    from lws.domain.device.properties import GpuSlug

    complete = {name: 0 for name in REQUIRED_METADATA_FIELDS}
    store = FakeReportStore(initial_micro_reports=[complete, {"gpu_name": "incomplete"}])

    classified = summarize_reports(store)

    assert len(classified.valid.items) == 1
    assert len(classified.rejected.items) == 1

    # write_verdict_report is keyed by GPU slug (principle P-011).
    store.write_verdict_report(GpuSlug(value="tesla-t4-sm75"), {"verdict": "GO"})
    assert store.verdict_reports[GpuSlug(value="tesla-t4-sm75")] == {"verdict": "GO"}


def test_filesystem_report_store_round_trips(tmp_path):
    """@spec:AC-012 @spec:AC-014"""
    from lws.domain.device.properties import GpuSlug

    store = FilesystemReportStore(tmp_path)
    gpu_slug = GpuSlug(value="tesla-t4-sm75")
    micro_payload = {"gpu_name": "Tesla T4", "results": []}
    verdict_payload = {"verdict": "GO"}

    store.write_micro_report(gpu_slug, micro_payload)
    store.write_verdict_report(gpu_slug, verdict_payload)

    reloaded = store.read_all_micro_reports()
    assert reloaded == (micro_payload,)
    micro_path = tmp_path / "micro-tesla-t4-sm75.json"
    verdict_path = tmp_path / "verdict-tesla-t4-sm75.json"
    assert micro_path.exists()
    assert verdict_path.exists()


@pytest.mark.gpu
def test_benchmark_report_records_full_p006_metadata_on_a_real_gpu():
    """@spec:AC-012 @principle:P-006 — needs real GPU hardware."""
    pytest.skip("requires real GPU hardware to execute — see fused-decode-gemv tests for the same caveat")


@pytest.mark.gpu
def test_raw_baseline_reaches_70_percent_of_peak_bandwidth_or_inconclusive():
    """@spec:AC-013 — needs real GPU hardware."""
    pytest.skip("requires real GPU hardware to execute — see fused-decode-gemv tests for the same caveat")
