"""Tests for the roofline performance model and calibration (US-015, AC-026)."""

from __future__ import annotations

import pytest

from lws.application.gpu.calibrate_device import calibrate_device, gpu_calibration_for_batch
from lws.application.ports.gpu.tuning_cache import BatchSize
from lws.domain.benchmark.statistics import Microseconds
from lws.domain.device.properties import GigabytesPerSecond
from lws.domain.performance.roofline import (
    RAW_BYTES_PER_WEIGHT,
    SE12_BYTES_PER_WEIGHT,
    ComputeCostPerWeight,
    FormatCosts,
    GpuCalibration,
    OverlapEfficiency,
    WeightFormat,
    bytes_per_weight,
    compute_time,
    fit_overlap_efficiency,
    memory_time,
    predict_shape_ratio,
    predicted_latency,
)
from lws.domain.weights import ElementCount
from tests.fakes import FakeCalibrationProbe


def test_bytes_per_weight_matches_the_formats():
    """@spec:AC-026"""
    assert bytes_per_weight(WeightFormat.RAW_BF16).value == RAW_BYTES_PER_WEIGHT
    assert bytes_per_weight(WeightFormat.SE12).value == pytest.approx(SE12_BYTES_PER_WEIGHT)
    assert bytes_per_weight(WeightFormat.SE12).value < bytes_per_weight(WeightFormat.RAW_BF16).value


def test_memory_time_scales_with_weight_count_and_bandwidth():
    """@spec:AC-026"""
    weight_count = ElementCount(10**9)
    bandwidth = GigabytesPerSecond(1000.0)  # 1000 GB/s = 1 byte/ns = 1e3 bytes/us
    time = memory_time(weight_count, bytes_per_weight(WeightFormat.RAW_BF16), bandwidth)
    # 1e9 weights * 2 bytes = 2e9 bytes; at 1000 GB/s = 1e6 bytes/us -> 2000 us
    assert time.value == pytest.approx(2_000.0, rel=1e-6)


def test_compute_time_scales_with_weight_count_and_cost():
    """@spec:AC-026"""
    weight_count = ElementCount(1000)
    cost = ComputeCostPerWeight(0.01)
    time = compute_time(weight_count, cost)
    assert time.value == pytest.approx(10.0)


def test_fit_overlap_efficiency_recovers_perfect_overlap():
    """@spec:AC-026 — when measured latency equals the larger phase exactly, overlap is 1.0 (fully hidden)."""
    mem = Microseconds(100.0)
    compute = Microseconds(20.0)
    measured = Microseconds(100.0)  # exactly max(mem, compute): the smaller phase fully hidden
    eta = fit_overlap_efficiency(measured, mem, compute)
    assert eta.value == pytest.approx(1.0)


def test_fit_overlap_efficiency_recovers_zero_overlap():
    """@spec:AC-026 — measured latency equals the full sum: nothing hidden."""
    mem = Microseconds(100.0)
    compute = Microseconds(20.0)
    measured = Microseconds(120.0)
    eta = fit_overlap_efficiency(measured, mem, compute)
    assert eta.value == pytest.approx(0.0)


def test_fit_overlap_efficiency_clamps_to_valid_range():
    """@spec:AC-026 — noisy/inconsistent measurements never produce an eta outside [0, 1]."""
    mem = Microseconds(100.0)
    compute = Microseconds(20.0)
    too_fast = Microseconds(50.0)  # faster than physically possible given the inputs
    eta = fit_overlap_efficiency(too_fast, mem, compute)
    assert 0.0 <= eta.value <= 1.0


def test_predicted_latency_matches_additive_and_hidden_extremes():
    """@spec:AC-026"""
    mem = Microseconds(100.0)
    compute = Microseconds(20.0)
    fully_hidden = predicted_latency(mem, compute, OverlapEfficiency(1.0))
    assert fully_hidden.value == pytest.approx(100.0)
    fully_exposed = predicted_latency(mem, compute, OverlapEfficiency(0.0))
    assert fully_exposed.value == pytest.approx(120.0)


def test_predict_shape_ratio_recovers_theoretical_ceiling_when_compute_is_negligible():
    """@spec:AC-026 — with negligible decode cost, the predicted ratio should
    approach SE12's byte ratio (~0.752, weight-codec design.md), regardless
    of GPU (bandwidth cancels out when both formats are memory-bound)."""
    weight_count = ElementCount(2304 * 9216)
    bandwidth = GigabytesPerSecond(732.0)
    negligible_cost = ComputeCostPerWeight(1e-12)
    costs = FormatCosts(raw=negligible_cost, se12=negligible_cost)
    calibration = GpuCalibration(bandwidth=bandwidth, costs=costs)
    ratio = predict_shape_ratio(weight_count, calibration, OverlapEfficiency(0.0))
    expected_ceiling = SE12_BYTES_PER_WEIGHT / RAW_BYTES_PER_WEIGHT
    assert ratio.value == pytest.approx(expected_ceiling, rel=1e-3)


def test_predict_shape_ratio_crosses_1_as_se12_compute_cost_grows():
    """@spec:AC-027 — the sweep's core property: increasing SE12's per-weight
    decode cost must eventually push the ratio above 1.0 (NO-GO territory),
    tracing a monotonic curve."""
    weight_count = ElementCount(2304 * 9216)
    bandwidth = GigabytesPerSecond(320.0)  # T4-like: lower bandwidth, less slack per byte
    raw_cost = ComputeCostPerWeight(1e-7)
    ratios = []
    for k_cost in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        costs = FormatCosts(raw=raw_cost, se12=ComputeCostPerWeight(k_cost))
        calibration = GpuCalibration(bandwidth=bandwidth, costs=costs)
        ratio = predict_shape_ratio(weight_count, calibration, OverlapEfficiency(0.5))
        ratios.append(ratio.value)
    assert ratios == sorted(ratios)  # monotonically increasing
    assert ratios[0] < 1.0
    assert ratios[-1] > 1.0


def test_calibrate_device_use_case_with_a_fake_probe():
    """@spec:AC-026 — the full use case, no GPU needed."""
    bandwidth = GigabytesPerSecond(732.0)
    costs_by_batch = {
        1: FormatCosts(raw=ComputeCostPerWeight(1e-6), se12=ComputeCostPerWeight(2e-6)),
        4: FormatCosts(raw=ComputeCostPerWeight(1.2e-6), se12=ComputeCostPerWeight(2.4e-6)),
        16: FormatCosts(raw=ComputeCostPerWeight(1.5e-6), se12=ComputeCostPerWeight(3e-6)),
    }
    probe = FakeCalibrationProbe(bandwidth=bandwidth, costs_by_batch=costs_by_batch)

    calibration = calibrate_device(probe)

    assert calibration.bandwidth is bandwidth
    per_batch = calibration.per_batch
    assert len(per_batch.items) == 3

    gpu_calibration = gpu_calibration_for_batch(calibration, BatchSize(value=4))
    assert gpu_calibration is not None
    costs = gpu_calibration.costs
    assert costs.raw.value == pytest.approx(1.2e-6)

    missing = gpu_calibration_for_batch(calibration, BatchSize(value=999))
    assert missing is None


@pytest.mark.gpu
def test_calibration_report_records_full_p006_metadata_on_a_real_gpu():
    """@spec:AC-026 @principle:P-006 — needs real GPU hardware."""
    pytest.skip("requires real GPU hardware to execute — see fused-decode-gemv tests for the same caveat")
