"""The roofline model itself (US-015, AC-026, AC-027):

    t_mem(format)     = bytes_per_weight(format) . W / BW_stream
    t_compute(format) = W . c(format, B)
    t(format)         ~ max(t_mem, t_compute) + (1 - eta) . min(t_mem, t_compute)
    ratio             = t(SE12) / t(baseline)

`eta` (overlap efficiency) is fitted ONCE per GPU from a REAL measured raw
latency plus the model's own t_mem/t_compute for the raw kernel — never
from the fused kernel, so predicting the fused kernel's ratio is always an
out-of-sample test of the model (design.md).
"""

from __future__ import annotations

import dataclasses
import enum

from lws.domain.benchmark.statistics import Microseconds
from lws.domain.device.properties import GigabytesPerSecond
from lws.domain.weights import ElementCount

SE12_BYTES_PER_WEIGHT = 12.03 / 8  # weight-codec design.md: ~12.03 bits/weight
RAW_BYTES_PER_WEIGHT = 2.0
BYTES_PER_MICROSECOND_PER_GIGABYTE_PER_SECOND = 1000  # 1 GB/s = 1e9 B/s = 1e3 B/us


class WeightFormat(enum.Enum):
    RAW_BF16 = "raw_bf16"
    SE12 = "se12"


@dataclasses.dataclass(frozen=True)
class BytesPerWeight:
    value: float


def bytes_per_weight(weight_format: WeightFormat) -> BytesPerWeight:
    if weight_format is WeightFormat.SE12:
        return BytesPerWeight(SE12_BYTES_PER_WEIGHT)
    return BytesPerWeight(RAW_BYTES_PER_WEIGHT)


@dataclasses.dataclass(frozen=True)
class ComputeCostPerWeight:
    """Microseconds of compute-bound time per weight (measured on
    cache-resident data — memory is not the limit, AC-026)."""

    value: float


def memory_time(weight_count: ElementCount, format_bytes: BytesPerWeight, bandwidth: GigabytesPerSecond) -> Microseconds:
    count = weight_count.value
    bytes_value = format_bytes.value
    total_bytes = count * bytes_value
    bandwidth_value = bandwidth.value
    bytes_per_microsecond = bandwidth_value * BYTES_PER_MICROSECOND_PER_GIGABYTE_PER_SECOND
    microseconds = total_bytes / bytes_per_microsecond
    return Microseconds(microseconds)


def compute_time(weight_count: ElementCount, cost: ComputeCostPerWeight) -> Microseconds:
    count = weight_count.value
    cost_value = cost.value
    microseconds = count * cost_value
    return Microseconds(microseconds)


@dataclasses.dataclass(frozen=True)
class OverlapEfficiency:
    """eta in [0, 1]: how much of the shorter phase hides under the longer one."""

    value: float


def fit_overlap_efficiency(measured: Microseconds, mem: Microseconds, compute: Microseconds) -> OverlapEfficiency:
    """Solves `measured = max(mem,compute) + (1-eta)*min(mem,compute)` for eta,
    from a REAL measured raw-kernel latency (never the fused kernel's)."""
    measured_value = measured.value
    mem_value = mem.value
    compute_value = compute.value
    larger = max(mem_value, compute_value)
    smaller = min(mem_value, compute_value)
    if smaller <= 0:
        return OverlapEfficiency(1.0)
    hidden_fraction = (measured_value - larger) / smaller
    eta = 1 - hidden_fraction
    clamped = min(max(eta, 0.0), 1.0)
    return OverlapEfficiency(clamped)


def predicted_latency(mem: Microseconds, compute: Microseconds, overlap: OverlapEfficiency) -> Microseconds:
    mem_value = mem.value
    compute_value = compute.value
    larger = max(mem_value, compute_value)
    smaller = min(mem_value, compute_value)
    eta = overlap.value
    hidden_fraction = 1 - eta
    result = larger + hidden_fraction * smaller
    return Microseconds(result)


@dataclasses.dataclass(frozen=True)
class PredictedRatio:
    value: float


def predict_ratio(fused_latency: Microseconds, baseline_latency: Microseconds) -> PredictedRatio:
    fused_value = fused_latency.value
    baseline_value = baseline_latency.value
    ratio = fused_value / baseline_value
    return PredictedRatio(ratio)


@dataclasses.dataclass(frozen=True)
class GpuCalibration:
    """Everything the model needs for ONE GPU: measured bandwidth and the
    raw/fused kernels' compute-bound cost per weight (2 fields; batch size
    is folded into which BatchCalibration the caller picked, not stored here)."""

    bandwidth: GigabytesPerSecond
    costs: FormatCosts


@dataclasses.dataclass(frozen=True)
class FormatCosts:
    raw: ComputeCostPerWeight
    se12: ComputeCostPerWeight


def predict_shape_ratio(weight_count: ElementCount, calibration: GpuCalibration, overlap: OverlapEfficiency) -> PredictedRatio:
    """The end-to-end prediction (AC-026's whole point): from a shape and one
    GPU's calibration, the expected fused ÷ raw-baseline latency ratio."""
    bandwidth = calibration.bandwidth
    costs = calibration.costs
    raw_bytes = bytes_per_weight(WeightFormat.RAW_BF16)
    se12_bytes = bytes_per_weight(WeightFormat.SE12)
    raw_mem = memory_time(weight_count, raw_bytes, bandwidth)
    se12_mem = memory_time(weight_count, se12_bytes, bandwidth)
    raw_cost = costs.raw
    se12_cost = costs.se12
    raw_compute = compute_time(weight_count, raw_cost)
    se12_compute = compute_time(weight_count, se12_cost)
    raw_latency = predicted_latency(raw_mem, raw_compute, overlap)
    se12_latency = predicted_latency(se12_mem, se12_compute, overlap)
    return predict_ratio(se12_latency, raw_latency)
