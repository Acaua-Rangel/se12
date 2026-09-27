"""Use case: time a set of kernel variants for one shape+batch and decide
the GO/NO-GO/INCONCLUSIVE verdict (US-006, US-007, AC-012, AC-013, AC-014,
principle P-007: a variant that fails its accuracy check is reported
"rejected" and never timed).

The caller (entrypoint) is responsible for constructing each variant's
`operation` closure (already bound to its own weights and activations) and
running its OWN accuracy/exactness pre-check — this use case only decides
whether to time a variant, and how to turn timings into a verdict.
"""

from __future__ import annotations

import dataclasses
import typing

from lws.application.ports.gpu.benchmark_timer import BenchmarkTimer
from lws.domain.benchmark.statistics import LatencyStatistics, RunCounts, compute_latency_statistics, effective_bandwidth
from lws.domain.benchmark.verdict import ShapeVerdict, baseline_is_bandwidth_bound, speed_ratio, verdict_for_shape
from lws.domain.device.properties import Bytes, GigabytesPerSecond


@dataclasses.dataclass(frozen=True)
class RunnableVariant:
    accuracy_passed: bool
    operation: typing.Callable[[], None]


@dataclasses.dataclass(frozen=True)
class KernelVariant:
    name: str
    runnable: RunnableVariant


@dataclasses.dataclass(frozen=True)
class RejectedOutcome:
    reason: str


@dataclasses.dataclass(frozen=True)
class TimedOutcome:
    statistics: LatencyStatistics
    bandwidth: GigabytesPerSecond


VariantOutcome = RejectedOutcome | TimedOutcome


@dataclasses.dataclass(frozen=True)
class VariantResult:
    name: str
    outcome: VariantOutcome


def benchmark_variants(variants: tuple[KernelVariant, ...], bytes_per_run: Bytes, counts: RunCounts, timer: BenchmarkTimer) -> tuple[VariantResult, ...]:
    return tuple(_benchmark_one(variant, bytes_per_run, counts, timer) for variant in variants)


def _benchmark_one(variant: KernelVariant, bytes_per_run: Bytes, counts: RunCounts, timer: BenchmarkTimer) -> VariantResult:
    name = variant.name
    runnable = variant.runnable
    if not runnable.accuracy_passed:
        return VariantResult(name=name, outcome=RejectedOutcome(reason="failed accuracy/exactness check"))
    operation = runnable.operation
    samples = timer.time_operation(operation, counts)
    statistics = compute_latency_statistics(samples)
    bandwidth = effective_bandwidth(bytes_per_run, statistics)
    return VariantResult(name=name, outcome=TimedOutcome(statistics=statistics, bandwidth=bandwidth))


def faster_variant(first: VariantResult, second: VariantResult) -> VariantResult:
    """P-009: the verdict uses the faster of the lossless baselines."""
    first_median = _median_or_none(first)
    second_median = _median_or_none(second)
    if first_median is None:
        return second
    if second_median is None:
        return first
    if first_median.value <= second_median.value:
        return first
    return second


def _median_or_none(result: VariantResult):
    outcome = result.outcome
    if isinstance(outcome, RejectedOutcome):
        return None
    statistics = outcome.statistics
    return statistics.median


def decide_shape_verdict(fused_result: VariantResult, baseline_result: VariantResult, peak_bandwidth: GigabytesPerSecond) -> ShapeVerdict | None:
    """None when either variant was rejected — a verdict needs both timed."""
    fused_outcome = fused_result.outcome
    baseline_outcome = baseline_result.outcome
    either_rejected = isinstance(fused_outcome, RejectedOutcome) or isinstance(baseline_outcome, RejectedOutcome)
    if either_rejected:
        return None
    fused_statistics = fused_outcome.statistics
    baseline_statistics = baseline_outcome.statistics
    fused_median = fused_statistics.median
    baseline_median = baseline_statistics.median
    ratio = speed_ratio(fused_median, baseline_median)
    baseline_bandwidth = baseline_outcome.bandwidth
    bandwidth_check = baseline_is_bandwidth_bound(baseline_bandwidth, peak_bandwidth)
    return verdict_for_shape(ratio, bandwidth_check)
