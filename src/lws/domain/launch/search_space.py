"""Search space for the per-GPU launch tuner (US-013, AC-024): every
candidate launch configuration worth trying for a shape, and the rules for
picking the best one — independently per kernel, and shared between raw and
fused (the "order-matched" configuration AC-032 needs for bit-identical
outputs).
"""

from __future__ import annotations

import dataclasses

from lws.domain.launch.heuristic import WARP_SIZE, BlockCount, GemvShape, LaunchConfiguration, ThreadsPerBlock

SEARCH_CANDIDATES = (32, 64, 128, 256, 512)


def candidate_configurations(shape: GemvShape) -> tuple[LaunchConfiguration, ...]:
    total_warps_needed = _total_warps_needed(shape)
    return tuple(_configuration_for(candidate, total_warps_needed) for candidate in SEARCH_CANDIDATES)


def _total_warps_needed(shape: GemvShape) -> int:
    rows = shape.output_rows
    batch = shape.batch_size
    return rows * batch


def _configuration_for(threads: int, total_warps_needed: int) -> LaunchConfiguration:
    warps_per_block = threads // WARP_SIZE
    blocks = _ceil_div(total_warps_needed, warps_per_block)
    return LaunchConfiguration(threads_per_block=ThreadsPerBlock(threads), blocks=BlockCount(blocks))


def _ceil_div(numerator: int, denominator: int) -> int:
    negated = -numerator // denominator
    return -negated


@dataclasses.dataclass(frozen=True)
class TimedCandidate:
    configuration: LaunchConfiguration
    measurement: TimingAndAccuracy


@dataclasses.dataclass(frozen=True)
class TimingAndAccuracy:
    median_latency_microseconds: float
    passes_accuracy_check: bool


def best_configuration(timed_candidates: tuple[TimedCandidate, ...]) -> LaunchConfiguration | None:
    """The fastest configuration that passed the accuracy check (AC-024:
    "only configurations that pass the AC-010 accuracy check are saved")."""
    passing = tuple(candidate for candidate in timed_candidates if _passes(candidate))
    if not passing:
        return None
    fastest = min(passing, key=_latency)
    return fastest.configuration


def order_matched_configuration(raw_timed: tuple[TimedCandidate, ...], fused_timed: tuple[TimedCandidate, ...]) -> LaunchConfiguration | None:
    """The single configuration, shared by both kernels, with the best
    combined time while both still pass accuracy — using the SAME
    configuration for both kernels is what makes their outputs
    bit-identical (AC-032), since reduction order depends only on it."""
    raw_passing = tuple(candidate for candidate in raw_timed if _passes(candidate))
    fused_passing = tuple(candidate for candidate in fused_timed if _passes(candidate))
    raw_configurations = frozenset(candidate.configuration for candidate in raw_passing)
    fused_configurations = frozenset(candidate.configuration for candidate in fused_passing)
    shared = raw_configurations & fused_configurations
    if not shared:
        return None
    combined = tuple(_combined_latency(configuration, raw_passing, fused_passing) for configuration in shared)
    best = min(combined, key=_combined_value)
    return best.configuration


@dataclasses.dataclass(frozen=True)
class _CombinedLatency:
    configuration: LaunchConfiguration
    total_microseconds: float


def _combined_latency(configuration: LaunchConfiguration, raw_passing: tuple[TimedCandidate, ...], fused_passing: tuple[TimedCandidate, ...]) -> _CombinedLatency:
    raw_time = _latency_for(configuration, raw_passing)
    fused_time = _latency_for(configuration, fused_passing)
    return _CombinedLatency(configuration=configuration, total_microseconds=raw_time + fused_time)


def _latency_for(configuration: LaunchConfiguration, candidates: tuple[TimedCandidate, ...]) -> float:
    matching = next(candidate for candidate in candidates if candidate.configuration == configuration)
    return _latency(matching)


def _passes(candidate: TimedCandidate) -> bool:
    measurement = candidate.measurement
    return measurement.passes_accuracy_check


def _latency(candidate: TimedCandidate) -> float:
    measurement = candidate.measurement
    return measurement.median_latency_microseconds


def _combined_value(combined: _CombinedLatency) -> float:
    return combined.total_microseconds
