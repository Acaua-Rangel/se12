"""Latency statistics: median, p10, p90, effective bandwidth (US-006, AC-012)."""

from __future__ import annotations

import dataclasses

import numpy

from lws.domain.device.properties import Bytes, GigabytesPerSecond

MICROSECONDS_PER_SECOND = 1_000_000
BYTES_PER_GIGABYTE = 1000**3
MINIMUM_WARMUP_RUNS = 25
MINIMUM_TIMED_RUNS = 100


@dataclasses.dataclass(frozen=True)
class Microseconds:
    value: float

    def __post_init__(self) -> None:
        value = self.value
        if value < 0:
            raise ValueError("Microseconds cannot be negative")


@dataclasses.dataclass(frozen=True)
class LatencySamples:
    """First-class collection: one run's raw per-timing measurements, in microseconds."""

    values: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class PercentileBand:
    p10: Microseconds
    p90: Microseconds


@dataclasses.dataclass(frozen=True)
class LatencyStatistics:
    median: Microseconds
    percentiles: PercentileBand


def compute_latency_statistics(samples: LatencySamples) -> LatencyStatistics:
    values = samples.values
    array = numpy.array(values, dtype=numpy.float64)
    median_value = float(numpy.median(array))
    p10_value = float(numpy.percentile(array, 10))
    p90_value = float(numpy.percentile(array, 90))
    percentiles = PercentileBand(p10=Microseconds(p10_value), p90=Microseconds(p90_value))
    return LatencyStatistics(median=Microseconds(median_value), percentiles=percentiles)


def effective_bandwidth(bytes_moved: Bytes, statistics: LatencyStatistics) -> GigabytesPerSecond:
    median = statistics.median
    median_seconds = median.value / MICROSECONDS_PER_SECOND
    moved_value = bytes_moved.value
    gigabytes_moved = moved_value / BYTES_PER_GIGABYTE
    value = gigabytes_moved / median_seconds
    return GigabytesPerSecond(value)


@dataclasses.dataclass(frozen=True)
class RunCounts:
    warmup: int
    timed: int


@dataclasses.dataclass(frozen=True)
class RunCountVerdict:
    meets_minimum: bool


def check_run_counts(counts: RunCounts) -> RunCountVerdict:
    """AC-012: warmup >= 25, timed runs >= 100 (principle P-006)."""
    warmup = counts.warmup
    timed = counts.timed
    meets_minimum = warmup >= MINIMUM_WARMUP_RUNS and timed >= MINIMUM_TIMED_RUNS
    return RunCountVerdict(meets_minimum=meets_minimum)
