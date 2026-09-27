"""The GO / NO-GO / INCONCLUSIVE verdict rule (US-007, AC-013, AC-014)."""

from __future__ import annotations

import dataclasses
import enum

from lws.domain.benchmark.statistics import Microseconds
from lws.domain.device.properties import GigabytesPerSecond

GO_RATIO_THRESHOLD = 0.95
PEAK_BANDWIDTH_FRACTION = 0.70


class Verdict(enum.Enum):
    GO = "GO"
    NO_GO = "NO-GO"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclasses.dataclass(frozen=True)
class SpeedRatio:
    """fused kernel's median latency divided by the faster baseline's (AC-014)."""

    value: float


def speed_ratio(fused_median: Microseconds, baseline_median: Microseconds) -> SpeedRatio:
    fused_value = fused_median.value
    baseline_value = baseline_median.value
    ratio = fused_value / baseline_value
    return SpeedRatio(ratio)


@dataclasses.dataclass(frozen=True)
class BandwidthBoundCheck:
    is_bandwidth_bound: bool


def baseline_is_bandwidth_bound(measured: GigabytesPerSecond, peak: GigabytesPerSecond) -> BandwidthBoundCheck:
    """AC-013: the baseline must reach at least 70% of measured peak copy
    bandwidth, or the comparison is not valid (verdict becomes INCONCLUSIVE)."""
    measured_value = measured.value
    peak_value = peak.value
    fraction = measured_value / peak_value
    is_bound = fraction >= PEAK_BANDWIDTH_FRACTION
    return BandwidthBoundCheck(is_bandwidth_bound=is_bound)


@dataclasses.dataclass(frozen=True)
class ShapeVerdict:
    ratio: SpeedRatio
    verdict: Verdict


def verdict_for_shape(ratio: SpeedRatio, bandwidth_check: BandwidthBoundCheck) -> ShapeVerdict:
    if not bandwidth_check.is_bandwidth_bound:
        return ShapeVerdict(ratio=ratio, verdict=Verdict.INCONCLUSIVE)
    value = ratio.value
    if value <= GO_RATIO_THRESHOLD:
        return ShapeVerdict(ratio=ratio, verdict=Verdict.GO)
    return ShapeVerdict(ratio=ratio, verdict=Verdict.NO_GO)


@dataclasses.dataclass(frozen=True)
class ShapeVerdicts:
    """First-class collection: one ShapeVerdict per shape checked at one batch size."""

    items: tuple[ShapeVerdict, ...]


def aggregate_verdict(shape_verdicts: ShapeVerdicts) -> Verdict:
    """AC-014: GO only when EVERY shape is GO; INCONCLUSIVE if any shape is
    INCONCLUSIVE (a bad baseline measurement anywhere invalidates the GPU's
    verdict, per AC-013); otherwise NO-GO."""
    items = shape_verdicts.items
    verdicts = tuple(item.verdict for item in items)
    if any(verdict is Verdict.INCONCLUSIVE for verdict in verdicts):
        return Verdict.INCONCLUSIVE
    if all(verdict is Verdict.GO for verdict in verdicts):
        return Verdict.GO
    return Verdict.NO_GO
