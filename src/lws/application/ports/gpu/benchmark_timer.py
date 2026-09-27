"""Port: times a GPU operation with CUDA events after an L2 flush, and
measures peak copy bandwidth (US-006, AC-012, AC-013).
"""

from __future__ import annotations

import typing

from lws.domain.benchmark.statistics import LatencySamples, RunCounts
from lws.domain.device.properties import GigabytesPerSecond


class BenchmarkTimer(typing.Protocol):
    def time_operation(self, operation: typing.Callable[[], None], counts: RunCounts) -> LatencySamples:
        """Flushes L2 (a buffer at least twice the queried L2 size) before
        each timed run, times `operation` with CUDA events."""
        ...

    def measure_peak_copy_bandwidth(self) -> GigabytesPerSecond:
        """Device-to-device copy of min(1 GiB, 25% of free VRAM) — AC-013's reference peak."""
        ...
