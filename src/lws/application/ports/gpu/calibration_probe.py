"""Port: measures the roofline model's inputs on a real GPU (US-015,
AC-026) — achievable bandwidth, INT32/FP32 instruction throughput, and the
raw/fused kernels' compute-bound cost per weight (on cache-resident data).
"""

from __future__ import annotations

import typing

from lws.application.ports.gpu.tuning_cache import BatchSize
from lws.domain.device.properties import GigabytesPerSecond
from lws.domain.performance.roofline import FormatCosts


class CalibrationProbe(typing.Protocol):
    def measure_bandwidth(self) -> GigabytesPerSecond:
        ...

    def measure_compute_bound_costs(self, batch: BatchSize) -> FormatCosts:
        ...
