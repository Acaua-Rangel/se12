"""Adapter: BenchmarkTimer via CUDA events (US-006, AC-012, AC-013).

UNVERIFIED ON REAL HARDWARE — see se12_common.cuh's note; this file shares
that caveat and could not be exercised at all while writing it.
"""

from __future__ import annotations

import torch

from lws.domain.benchmark.statistics import LatencySamples, RunCounts, compute_latency_statistics, effective_bandwidth
from lws.domain.device.properties import Bytes, GigabytesPerSecond

BYTES_PER_GIGABYTE = 1000**3
GIBIBYTE = 1024**3
PEAK_COPY_FRACTION_OF_FREE = 0.25
MICROSECONDS_PER_MILLISECOND = 1000


class CudaBenchmarkTimer:  # calisthenics: allow 8 — hot-path timer, constitution's carve-out for adapters
    """Flushes L2 with a buffer at least twice the queried L2 size before
    every timed run; times with `torch.cuda.Event` pairs."""

    def __init__(self, device_index: int, l2_size: Bytes) -> None:
        self.device_index = device_index
        self.l2_size = l2_size
        flush_bytes = max(2 * l2_size.value, 1)
        flush_elements = -(-flush_bytes // 4)  # ceil division; float32 elements
        self.flush_buffer = torch.empty(flush_elements, dtype=torch.float32, device=f"cuda:{device_index}")

    def time_operation(self, operation, counts: RunCounts) -> LatencySamples:
        warmup = counts.warmup
        timed = counts.timed
        for _ in range(warmup):
            operation()
        torch.cuda.synchronize(self.device_index)
        samples = tuple(self._timed_run(operation) for _ in range(timed))
        return LatencySamples(values=samples)

    def _timed_run(self, operation) -> float:
        self._flush_l2()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        operation()
        end_event.record()
        torch.cuda.synchronize(self.device_index)
        elapsed_milliseconds = start_event.elapsed_time(end_event)
        return elapsed_milliseconds * MICROSECONDS_PER_MILLISECOND

    def _flush_l2(self) -> None:
        buffer = self.flush_buffer
        buffer.fill_(1.0)

    def measure_peak_copy_bandwidth(self) -> GigabytesPerSecond:
        device_index = self.device_index
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device_index)
        target_bytes = min(GIBIBYTE, int(free_bytes * PEAK_COPY_FRACTION_OF_FREE))
        element_count = -(-target_bytes // 4)
        source = torch.empty(element_count, dtype=torch.float32, device=f"cuda:{device_index}")
        destination = torch.empty(element_count, dtype=torch.float32, device=f"cuda:{device_index}")
        copy_counts = RunCounts(warmup=10, timed=20)
        samples = self.time_operation(lambda: destination.copy_(source), copy_counts)
        statistics = compute_latency_statistics(samples)
        moved_bytes = Bytes(element_count * 4)
        return effective_bandwidth(moved_bytes, statistics)
