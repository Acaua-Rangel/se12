"""Adapter: CalibrationProbe for the roofline model (US-015, AC-026).

Compute-bound cost per weight is measured with the SAME raw/fused GEMV
kernels (gemv.py), but on a small weight matrix sized to stay resident in
L2 across repeated calls — so timing measures compute, not the memory
system (design.md: "measured by running the kernel on cache-resident
data"). Bandwidth reuses the peak-copy measurement (AC-013).

UNVERIFIED ON REAL HARDWARE — see se12_common.cuh's note; this file shares
that caveat and could not be exercised at all while writing it.
"""

from __future__ import annotations

import numpy
import torch

from lws.adapters.cuda.benchmark_timer import CudaBenchmarkTimer
from lws.adapters.cuda.gemv import RawBf16MatVecKernel, Se12MatVecKernel
from lws.application.ports.gpu.kernel_compiler import KernelCompiler
from lws.application.ports.gpu.tuning_cache import BatchSize
from lws.domain.benchmark.statistics import RunCounts, compute_latency_statistics
from lws.domain.device.decisions import CompileTarget
from lws.domain.device.properties import GigabytesPerSecond, SmCount
from lws.domain.performance.roofline import ComputeCostPerWeight, FormatCosts

CACHE_RESIDENT_ROWS = 16
CACHE_RESIDENT_COLS = 256  # exactly one SE12 tile: the smallest meaningful, cache-resident shape
CALIBRATION_RUN_COUNTS = RunCounts(warmup=25, timed=100)


class CudaCalibrationProbe:  # calisthenics: allow 8 — hot-path probe, constitution's carve-out for adapters
    def __init__(self, compiler: KernelCompiler, target: CompileTarget, device_index: int, sm_count: SmCount, timer: CudaBenchmarkTimer) -> None:
        self.compiler = compiler
        self.target = target
        self.device_index = device_index
        self.sm_count = sm_count
        self.timer = timer

    def measure_bandwidth(self) -> GigabytesPerSecond:
        timer = self.timer
        return timer.measure_peak_copy_bandwidth()

    def measure_compute_bound_costs(self, batch: BatchSize) -> FormatCosts:
        batch_value = batch.value
        weights = _small_weights()
        raw_kernel = RawBf16MatVecKernel(self.compiler, self.target, weights, self.sm_count)
        fused_kernel = _small_fused_kernel(self.compiler, self.target, weights, self.sm_count)
        activations = torch.randn(batch_value, CACHE_RESIDENT_COLS, dtype=torch.float32, device=f"cuda:{self.device_index}")

        raw_cost = self._cost_per_weight(lambda: raw_kernel.multiply(activations))
        fused_cost = self._cost_per_weight(lambda: fused_kernel.multiply(activations))
        return FormatCosts(raw=raw_cost, se12=fused_cost)

    def _cost_per_weight(self, operation) -> ComputeCostPerWeight:
        timer = self.timer
        samples = timer.time_operation(operation, CALIBRATION_RUN_COUNTS)
        statistics = compute_latency_statistics(samples)
        median = statistics.median
        total_weights = CACHE_RESIDENT_ROWS * CACHE_RESIDENT_COLS
        per_weight = median.value / total_weights
        return ComputeCostPerWeight(per_weight)


def _small_weights() -> torch.Tensor:
    array = numpy.random.default_rng(0).standard_normal((CACHE_RESIDENT_ROWS, CACHE_RESIDENT_COLS)).astype(numpy.float32)
    return torch.from_numpy(array).to(torch.bfloat16).to("cuda")


def _small_fused_kernel(compiler, target, weights: torch.Tensor, sm_count):
    from lws.domain.se12.codec import encode_tensor
    from lws.domain.se12.tile import MatrixShape
    from lws.domain.weights import Bf16Weights

    bit_pattern = weights.reshape(-1).view(torch.int16).cpu().numpy()
    matrix_shape = MatrixShape(rows=CACHE_RESIDENT_ROWS, cols=CACHE_RESIDENT_COLS)
    encoded = encode_tensor(Bf16Weights(bit_pattern=bit_pattern), matrix_shape)
    return Se12MatVecKernel(compiler, target, encoded, sm_count)
