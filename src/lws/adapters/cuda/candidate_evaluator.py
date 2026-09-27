"""Adapter: CandidateEvaluator for the launch tuner (US-013, AC-024).

Scope note: this checks that a candidate CONFIGURATION still computes the
CORRECT result (cross-checked against the kernel's own default/untuned
configuration, run once as a reference) and times it — it is not where the
fused-vs-raw AC-010 comparison (fused error ≤ 2x raw error against an FP64
reference) is decided; that is the microbenchmark/verdict's job (T-006).
Changing only launch geometry (threads per block, block count) for the SAME
kernel and the SAME weights cannot change the arithmetic — a mismatch here
means the configuration under-covers the problem (a real bug), not a
precision difference.

UNVERIFIED ON REAL HARDWARE — see se12_common.cuh's note; this file shares
that caveat and could not be exercised at all while writing it.
"""

from __future__ import annotations

import time

import torch

from lws.domain.launch.heuristic import LaunchConfiguration
from lws.domain.launch.search_space import TimingAndAccuracy

WARMUP_RUNS = 5
TIMED_RUNS = 20
MATCH_TOLERANCE = 1e-4


class KernelCandidateEvaluator:  # calisthenics: allow 8 — hot-path evaluator, constitution's carve-out for adapters
    """Times one kernel (already bound to its weights) against a fixed
    activation batch, on a chosen device, treating the kernel's own
    default-configuration output as the reference."""

    def __init__(self, kernel, activations: torch.Tensor) -> None:
        self.kernel = kernel
        self.activations = activations
        self.reference = kernel.multiply(activations)

    def evaluate(self, configuration: LaunchConfiguration) -> TimingAndAccuracy:
        kernel = self.kernel
        activations = self.activations
        for _ in range(WARMUP_RUNS):
            kernel.multiply(activations, configuration)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(TIMED_RUNS):
            output = kernel.multiply(activations, configuration)
        torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - start
        median_microseconds = (elapsed_seconds / TIMED_RUNS) * 1_000_000
        passes = _matches_reference(output, self.reference)
        return TimingAndAccuracy(median_latency_microseconds=median_microseconds, passes_accuracy_check=passes)


def _matches_reference(output: torch.Tensor, reference: torch.Tensor) -> bool:
    difference = torch.abs(output - reference)
    reference_scale = torch.abs(reference).clamp_min(1.0)
    relative_difference = difference / reference_scale
    maximum_relative_difference = relative_difference.max()
    return bool(maximum_relative_difference.item() <= MATCH_TOLERANCE)
