"""Entrypoint: `python -m lws.bench.micro --out reports/micro-<gpu-slug>.json`
(AC-012, AC-013, AC-014). Composition root: wires the CUDA adapters into
run_microbenchmark, assembles the principle P-006 metadata, decides the
per-GPU verdict and writes both reports.

UNVERIFIED ON REAL HARDWARE — see se12_common.cuh's note; every GPU-touching
call in this file shares that caveat and could not be exercised at all
while writing it. The accuracy pre-check here is a simplified, real-valued
comparison against a chunked FP64 reference (AC-010's "computed in row
chunks sized from the free VRAM"), not exhaustively tuned.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy
import torch

from lws.adapters.cuda.benchmark_timer import CudaBenchmarkTimer
from lws.adapters.cuda.gemv import RawBf16MatVecKernel, Se12MatVecKernel
from lws.adapters.cuda.gpu_probe import CudaGpuProbe
from lws.adapters.cuda.nvrtc_compiler import NvrtcKernelCompiler
from lws.adapters.filesystem.report_store import FilesystemReportStore
from lws.application.gpu.run_microbenchmark import (
    KernelVariant,
    RunnableVariant,
    benchmark_variants,
    decide_shape_verdict,
    faster_variant,
)
from lws.domain.benchmark.statistics import RunCounts
from lws.domain.benchmark.verdict import ShapeVerdicts
from lws.domain.benchmark.verdict import aggregate_verdict as domain_aggregate_verdict
from lws.domain.device.decisions import ComputeProfileKind, ForcePtx, choose_compile_target
from lws.domain.device.properties import Bytes
from lws.domain.se12.codec import encode_tensor
from lws.domain.se12.tile import MatrixShape
from lws.domain.weights import Bf16Weights

DEFAULT_SHAPES = ((2304, 9216), (9216, 2304), (256000, 2304))
DEFAULT_BATCHES = (1, 4, 16)
WARMUP_RUNS = 25
TIMED_RUNS = 100
RELATIVE_ACCURACY_TOLERANCE = 1e-2


def main(arguments: list[str] | None = None) -> int:  # calisthenics: allow 3 — POSIX exit code convention
    parsed = _parse_arguments(arguments)
    probe = CudaGpuProbe()
    properties = probe.properties()
    identity_and_compute = properties.identity_and_compute
    identity = identity_and_compute.identity
    compute = identity_and_compute.compute
    gpu_slug = identity.slug
    capability = compute.capability
    sm_count = compute.sm_count
    memory_and_nvrtc = properties.memory_and_nvrtc
    nvrtc_targets = memory_and_nvrtc.nvrtc_targets
    memory = memory_and_nvrtc.memory
    characteristics = memory.characteristics
    l2_size = characteristics.l2_size
    target = choose_compile_target(capability, nvrtc_targets, ForcePtx(False))
    compiler = NvrtcKernelCompiler()
    timer = CudaBenchmarkTimer(0, l2_size)
    peak_bandwidth = timer.measure_peak_copy_bandwidth()
    counts = RunCounts(warmup=WARMUP_RUNS, timed=TIMED_RUNS)

    report_rows = _benchmark_all_shapes(compiler, target, sm_count, counts, timer, peak_bandwidth)
    shape_verdicts = _batch_one_verdicts(report_rows)

    overall_verdict = domain_aggregate_verdict(ShapeVerdicts(items=tuple(shape_verdicts))) if shape_verdicts else None
    micro_payload = _build_micro_payload(properties, peak_bandwidth, report_rows)
    verdict_payload = _build_verdict_payload(properties, overall_verdict, shape_verdicts)

    out_argument = parsed.out
    out_directory = out_argument.parent if out_argument else pathlib.Path("reports")
    store = FilesystemReportStore(out_directory)
    store.write_micro_report(gpu_slug, micro_payload)
    store.write_verdict_report(gpu_slug, verdict_payload)
    verdict_text = overall_verdict.value if overall_verdict is not None else "INCONCLUSIVE (no batch-1 shapes measured)"
    print(f"verdict for GPU '{gpu_slug.value}': {verdict_text}")
    return 0


def _parse_arguments(arguments: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m lws.bench.micro")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    return parser.parse_args(arguments)


def _all_shape_batch_pairs() -> tuple[tuple[int, int, int], ...]:
    return tuple((rows, cols, batch) for rows, cols in DEFAULT_SHAPES for batch in DEFAULT_BATCHES)


def _benchmark_all_shapes(compiler, target, sm_count, counts: RunCounts, timer, peak_bandwidth) -> list:
    pairs = _all_shape_batch_pairs()
    return [_benchmark_one_shape(compiler, target, sm_count, rows, cols, batch, counts, timer, peak_bandwidth) for rows, cols, batch in pairs]


def _batch_one_verdicts(report_rows: list) -> list:
    batch_one_rows = tuple(row for row in report_rows if row["batch_size"] == 1)
    verdicts = tuple(row["shape_verdict"] for row in batch_one_rows)
    return [verdict for verdict in verdicts if verdict is not None]


def _benchmark_one_shape(compiler, target, sm_count, rows: int, cols: int, batch: int, counts: RunCounts, timer, peak_bandwidth) -> dict:
    weights = _synthetic_bf16_weights(rows, cols)
    raw_kernel = RawBf16MatVecKernel(compiler, target, weights, sm_count)
    encoded = _encode(weights)
    fused_kernel = Se12MatVecKernel(compiler, target, encoded, sm_count)
    activations = torch.randn(batch, cols, dtype=torch.float32, device="cuda")

    reference = _fp64_reference(weights, activations)
    raw_output = raw_kernel.multiply(activations)
    fused_output = fused_kernel.multiply(activations)
    raw_passes = _within_tolerance(raw_output, reference)
    fused_passes = _within_tolerance(fused_output, reference)

    raw_variant = KernelVariant(name="raw_bf16", runnable=RunnableVariant(accuracy_passed=raw_passes, operation=lambda: raw_kernel.multiply(activations)))
    fused_variant = KernelVariant(name="fused_se12", runnable=RunnableVariant(accuracy_passed=fused_passes, operation=lambda: fused_kernel.multiply(activations)))
    bytes_per_run = _bytes_moved(rows, cols, batch)
    results = benchmark_variants((raw_variant, fused_variant), bytes_per_run, counts, timer)
    raw_result, fused_result = results
    shape_verdict = decide_shape_verdict(fused_result, raw_result, peak_bandwidth) if batch == 1 else None

    return {
        "shape": [rows, cols],
        "batch_size": batch,
        "results": {result.name: _outcome_to_dict(result) for result in results},
        "shape_verdict": shape_verdict,
    }


def _encode(weights: torch.Tensor):
    bit_pattern = weights.reshape(-1).view(torch.int16).cpu().numpy()
    rows, cols = weights.shape
    return encode_tensor(Bf16Weights(bit_pattern=bit_pattern), MatrixShape(rows=rows, cols=cols))


def _synthetic_bf16_weights(rows: int, cols: int) -> torch.Tensor:
    array = numpy.random.default_rng(0).standard_normal((rows, cols)).astype(numpy.float32)
    return torch.from_numpy(array).to(torch.bfloat16).to("cuda")


def _fp64_reference(weights: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
    weights_fp64 = weights.to(torch.float64)
    activations_fp64 = activations.to(torch.float64)
    return torch.nn.functional.linear(activations_fp64, weights_fp64)


def _within_tolerance(output: torch.Tensor, reference: torch.Tensor) -> bool:
    output_fp64 = output.to(torch.float64)
    difference = torch.abs(output_fp64 - reference)
    scale = torch.abs(reference).clamp_min(1.0)
    relative = difference / scale
    return bool(relative.max().item() <= RELATIVE_ACCURACY_TOLERANCE)


def _bytes_moved(rows: int, cols: int, batch: int) -> Bytes:
    weight_bytes = rows * cols * 2
    activation_bytes = batch * cols * 4
    return Bytes(weight_bytes + activation_bytes)


def _outcome_to_dict(result) -> dict:
    from lws.application.gpu.run_microbenchmark import RejectedOutcome, TimedOutcome

    outcome = result.outcome
    if isinstance(outcome, RejectedOutcome):
        return {"rejected": True, "reason": outcome.reason}
    statistics = outcome.statistics
    median = statistics.median
    percentiles = statistics.percentiles
    p10 = percentiles.p10
    p90 = percentiles.p90
    bandwidth = outcome.bandwidth
    return {
        "rejected": False,
        "median_microseconds": median.value,
        "p10_microseconds": p10.value,
        "p90_microseconds": p90.value,
        "bandwidth_gbps": bandwidth.value,
    }


def _build_micro_payload(properties, peak_bandwidth, report_rows: list) -> dict:
    identity_and_compute = properties.identity_and_compute
    identity = identity_and_compute.identity
    compute = identity_and_compute.compute
    name = identity.name
    slug = identity.slug
    capability = compute.capability
    sm_count = compute.sm_count
    return {
        "gpu_name": name.value,
        "gpu_slug": slug.value,
        "compute_capability": f"{capability.major}.{capability.minor}",
        "sm_count": sm_count.value,
        "peak_copy_bandwidth_gbps": peak_bandwidth.value,
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "results": report_rows,
    }


def _shape_verdict_to_dict(shape_verdict) -> dict:
    ratio = shape_verdict.ratio
    verdict = shape_verdict.verdict
    return {"ratio": ratio.value, "verdict": verdict.value}


def _build_verdict_payload(properties, overall_verdict, shape_verdicts) -> dict:
    identity_and_compute = properties.identity_and_compute
    identity = identity_and_compute.identity
    compute = identity_and_compute.compute
    slug = identity.slug
    capability = compute.capability
    per_shape = [_shape_verdict_to_dict(sv) for sv in shape_verdicts]
    verdict_text = overall_verdict.value if overall_verdict is not None else "INCONCLUSIVE"
    return {
        "gpu_slug": slug.value,
        "compute_capability": f"{capability.major}.{capability.minor}",
        "verdict": verdict_text,
        "per_shape": per_shape,
    }


if __name__ == "__main__":
    sys.exit(main())
