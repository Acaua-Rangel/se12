"""Entrypoint: `python -m lws.tune <N,K>...` (US-013, AC-024).

Composition root: for each `N,K` shape given (default: Gemma 2 2B's Linear
shapes) and batch sizes 1, 4, 16, builds synthetic weights of that shape
(launch tuning only cares about SHAPE, not content — the numbers are random
but the round trip, sizes and grid geometry are exactly what a real
tensor's would be), compiles both kernels, and tunes.
"""

from __future__ import annotations

import argparse
import sys

import numpy
import torch

from lws.adapters.cuda.candidate_evaluator import KernelCandidateEvaluator
from lws.adapters.cuda.gemv import RawBf16MatVecKernel, Se12MatVecKernel
from lws.adapters.cuda.gpu_probe import CudaGpuProbe
from lws.adapters.cuda.nvrtc_compiler import NvrtcKernelCompiler
from lws.adapters.filesystem.tuning_cache import FilesystemTuningCache
from lws.application.gpu.tune_launch import EvaluatorPair, TuneRequest, tune_launch
from lws.application.ports.gpu.tuning_cache import BatchSize, WeightShape
from lws.domain.device.decisions import ForcePtx, choose_compile_target
from lws.domain.se12.codec import encode_tensor
from lws.domain.se12.tile import MatrixShape

DEFAULT_SHAPES = ((2304, 9216), (9216, 2304), (256000, 2304))
DEFAULT_BATCHES = (1, 4, 16)
CACHE_DIRECTORY_ENV_VAR = "LWS_CACHE_DIR"


def main(arguments: list[str] | None = None) -> int:  # calisthenics: allow 3 — POSIX exit code convention
    parsed = _parse_arguments(arguments)
    shapes = parsed.shapes or DEFAULT_SHAPES
    probe = CudaGpuProbe()
    properties = probe.properties()
    identity_and_compute = properties.identity_and_compute
    identity = identity_and_compute.identity
    compute = identity_and_compute.compute
    gpu_slug = identity.slug
    sm_count = compute.sm_count
    capability = compute.capability
    memory_and_nvrtc = properties.memory_and_nvrtc
    nvrtc_targets = memory_and_nvrtc.nvrtc_targets
    target = choose_compile_target(capability, nvrtc_targets, ForcePtx(False))

    compiler = NvrtcKernelCompiler()
    cache_directory = compiler.cache_directory
    cache = FilesystemTuningCache(cache_directory, gpu_slug)

    requests = tuple(_request_for(rows, cols, batch) for rows, cols in shapes for batch in DEFAULT_BATCHES)
    evaluators = _evaluator_factory(compiler, target, sm_count)
    tune_launch(gpu_slug, requests, evaluators, cache)
    print(f"tuning written for GPU '{gpu_slug.value}' ({len(requests)} (shape, batch) combinations)")
    return 0


def _parse_arguments(arguments: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m lws.tune")
    parser.add_argument("shapes", nargs="*", type=_parse_shape, help="N,K pairs; default: reference Gemma 2 2B shapes")
    return parser.parse_args(arguments)


def _parse_shape(text: str) -> tuple[int, int]:
    rows_text, cols_text = text.split(",")
    return int(rows_text), int(cols_text)


def _request_for(rows: int, cols: int, batch: int) -> TuneRequest:
    weight_shape = WeightShape(output_rows=rows, k=cols)
    return TuneRequest(weight_shape=weight_shape, batch=BatchSize(value=batch))


def _evaluator_factory(compiler, target, sm_count):
    def build(request: TuneRequest) -> EvaluatorPair:
        weight_shape = request.weight_shape
        batch = request.batch
        rows = weight_shape.output_rows
        cols = weight_shape.k
        raw_kernel = _build_raw_kernel(compiler, target, rows, cols, sm_count)
        fused_kernel = _build_fused_kernel(compiler, target, rows, cols, sm_count)
        activations = torch.randn(batch.value, cols, dtype=torch.float32, device="cuda")
        raw_evaluator = KernelCandidateEvaluator(raw_kernel, activations)
        fused_evaluator = KernelCandidateEvaluator(fused_kernel, activations)
        return EvaluatorPair(raw=raw_evaluator, fused=fused_evaluator)

    return build


def _build_raw_kernel(compiler, target, rows: int, cols: int, sm_count) -> RawBf16MatVecKernel:
    weights = _synthetic_bf16_weights(rows, cols)
    return RawBf16MatVecKernel(compiler, target, weights, sm_count)


def _build_fused_kernel(compiler, target, rows: int, cols: int, sm_count) -> Se12MatVecKernel:
    weights = _synthetic_bf16_weights(rows, cols)
    bit_pattern = weights.reshape(-1).view(torch.int16).cpu().numpy()
    from lws.domain.weights import Bf16Weights

    matrix_shape = MatrixShape(rows=rows, cols=cols)
    encoded = encode_tensor(Bf16Weights(bit_pattern=bit_pattern), matrix_shape)
    return Se12MatVecKernel(compiler, target, encoded, sm_count)


def _synthetic_bf16_weights(rows: int, cols: int) -> torch.Tensor:
    array = numpy.random.default_rng(0).standard_normal((rows, cols)).astype(numpy.float32)
    return torch.from_numpy(array).to(torch.bfloat16).to("cuda")


if __name__ == "__main__":
    sys.exit(main())
