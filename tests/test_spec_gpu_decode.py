"""Tests for the GPU decode-only kernel (US-004, AC-009).

@spec:AC-009 @principle:P-004

The CPU-side tests exercise the WeightDecoder port's CPU fake
(Se12CodecWeightDecoder, backed by lws.domain.se12.codec — already
extensively fuzz-tested in test_spec_se12_roundtrip.py). The `gpu`-marked
tests exercise the REAL CUDA kernel (se12_decode.cu) and are the actual
proof this feature asks for (AC-009: "every rebuilt tensor is identical bit
for bit ... on any supported GPU") — they skip on a machine without a GPU
(this one), because a skip is never proof (see scripts/verify.sh).
"""

from __future__ import annotations

import os

import numpy
import pytest
import torch

from lws.application.ports.gpu.weight_decoder import RowIndices
from lws.domain.se12.codec import encode_tensor
from lws.domain.se12.tile import MatrixShape
from lws.domain.weights import Bf16Weights
from tests.fakes import Se12CodecWeightDecoder


def _bits(sign: int, exponent: int, mantissa: int) -> int:
    return (sign << 15) | (exponent << 7) | mantissa


def _random_matrix(rows: int, cols: int, seed: int) -> numpy.ndarray:
    rng = numpy.random.default_rng(seed)
    bits = rng.integers(0, 65536, size=rows * cols, dtype=numpy.uint32).astype(numpy.uint16)
    return bits.astype(numpy.int16)


def test_cpu_oracle_decode_is_bit_exact_on_edge_cases_and_random_data():
    """@spec:AC-009 @principle:P-004 — the oracle the GPU kernel is checked against."""
    edge_cases = [_bits(0, 0, 0), _bits(1, 0, 0), _bits(0, 0xFF, 0), _bits(1, 0xFF, 0), _bits(0, 0xFF, 0x7F), _bits(1, 0xFF, 0x01)]
    random_values = list(_random_matrix(1, 200, seed=3))
    pattern = numpy.array(edge_cases, dtype=numpy.uint16).astype(numpy.int16)
    pattern = numpy.concatenate([pattern, numpy.array(random_values, dtype=numpy.int16)])
    weights = Bf16Weights(bit_pattern=pattern)
    shape = MatrixShape(rows=1, cols=pattern.size)
    encoded = encode_tensor(weights, shape)

    decoder = Se12CodecWeightDecoder()
    decoded = decoder.decode(encoded)

    assert numpy.array_equal(decoded.bit_pattern, pattern)


def test_cpu_oracle_decode_rows_matches_the_full_decode_sliced():
    """@spec:AC-009 — the row-subset path (used by the compressed embedding lookup, model-integration)."""
    pattern = _random_matrix(48, 512, seed=4)
    weights = Bf16Weights(bit_pattern=pattern)
    shape = MatrixShape(rows=48, cols=512)
    encoded = encode_tensor(weights, shape)

    decoder = Se12CodecWeightDecoder()
    row_indices = RowIndices(values=(5, 0, 47, 12))
    partial = decoder.decode_rows(encoded, row_indices)

    matrix = pattern.reshape(48, 512)
    expected = matrix[[5, 0, 47, 12], :].reshape(-1)
    assert numpy.array_equal(partial.bit_pattern, expected)


def _gpu_available() -> bool:
    return torch.cuda.is_available()


@pytest.mark.gpu
@pytest.mark.parametrize("force_ptx", [False, True], ids=["sass", "ptx"])
def test_gpu_decode_matches_the_cpu_oracle_bit_exact(force_ptx: bool):
    """@spec:AC-009 @principle:P-004 — real GPU kernel vs the CPU oracle,
    parametrized over the SASS and the forced-PTX target (AC-022)."""
    if not _gpu_available():
        pytest.skip("no CUDA GPU available")
    if force_ptx:
        os.environ["LWS_FORCE_PTX"] = "1"
    else:
        os.environ.pop("LWS_FORCE_PTX", None)

    from lws.adapters.cuda.gpu_probe import CudaGpuProbe
    from lws.adapters.cuda.nvrtc_compiler import NvrtcKernelCompiler
    from lws.adapters.cuda.se12_decoder import CudaSe12WeightDecoder
    from lws.domain.device.decisions import ForcePtx, choose_compile_target

    probe = CudaGpuProbe()
    properties = probe.properties()
    identity_and_compute = properties.identity_and_compute
    compute = identity_and_compute.compute
    capability = compute.capability
    memory_and_nvrtc = properties.memory_and_nvrtc
    nvrtc_targets = memory_and_nvrtc.nvrtc_targets
    target = choose_compile_target(capability, nvrtc_targets, ForcePtx(force_ptx))

    compiler = NvrtcKernelCompiler()
    gpu_decoder = CudaSe12WeightDecoder(compiler, target)
    cpu_decoder = Se12CodecWeightDecoder()

    pattern = _random_matrix(64, 512, seed=7)
    weights = Bf16Weights(bit_pattern=pattern)
    shape = MatrixShape(rows=64, cols=512)
    encoded = encode_tensor(weights, shape)

    expected = cpu_decoder.decode(encoded)
    actual = gpu_decoder.decode(encoded)

    assert numpy.array_equal(actual.bit_pattern, expected.bit_pattern)


@pytest.mark.gpu
@pytest.mark.model
def test_gpu_decode_every_tensor_of_the_real_model():
    """@spec:AC-009 — every tensor of the real model, when available; skips (not proof) otherwise."""
    model_dir = os.environ.get("LWS_MODEL_DIR")
    if not model_dir or not _gpu_available():
        pytest.skip("LWS_MODEL_DIR or a CUDA GPU not available")

    import pathlib

    import safetensors as st

    from lws.adapters.cuda.gpu_probe import CudaGpuProbe
    from lws.adapters.cuda.nvrtc_compiler import NvrtcKernelCompiler
    from lws.adapters.cuda.se12_decoder import CudaSe12WeightDecoder
    from lws.domain.device.decisions import ForcePtx, choose_compile_target

    probe = CudaGpuProbe()
    properties = probe.properties()
    identity_and_compute = properties.identity_and_compute
    compute = identity_and_compute.compute
    capability = compute.capability
    memory_and_nvrtc = properties.memory_and_nvrtc
    nvrtc_targets = memory_and_nvrtc.nvrtc_targets
    target = choose_compile_target(capability, nvrtc_targets, ForcePtx(False))
    compiler = NvrtcKernelCompiler()
    gpu_decoder = CudaSe12WeightDecoder(compiler, target)
    cpu_decoder = Se12CodecWeightDecoder()

    checked = 0
    for shard_path in sorted(pathlib.Path(model_dir).glob("*.safetensors")):
        with st.safe_open(str(shard_path), framework="pt") as shard:
            for name in shard.keys():
                tensor = shard.get_tensor(name)
                if tensor.dtype != torch.bfloat16 or tensor.ndim != 2:
                    continue
                rows, cols = tensor.shape
                flat = tensor.reshape(-1).view(torch.int16).numpy().copy()
                weights = Bf16Weights(bit_pattern=flat)
                shape = MatrixShape(rows=rows, cols=cols)
                encoded = encode_tensor(weights, shape)
                expected = cpu_decoder.decode(encoded)
                actual = gpu_decoder.decode(encoded)
                assert numpy.array_equal(actual.bit_pattern, expected.bit_pattern), f"mismatch in {name}"
                checked += 1
    assert checked > 0
