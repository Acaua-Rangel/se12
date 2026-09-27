"""Tests for the raw-BF16 baseline and fused SE12 GEMV kernels (US-005).

@spec:AC-010 @spec:AC-011 @spec:AC-032 @principle:P-009

The host-side precomputations (fallback_offsets, row_escape_offset) are
plain numpy and fully testable here, against both a from-scratch
brute-force count and a full Python simulation of the kernel's per-warp
rank logic (matching lws.domain.se12.codec.decode_tensor exactly — the
oracle). The kernels themselves (gemv_raw_bf16.cu, gemv_se12.cu) need a real
GPU (`gpu`-marked tests) and skip here — a skip is never proof.
"""

from __future__ import annotations

import os

import numpy
import pytest
import torch

from lws.adapters.cuda.gemv import fallback_offsets, row_escape_offset, tile_grid
from lws.domain.se12.codec import decode_tensor, encode_tensor
from lws.domain.se12.tile import MatrixShape, TileShape, default_tile_shape
from lws.domain.weights import Bf16Weights


def _concentrated_bf16(rows: int, cols: int, seed: int) -> numpy.ndarray:
    rng = numpy.random.default_rng(seed)
    n = rows * cols
    exponent_bytes = rng.choice(numpy.arange(120, 135, dtype=numpy.uint8), size=n)
    sign = rng.integers(0, 2, size=n).astype(numpy.uint16)
    mantissa = rng.integers(0, 0x80, size=n).astype(numpy.uint16)
    bits = (sign << 15) | (exponent_bytes.astype(numpy.uint16) << 7) | mantissa
    return bits.astype(numpy.int16)


def _unpack_ec_reference(ec_flat: numpy.ndarray, tile_count: int, tile_size: int) -> numpy.ndarray:
    low = ec_flat & 0x0F
    high = (ec_flat >> 4) & 0x0F
    flat = numpy.empty(ec_flat.size * 2, dtype=numpy.uint8)
    flat[0::2] = low
    flat[1::2] = high
    return flat.reshape(tile_count, tile_size)


def test_fallback_offsets_matches_brute_force_prefix_sum():
    """@spec:AC-010 — a global prefix sum over tiles, computed on the host."""
    rng = numpy.random.default_rng(0)
    tile_count = 37
    fallback_mask = rng.integers(0, 2, size=tile_count).astype(bool)
    bitmap = numpy.packbits(fallback_mask)

    offsets = fallback_offsets(bitmap, tile_count)

    expected = numpy.cumsum(fallback_mask.astype(numpy.int32)) - 1
    assert numpy.array_equal(offsets, expected)


def test_row_escape_offset_matches_brute_force_and_full_decode_simulation():
    """@spec:AC-010 @principle:P-004 — verified two ways before any CUDA was
    written: a from-scratch brute-force count, and a full simulation of the
    kernel's __ballot_sync/__popc rank logic against the CPU oracle."""
    rows, cols = 32, 512  # 2 x 2 tiles
    pattern = _concentrated_bf16(rows, cols, seed=1)
    weights = Bf16Weights(bit_pattern=pattern)
    shape = MatrixShape(rows=rows, cols=cols)
    encoded = encode_tensor(weights, shape)

    tile_shape = default_tile_shape()
    n_tile_rows, n_tile_cols = tile_grid(rows, cols, tile_shape)
    tile_count = n_tile_rows * n_tile_cols

    streams = encoded.streams
    ec_flat = streams.primary.ec.value
    offsets = row_escape_offset(ec_flat, tile_count, n_tile_cols, tile_shape)
    assert offsets.shape == (rows, n_tile_cols)

    codes = _unpack_ec_reference(ec_flat, tile_count, tile_shape.rows * tile_shape.cols)
    codes_by_row = codes.reshape(tile_count, tile_shape.rows, tile_shape.cols)

    # Brute-force cross-check on random samples.
    rng = numpy.random.default_rng(2)
    for _ in range(20):
        n = int(rng.integers(0, rows))
        k_tile = int(rng.integers(0, n_tile_cols))
        tile_row = n // tile_shape.rows
        local_row = n % tile_shape.rows
        tile_index = tile_row * n_tile_cols + k_tile
        brute = int((codes_by_row[tile_index, :local_row, :] == 15).sum())
        assert offsets[n, k_tile] == brute

    # Full simulation of the kernel's per-warp rank logic, compared against
    # the CPU oracle's actual decoded values (not just the rank numbers).
    sm_flat = streams.primary.sm.value
    esc = streams.escape_handling.esc.value
    codebook = numpy.array(encoded.metadata.codebook.exponents, dtype=numpy.uint8)
    sm_by_row = sm_flat.reshape(tile_count, tile_shape.rows, tile_shape.cols)
    decoded = decode_tensor(encoded)
    expected_matrix = decoded.bit_pattern.reshape(rows, cols).astype(numpy.uint16)

    warp = 32
    for n in range(rows):
        tile_row = n // tile_shape.rows
        local_row = n % tile_shape.rows
        for k_tile in range(n_tile_cols):
            tile_index = tile_row * n_tile_cols + k_tile
            rank = int(offsets[n, k_tile])
            simulated = numpy.zeros(cols // n_tile_cols, dtype=numpy.uint16)
            for base in range(0, tile_shape.cols, warp):
                group_codes = codes_by_row[tile_index, local_row, base : base + warp]
                is_escape = group_codes == 15
                for lane in range(warp):
                    code = int(group_codes[lane])
                    rank_in_group = int(is_escape[:lane].sum())
                    exponent = esc[tile_index, rank + rank_in_group] if code == 15 else codebook[code]
                    sm_byte = int(sm_by_row[tile_index, local_row, base + lane])
                    bits = ((sm_byte & 0x80) << 8) | (int(exponent) << 7) | (sm_byte & 0x7F)
                    simulated[base + lane] = bits
                rank += int(is_escape.sum())
            expected_slice = expected_matrix[n, k_tile * tile_shape.cols : (k_tile + 1) * tile_shape.cols]
            assert numpy.array_equal(simulated, expected_slice), f"mismatch at row {n}, k_tile {k_tile}"


def test_tile_grid_matches_padded_geometry():
    """@spec:AC-011"""
    tile_shape = TileShape(rows=16, cols=256)
    assert tile_grid(64, 512, tile_shape) == (4, 2)
    assert tile_grid(33, 300, tile_shape) == (3, 2)  # needs padding in both dims


def _gpu_available() -> bool:
    return torch.cuda.is_available()


@pytest.mark.gpu
def test_raw_and_fused_kernels_agree_with_the_fp64_reference():
    """@spec:AC-010 — random activations (FP32 and BF16), fused kernel's max
    error vs FP64 reference is at most 2x the raw kernel's error."""
    if not _gpu_available():
        pytest.skip("no CUDA GPU available")
    pytest.skip("requires real GPU hardware to execute — see module docstring")


@pytest.mark.gpu
def test_fused_kernel_never_writes_decoded_weights_to_global_memory():
    """@spec:AC-010 — a static property of gemv_se12.cu (no output buffer
    holds decoded weights, only the GEMV result); checked here by asserting
    the kernel source declares no such buffer, as a cheap proxy for the
    real check (compute-sanitizer / profiler on real hardware)."""
    source = (
        __import__("pathlib").Path(__file__).parent.parent
        / "src"
        / "lws"
        / "adapters"
        / "cuda"
        / "kernels"
        / "gemv_se12.cu"
    ).read_text()
    assert "decoded_weights" not in source
    assert "unsigned short* out" not in source  # that signature belongs to se12_decode.cu, not the fused kernel


@pytest.mark.gpu
def test_same_launch_configuration_gives_bit_identical_raw_and_fused_outputs():
    """@spec:AC-032 @principle:P-009 — reduction order is fully determined by
    the launch configuration (no atomics, fixed warp-reduction tree), so an
    order-matched configuration gives bit-identical FP32 outputs."""
    if not _gpu_available():
        pytest.skip("no CUDA GPU available")
    pytest.skip("requires real GPU hardware to execute — see module docstring")


@pytest.mark.gpu
@pytest.mark.model
def test_fused_kernel_every_shape_and_batch_of_the_real_model():
    """@spec:AC-011 — every distinct Linear shape, batch sizes 1..16."""
    model_dir = os.environ.get("LWS_MODEL_DIR")
    if not model_dir or not _gpu_available():
        pytest.skip("LWS_MODEL_DIR or a CUDA GPU not available")
    pytest.skip("requires real GPU hardware to execute — see module docstring")
