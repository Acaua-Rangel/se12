"""Adapter: MatVecKernel over the raw-BF16 baseline and the fused SE12
kernels (US-005, AC-010, AC-011, AC-032).

The host-side precomputations (`fallback_offsets`, `row_escape_offset`) are
plain numpy functions, deliberately factored out so they are testable
without a GPU — they were verified against lws.domain.se12.codec (the CPU
oracle) with a full simulation of the kernel's per-warp rank logic before a
single line of CUDA C was written (see tests/test_spec_fused_gemv.py).

UNVERIFIED ON REAL HARDWARE past that point — the kernel launches themselves
(this file's cupy/DLPack glue and gemv_raw_bf16.cu / gemv_se12.cu) could not
be exercised at all while writing them (se12_common.cuh's note applies here
too).
"""

from __future__ import annotations

import pathlib

import cupy
import numpy
import torch

from lws.application.ports.gpu.kernel_compiler import CudaSource, KernelCompiler
from lws.domain.device.decisions import CompileTarget
from lws.domain.device.properties import SmCount
from lws.domain.launch.heuristic import GemvShape, default_launch_configuration
from lws.domain.se12.codec import Se12Tensor
from lws.domain.se12.tile import TileShape

KERNELS_DIR = pathlib.Path(__file__).parent / "kernels"
HEADER_FILE = KERNELS_DIR / "se12_common.cuh"
INCLUDE_DIRECTIVE = '#include "se12_common.cuh"'
TILE_SIZE = 16 * 256


# --------------------------------------------------------------------------
# host-side precomputation (pure numpy, testable without a GPU)
# --------------------------------------------------------------------------


def fallback_offsets(bitmap: numpy.ndarray, tile_count: int) -> numpy.ndarray:  # calisthenics: allow 3 — plain arrays, deliberately testable without wrapping
    """Each tile's 0-indexed rank among fallback tiles (numpy.packbits'
    default 'big' bit order — element i of a byte-group of 8 is bit
    (7 - i % 8), matching lws.domain.se12.codec._pack_bitmap exactly)."""
    bits = numpy.unpackbits(bitmap, count=tile_count)
    cumulative = numpy.cumsum(bits.astype(numpy.int32))
    return (cumulative - 1).astype(numpy.int32)


def row_escape_offset(ec_flat: numpy.ndarray, tile_count: int, n_tile_cols: int, tile_shape: TileShape) -> numpy.ndarray:  # calisthenics: allow 3 — plain arrays, deliberately testable without wrapping
    """For every (output row n, k-tile) pair: how many escapes occur in the
    EARLIER rows of n's own tile — the starting rank a warp adds its own
    __popc-based in-row rank to (gemv_se12.cu). A per-tile, per-local-row
    exclusive cumsum of the escape count, gathered into an [N, K/TILE_COLS]
    table; verified against a from-scratch brute-force count and against a
    full Python simulation of the kernel's rank logic (matching
    lws.domain.se12.codec's decode exactly) before this file was written.
    """
    tile_rows = tile_shape.rows
    tile_cols = tile_shape.cols
    codes = _unpack_ec(ec_flat, tile_count, tile_rows * tile_cols)
    codes_by_row = codes.reshape(tile_count, tile_rows, tile_cols)
    escape_mask = codes_by_row == 15
    per_row_counts = escape_mask.sum(axis=-1)
    inclusive = numpy.cumsum(per_row_counts, axis=1)
    exclusive = inclusive - per_row_counts  # (tile_count, tile_rows)
    n_tile_rows = tile_count // n_tile_cols
    by_tile_row = exclusive.reshape(n_tile_rows, n_tile_cols, tile_rows)
    # -> [n (= tile_row*tile_rows + local_row), k_tile]
    transposed = by_tile_row.transpose(0, 2, 1)
    return transposed.reshape(n_tile_rows * tile_rows, n_tile_cols).astype(numpy.int32)


def _unpack_ec(ec_flat: numpy.ndarray, tile_count: int, tile_size: int) -> numpy.ndarray:
    low = ec_flat & 0x0F
    high = (ec_flat >> 4) & 0x0F
    flat = numpy.empty(ec_flat.size * 2, dtype=numpy.uint8)
    flat[0::2] = low
    flat[1::2] = high
    return flat.reshape(tile_count, tile_size)


def tile_grid(shape_rows: int, shape_cols: int, tile_shape: TileShape) -> tuple[int, int]:  # calisthenics: allow 3 — plain dimensions, deliberately testable without wrapping
    n_tile_rows = _ceil_div(shape_rows, tile_shape.rows)
    n_tile_cols = _ceil_div(shape_cols, tile_shape.cols)
    return n_tile_rows, n_tile_cols


def _ceil_div(numerator: int, denominator: int) -> int:
    negated = -numerator // denominator
    return -negated


# --------------------------------------------------------------------------
# source loading
# --------------------------------------------------------------------------


def _combined_source(body_path: pathlib.Path, extra_decode_ops: int = 0) -> str:
    header_text = HEADER_FILE.read_text()
    body_text = body_path.read_text()
    combined = body_text.replace(INCLUDE_DIRECTIVE, header_text)
    if extra_decode_ops:
        define = f"#define LWS_EXTRA_DECODE_OPS {extra_decode_ops}\n"
        combined = define + combined
    return combined


# --------------------------------------------------------------------------
# torch <-> cupy crossing (DLPack, no host round trip)
# --------------------------------------------------------------------------


def _weights_to_device(weights: torch.Tensor) -> cupy.ndarray:
    contiguous = weights.contiguous()
    bit_view = contiguous.view(torch.int16)
    flattened = bit_view.reshape(-1)
    return cupy.from_dlpack(flattened)


def _activations_to_device(activations: torch.Tensor) -> cupy.ndarray:
    contiguous = activations.contiguous()
    upcast = contiguous.to(torch.float32) if contiguous.dtype == torch.bfloat16 else contiguous
    return cupy.from_dlpack(upcast)


def _output_from_device(output_device: cupy.ndarray, batch_size: int, output_rows: int) -> torch.Tensor:
    reshaped = output_device.reshape(batch_size, output_rows)
    return torch.from_dlpack(reshaped)


# --------------------------------------------------------------------------
# raw-BF16 baseline
# --------------------------------------------------------------------------


class RawBf16MatVecKernel:  # calisthenics: allow 8 — hot-path kernel wrapper, constitution's carve-out for adapters
    """Bound to one [N, K] BF16 weight matrix (principle P-009's baseline)."""

    def __init__(self, compiler: KernelCompiler, target: CompileTarget, weights: torch.Tensor, sm_count: SmCount) -> None:
        source = _combined_source(KERNELS_DIR / "gemv_raw_bf16.cu")
        handle = compiler.compile(CudaSource(value=source), target)
        self.handle = handle
        self.kernel = handle.function("lws_gemv_raw_bf16")
        self.weights_device = _weights_to_device(weights)
        shape = weights.shape
        self.output_rows = shape[0]
        self.k = shape[1]
        self.sm_count = sm_count

    def multiply(self, activations: torch.Tensor) -> torch.Tensor:
        shape = activations.shape
        batch_size = shape[0]
        gemv_shape = GemvShape(output_rows=self.output_rows, batch_size=batch_size)
        launch = default_launch_configuration(self.sm_count, gemv_shape)
        activations_device = _activations_to_device(activations)
        output_device = cupy.empty(self.output_rows * batch_size, dtype=cupy.float32)
        threads_per_block = launch.threads_per_block
        block_count = launch.blocks
        threads = threads_per_block.value
        blocks = block_count.value
        self.kernel(
            (blocks,),
            (threads,),
            (self.weights_device, activations_device, output_device, numpy.int32(self.output_rows), numpy.int32(self.k), numpy.int32(batch_size)),
        )
        return _output_from_device(output_device, batch_size, self.output_rows)


# --------------------------------------------------------------------------
# fused SE12 decode + GEMV
# --------------------------------------------------------------------------


class Se12MatVecKernel:  # calisthenics: allow 8 — hot-path kernel wrapper, constitution's carve-out for adapters
    """Bound to one SE12-encoded weight matrix (US-005, AC-010, AC-011)."""

    def __init__(self, compiler: KernelCompiler, target: CompileTarget, tensor: Se12Tensor, sm_count: SmCount, extra_decode_ops: int = 0) -> None:
        source = _combined_source(KERNELS_DIR / "gemv_se12.cu", extra_decode_ops)
        handle = compiler.compile(CudaSource(value=source), target)
        self.handle = handle
        self.kernel = handle.function("lws_gemv_se12")
        self.device_arrays = _upload_se12_streams(tensor)
        metadata = tensor.metadata
        geometry = metadata.geometry
        shape = geometry.shape
        self.output_rows = shape.rows
        self.k = shape.cols
        self.sm_count = sm_count

    def multiply(self, activations: torch.Tensor) -> torch.Tensor:
        shape = activations.shape
        batch_size = shape[0]
        gemv_shape = GemvShape(output_rows=self.output_rows, batch_size=batch_size)
        launch = default_launch_configuration(self.sm_count, gemv_shape)
        activations_device = _activations_to_device(activations)
        output_device = cupy.empty(self.output_rows * batch_size, dtype=cupy.float32)
        threads_per_block = launch.threads_per_block
        block_count = launch.blocks
        threads = threads_per_block.value
        blocks = block_count.value
        arrays = self.device_arrays
        self.kernel(
            (blocks,),
            (threads,),
            (
                arrays["sm"],
                arrays["ec"],
                arrays["esc"],
                arrays["codebook"],
                arrays["fallback_bitmap"],
                arrays["fallback_offsets"],
                arrays["fallback_raw"],
                arrays["row_escape_offset"],
                activations_device,
                output_device,
                numpy.int32(self.output_rows),
                numpy.int32(self.k),
                numpy.int32(batch_size),
            ),
        )
        return _output_from_device(output_device, batch_size, self.output_rows)


def _upload_se12_streams(tensor: Se12Tensor) -> dict:
    streams = tensor.streams
    primary = streams.primary
    escape_handling = streams.escape_handling
    fallback = escape_handling.fallback
    metadata = tensor.metadata
    codebook = metadata.codebook
    geometry = metadata.geometry
    shape = geometry.shape
    tiling = geometry.tiling
    tile_shape = tiling.tile_shape

    sm = primary.sm
    ec = primary.ec
    esc = escape_handling.esc
    bitmap = fallback.bitmap
    raw = fallback.raw

    n_tile_rows, n_tile_cols = tile_grid(shape.rows, shape.cols, tile_shape)
    tile_count = n_tile_rows * n_tile_cols
    bitmap_host = bitmap.value
    ec_host = ec.value
    fallback_offsets_host = fallback_offsets(bitmap_host, tile_count)
    row_escape_offset_host = row_escape_offset(ec_host, tile_count, n_tile_cols, tile_shape)
    codebook_array = numpy.array(codebook.exponents, dtype=numpy.uint8)

    return {
        "sm": cupy.asarray(sm.value),
        "ec": cupy.asarray(ec_host),
        "esc": cupy.asarray(esc.value),
        "codebook": cupy.asarray(codebook_array),
        "fallback_bitmap": cupy.asarray(bitmap_host),
        "fallback_offsets": cupy.asarray(fallback_offsets_host),
        "fallback_raw": cupy.asarray(raw.value),
        "row_escape_offset": cupy.asarray(row_escape_offset_host),
    }
