"""Adapter: WeightDecoder on a real GPU, via se12_decode.cu (US-004, AC-009).

Two ways to get a decoded tensor out, deliberately different:
  - `decode()` (the WeightDecoder port's contract) returns a CPU-side
    Bf16Weights — a real host copy, on purpose: this is the verification
    contract AC-009 checks bit-exactness against, and CPU-side data is what
    a bit-exact comparison needs anyway.
  - `decode_to_device()` returns the decoded weights as a torch tensor still
    resident on the GPU, crossed from CuPy via DLPack with no host round
    trip — the path model-integration's "exact" mode (a later feature) uses
    for its decode-then-multiply fallback, where a host copy would defeat
    the point.
`fallback_offsets` (each fallback tile's rank among fallback tiles) is a
global prefix sum over the whole tensor's tile count, computed once on the
host with numpy: cheap, and not the kind of per-weight work that belongs on
the GPU (fused-decode-gemv design.md).

UNVERIFIED ON REAL HARDWARE — see se12_common.cuh's note. This adapter could
not be exercised at all while writing it (no NVIDIA driver in this
environment; confirmed even NVRTC fails to load here).
"""

from __future__ import annotations

import math
import pathlib

import cupy
import numpy
import torch

from lws.application.ports.gpu.kernel_compiler import CudaSource, KernelCompiler
from lws.application.ports.gpu.weight_decoder import RowIndices
from lws.domain.device.decisions import CompileTarget
from lws.domain.se12.codec import Se12Tensor
from lws.domain.weights import Bf16Weights

KERNEL_FUNCTION_NAME = "lws_se12_decode"
KERNEL_FILE = pathlib.Path(__file__).parent / "kernels" / "se12_decode.cu"
HEADER_FILE = pathlib.Path(__file__).parent / "kernels" / "se12_common.cuh"
INCLUDE_DIRECTIVE = '#include "se12_common.cuh"'
THREADS_PER_BLOCK = 256


class CudaSe12WeightDecoder:
    """Decodes SE12 tensors with the compiled `lws_se12_decode` kernel."""

    def __init__(self, compiler: KernelCompiler, target: CompileTarget) -> None:
        source = _combined_source()
        handle = compiler.compile(CudaSource(value=source), target)
        self.handle = handle
        self.kernel = handle.function(KERNEL_FUNCTION_NAME)

    def decode(self, tensor: Se12Tensor) -> Bf16Weights:
        cropped_device = self._decode_cropped(tensor)
        return _to_bf16_weights_host(cropped_device)

    def decode_to_device(self, tensor: Se12Tensor) -> torch.Tensor:
        """The decoded tensor as a GPU-resident torch BF16 tensor, never
        touching host memory (model-integration's exact-mode fallback)."""
        cropped_device = self._decode_cropped(tensor)
        return _to_torch_bf16_device(cropped_device)

    def decode_rows(self, tensor: Se12Tensor, rows: RowIndices) -> Bf16Weights:
        # First draft: decode the whole tensor and select rows on the GPU
        # array before the final host-side reshape. A faster path (launch
        # the kernel only for the tile-rows the requested rows fall into)
        # is a later optimization, not a correctness requirement here.
        full = self.decode(tensor)
        return _select_rows(full, tensor, rows)

    def _decode_cropped(self, tensor: Se12Tensor) -> cupy.ndarray:
        tile_count, padded_shape, tile_shape = _tile_geometry(tensor)
        device_arrays = _upload_streams(tensor, tile_count)
        decoded_tiles = _run_kernel(self.kernel, device_arrays, tile_count)
        return _untile_and_crop(decoded_tiles, tensor, padded_shape, tile_shape)


def _combined_source() -> str:
    header_text = HEADER_FILE.read_text()
    body_text = KERNEL_FILE.read_text()
    return body_text.replace(INCLUDE_DIRECTIVE, header_text)


def _tile_geometry(tensor: Se12Tensor):
    metadata = tensor.metadata
    geometry = metadata.geometry
    tiling = geometry.tiling
    tile_shape = tiling.tile_shape
    shape = geometry.shape
    padded_rows = _round_up(shape.rows, tile_shape.rows)
    padded_cols = _round_up(shape.cols, tile_shape.cols)
    tile_rows_count = padded_rows // tile_shape.rows
    tile_cols_count = padded_cols // tile_shape.cols
    tile_count = tile_rows_count * tile_cols_count
    return tile_count, (padded_rows, padded_cols), tile_shape


def _round_up(value: int, multiple: int) -> int:
    remainder = value % multiple
    if remainder == 0:
        return value
    return value + (multiple - remainder)


def _upload_streams(tensor: Se12Tensor, tile_count: int) -> dict:
    streams = tensor.streams
    primary = streams.primary
    escape_handling = streams.escape_handling
    fallback = escape_handling.fallback
    metadata = tensor.metadata
    codebook = metadata.codebook

    sm = primary.sm
    ec = primary.ec
    esc = escape_handling.esc
    bitmap = fallback.bitmap
    raw = fallback.raw
    sm_device = cupy.asarray(sm.value)
    ec_device = cupy.asarray(ec.value)
    esc_device = cupy.asarray(esc.value)
    codebook_device = cupy.asarray(numpy.array(codebook.exponents, dtype=numpy.uint8))
    bitmap_host = bitmap.value
    bitmap_device = cupy.asarray(bitmap_host)
    fallback_offsets_host = _fallback_offsets(bitmap_host, tile_count)
    fallback_offsets_device = cupy.asarray(fallback_offsets_host)
    fallback_raw_device = cupy.asarray(raw.value)

    return {
        "sm": sm_device,
        "ec": ec_device,
        "esc": esc_device,
        "codebook": codebook_device,
        "fallback_bitmap": bitmap_device,
        "fallback_offsets": fallback_offsets_device,
        "fallback_raw": fallback_raw_device,
    }


def _fallback_offsets(bitmap: numpy.ndarray, tile_count: int) -> numpy.ndarray:
    bits = numpy.unpackbits(bitmap, count=tile_count)
    cumulative = numpy.cumsum(bits.astype(numpy.int32))
    offsets = cumulative - 1
    return offsets.astype(numpy.int32)


def _run_kernel(kernel, device_arrays: dict, tile_count: int) -> cupy.ndarray:
    tile_size = 16 * 256
    out_device = cupy.empty(tile_count * tile_size, dtype=cupy.uint16)
    blocks = math.ceil(tile_count / THREADS_PER_BLOCK)
    kernel(
        (blocks,),
        (THREADS_PER_BLOCK,),
        (
            device_arrays["sm"],
            device_arrays["ec"],
            device_arrays["esc"],
            device_arrays["codebook"],
            device_arrays["fallback_bitmap"],
            device_arrays["fallback_offsets"],
            device_arrays["fallback_raw"],
            out_device,
            numpy.int32(tile_count),
        ),
    )
    return out_device


def _untile_and_crop(decoded_tiles: cupy.ndarray, tensor: Se12Tensor, padded_shape: tuple[int, int], tile_shape) -> cupy.ndarray:
    """Un-tiles the kernel's tile-major output back to (rows, cols) order
    (mirroring lws.domain.se12.codec._from_tiles exactly) and crops the
    padding off — all as GPU array reshapes/copies, no host round trip."""
    padded_rows, padded_cols = padded_shape
    tile_rows = tile_shape.rows
    tile_cols = tile_shape.cols
    n_tile_rows = padded_rows // tile_rows
    n_tile_cols = padded_cols // tile_cols
    blocked = decoded_tiles.reshape(n_tile_rows, n_tile_cols, tile_rows, tile_cols)
    transposed = blocked.transpose(0, 2, 1, 3)
    padded_matrix = transposed.reshape(padded_rows, padded_cols)
    metadata = tensor.metadata
    geometry = metadata.geometry
    shape = geometry.shape
    cropped = padded_matrix[: shape.rows, : shape.cols]
    return cupy.ascontiguousarray(cropped)


def _to_bf16_weights_host(cropped_device: cupy.ndarray) -> Bf16Weights:
    """The verification path (WeightDecoder.decode): a genuine host copy,
    because a bit-exact comparison (AC-009) needs CPU-side data anyway."""
    host_array = cupy.asnumpy(cropped_device)
    contiguous = numpy.ascontiguousarray(host_array).astype(numpy.uint16)
    signed = contiguous.view(numpy.int16).reshape(-1)
    return Bf16Weights(bit_pattern=signed)


def _to_torch_bf16_device(cropped_device: cupy.ndarray) -> torch.Tensor:
    """The production path: stays on the GPU, crossed via DLPack, reinterpreted
    as BF16 (the uint16 bit pattern IS the BF16 memory layout — no conversion)."""
    torch_view = _torch_from_cupy(cropped_device)
    return torch_view.view(torch.bfloat16)


def _select_rows(full: Bf16Weights, tensor: Se12Tensor, rows: RowIndices) -> Bf16Weights:
    metadata = tensor.metadata
    geometry = metadata.geometry
    shape = geometry.shape
    cols = shape.cols
    pattern = full.bit_pattern
    matrix = pattern.reshape(-1, cols)
    row_values = list(rows.values)
    selected = matrix[row_values, :]
    flattened = numpy.ascontiguousarray(selected).reshape(-1)
    return Bf16Weights(bit_pattern=flattened)


def _torch_from_cupy(array: cupy.ndarray) -> torch.Tensor:
    """DLPack crossing (the modern `__dlpack__` protocol, not the legacy
    `.toDlpack()` capsule dance), used by callers that want the decoded
    tensor back as torch without a host round trip."""
    return torch.utils.dlpack.from_dlpack(array)
