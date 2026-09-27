"""SE12 reference encoder/decoder (weight-codec design.md, T-002).

Pure numpy integer bit manipulation, no torch. This is the bit-exact oracle
(principle P-004) that the GPU decode-only kernel (fused-decode-gemv T-004)
and the fused decode+GEMV kernel (T-005) are checked against.

Internally, everything works on plain (padded) 2-D numpy arrays — only the
public functions (`encode_tensor`, `decode_tensor`) and the container types
wrap primitives/arrays, matching how this codebase scopes calisthenics rule 3
(wrapping is required at public boundaries, not in every private helper).
"""

from __future__ import annotations

import dataclasses

import numpy

from lws.domain.se12.codebook import Codebook, MaskedExponents, build_codebook
from lws.domain.se12.tile import (
    ESCAPE_CODE,
    EscapeBudget,
    MatrixShape,
    TileShape,
    ValidMask,
    default_escape_budget,
    default_tile_shape,
)
from lws.domain.weights import Bf16Weights, ElementCount, ExponentBytes, SignMantissaBytes, join_bit_pattern, split_bit_pattern


@dataclasses.dataclass(frozen=True)
class EcBytes:
    """First-class collection: exponent codes, packed 2 per byte."""

    value: numpy.ndarray


@dataclasses.dataclass(frozen=True)
class EscBytes:
    """First-class collection: per-tile escape slots (the full exponent byte
    of each escaped weight, in order of appearance, zero-padded)."""

    value: numpy.ndarray


@dataclasses.dataclass(frozen=True)
class FallbackBitmap:
    """First-class collection: one bit per tile, set when that tile fell back to raw storage."""

    value: numpy.ndarray


@dataclasses.dataclass(frozen=True)
class FallbackRaw:
    """First-class collection: the untouched int16 bit pattern of every fallback tile's weights."""

    value: numpy.ndarray


@dataclasses.dataclass(frozen=True)
class PrimaryStreams:
    sm: SignMantissaBytes
    ec: EcBytes


@dataclasses.dataclass(frozen=True)
class FallbackData:
    bitmap: FallbackBitmap
    raw: FallbackRaw


@dataclasses.dataclass(frozen=True)
class EscapeHandling:
    esc: EscBytes
    fallback: FallbackData


@dataclasses.dataclass(frozen=True)
class Se12Streams:
    primary: PrimaryStreams
    escape_handling: EscapeHandling


@dataclasses.dataclass(frozen=True)
class TilingParameters:
    tile_shape: TileShape
    escape_budget: EscapeBudget


@dataclasses.dataclass(frozen=True)
class Se12Geometry:
    shape: MatrixShape
    tiling: TilingParameters


@dataclasses.dataclass(frozen=True)
class Se12Metadata:
    codebook: Codebook
    geometry: Se12Geometry


@dataclasses.dataclass(frozen=True)
class Se12Tensor:
    """A fully encoded SE12 tensor: every stream plus the metadata needed to decode it."""

    streams: Se12Streams
    metadata: Se12Metadata


def encode_tensor(weights: Bf16Weights, shape: MatrixShape) -> Se12Tensor:
    """Encodes a BF16 matrix into SE12 (US-002)."""
    tile_shape = default_tile_shape()
    escape_budget = default_escape_budget()
    padded, valid = _pad_bit_pattern(weights, shape, tile_shape)
    exponent_2d, sign_mantissa_2d = _split_2d(padded)
    codebook = build_codebook(MaskedExponents(exponent=ExponentBytes(value=exponent_2d), valid=ValidMask(value=valid)))
    code_2d, escape_mask_2d = _assign_codes(exponent_2d, codebook)
    real_escape_2d = escape_mask_2d & valid
    # Padding positions never carry a real escape (design note in tile.py):
    # forcing their code to a fixed, always-valid codebook entry (never
    # ESCAPE_CODE) keeps decode's rank count — which cannot itself tell
    # padding from real weights — aligned with the rank encode used when
    # filling `esc` from real escapes only.
    code_2d = numpy.where(valid, code_2d, 0).astype(numpy.uint8)
    tiled_exponent = _to_tiles(exponent_2d, tile_shape)
    tiled_escape = _to_tiles(real_escape_2d, tile_shape)
    tiled_code = _to_tiles(code_2d, tile_shape)
    tiled_sign_mantissa = _to_tiles(sign_mantissa_2d, tile_shape)
    tiled_raw = _to_tiles(padded, tile_shape)
    escape_rank = numpy.cumsum(tiled_escape.astype(numpy.int32), axis=-1) - 1
    escape_count = tiled_escape.sum(axis=-1)
    fallback_mask = escape_count > escape_budget.value
    primary = PrimaryStreams(sm=SignMantissaBytes(value=tiled_sign_mantissa.reshape(-1)), ec=EcBytes(value=_pack_ec(tiled_code)))
    escape_handling = EscapeHandling(
        esc=EscBytes(value=_pack_esc(tiled_exponent, tiled_escape, escape_rank, fallback_mask, escape_budget)),
        fallback=FallbackData(bitmap=FallbackBitmap(value=_pack_bitmap(fallback_mask)), raw=FallbackRaw(value=_pack_fallback_raw(tiled_raw, fallback_mask))),
    )
    tiling = TilingParameters(tile_shape=tile_shape, escape_budget=escape_budget)
    metadata = Se12Metadata(codebook=codebook, geometry=Se12Geometry(shape=shape, tiling=tiling))
    return Se12Tensor(streams=Se12Streams(primary=primary, escape_handling=escape_handling), metadata=metadata)


def decode_tensor(tensor: Se12Tensor) -> Bf16Weights:
    """Decodes SE12 streams back to the exact original BF16 bit pattern (US-002, AC-003, AC-004)."""
    streams = tensor.streams
    metadata = tensor.metadata
    geometry = metadata.geometry
    tiling = geometry.tiling
    tile_shape = tiling.tile_shape
    shape = geometry.shape
    padded_shape = _padded_shape(shape, tile_shape)
    grid = _tile_grid(padded_shape, tile_shape)
    tile_size = _tile_weight_count(tile_shape)
    primary = streams.primary
    escape_handling = streams.escape_handling
    sm = primary.sm
    ec = primary.ec
    code_flat = _unpack_ec(ec.value, grid, tile_size)
    esc = escape_handling.esc
    fallback = escape_handling.fallback
    bitmap = fallback.bitmap
    raw = fallback.raw
    fallback_mask = _unpack_bitmap(bitmap.value, grid)
    codebook = metadata.codebook
    codebook_array = numpy.array(codebook.exponents, dtype=numpy.uint8)
    escape_budget = tiling.escape_budget
    exponent_from_code = _decode_exponent(code_flat, codebook_array, esc.value, fallback_mask, grid, tile_size, escape_budget)
    exponent_2d = _from_tiles(exponent_from_code, tile_shape, padded_shape)
    sign_mantissa_flat = sm.value
    n_tile_rows, n_tile_cols = grid
    tiled_sign_mantissa = sign_mantissa_flat.reshape(n_tile_rows, n_tile_cols, tile_size)
    sign_mantissa_2d = _from_tiles(tiled_sign_mantissa, tile_shape, padded_shape)
    exponent_flat = numpy.ascontiguousarray(exponent_2d).reshape(-1)
    sign_mantissa_flat_contig = numpy.ascontiguousarray(sign_mantissa_2d).reshape(-1)
    rebuilt = join_bit_pattern(ExponentBytes(value=exponent_flat), SignMantissaBytes(value=sign_mantissa_flat_contig))
    padded_pattern_flat = rebuilt.bit_pattern
    padded_pattern = padded_pattern_flat.reshape(padded_shape)
    tiled_raw = _to_tiles(padded_pattern, tile_shape)
    tiled_raw_with_fallback = _restore_fallback_raw(tiled_raw, raw.value, fallback_mask)
    full_pattern = _from_tiles(tiled_raw_with_fallback, tile_shape, padded_shape)
    rows = shape.rows
    cols = shape.cols
    cropped = full_pattern[:rows, :cols]
    flattened = numpy.ascontiguousarray(cropped).reshape(-1)
    return Bf16Weights(bit_pattern=flattened)


def fallback_tile_count(tensor: Se12Tensor) -> ElementCount:
    """How many tiles were stored raw because they exceeded the escape budget (AC-005).

    The bitmap is packed 8 tiles per byte (numpy.packbits), so this counts
    set BITS, not the packed byte values themselves.
    """
    streams = tensor.streams
    escape_handling = streams.escape_handling
    fallback = escape_handling.fallback
    bitmap = fallback.bitmap
    packed = bitmap.value
    bits = numpy.unpackbits(packed)
    total = bits.sum()
    return ElementCount(int(total))


# --------------------------------------------------------------------------
# padding
# --------------------------------------------------------------------------


def _round_up(value: int, multiple: int) -> int:
    remainder = value % multiple
    if remainder == 0:
        return value
    return value + (multiple - remainder)


def _padded_shape(shape: MatrixShape, tile_shape: TileShape) -> tuple[int, int]:
    rows = shape.rows
    cols = shape.cols
    padded_rows = _round_up(rows, tile_shape.rows)
    padded_cols = _round_up(cols, tile_shape.cols)
    return padded_rows, padded_cols


def _pad_bit_pattern(weights: Bf16Weights, shape: MatrixShape, tile_shape: TileShape) -> tuple[numpy.ndarray, numpy.ndarray]:
    pattern = weights.bit_pattern
    rows = shape.rows
    cols = shape.cols
    matrix = pattern.reshape(rows, cols)
    padded_shape = _padded_shape(shape, tile_shape)
    padded = numpy.zeros(padded_shape, dtype=numpy.int16)
    padded[:rows, :cols] = matrix
    valid = numpy.zeros(padded_shape, dtype=bool)
    valid[:rows, :cols] = True
    return padded, valid


def _split_2d(padded: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    flat = Bf16Weights(bit_pattern=padded.reshape(-1))
    exponent, sign_mantissa = split_bit_pattern(flat)
    shape = padded.shape
    exponent_flat = exponent.value
    sign_mantissa_flat = sign_mantissa.value
    exponent_2d = exponent_flat.reshape(shape)
    sign_mantissa_2d = sign_mantissa_flat.reshape(shape)
    return exponent_2d, sign_mantissa_2d


# --------------------------------------------------------------------------
# tiling (flatten a padded 2-D array into (n_tile_rows, n_tile_cols, tile_size))
# --------------------------------------------------------------------------


def _tile_grid(padded_shape: tuple[int, int], tile_shape: TileShape) -> tuple[int, int]:
    padded_rows, padded_cols = padded_shape
    n_tile_rows = padded_rows // tile_shape.rows
    n_tile_cols = padded_cols // tile_shape.cols
    return n_tile_rows, n_tile_cols


def _tile_weight_count(tile_shape: TileShape) -> int:
    return tile_shape.rows * tile_shape.cols


def _to_tiles(array_2d: numpy.ndarray, tile_shape: TileShape) -> numpy.ndarray:
    rows, cols = array_2d.shape
    tile_rows = tile_shape.rows
    tile_cols = tile_shape.cols
    n_tile_rows = rows // tile_rows
    n_tile_cols = cols // tile_cols
    blocked = array_2d.reshape(n_tile_rows, tile_rows, n_tile_cols, tile_cols)
    transposed = blocked.transpose(0, 2, 1, 3)
    return transposed.reshape(n_tile_rows, n_tile_cols, tile_rows * tile_cols)


def _from_tiles(tiled: numpy.ndarray, tile_shape: TileShape, padded_shape: tuple[int, int]) -> numpy.ndarray:
    n_tile_rows, n_tile_cols, _ = tiled.shape
    tile_rows = tile_shape.rows
    tile_cols = tile_shape.cols
    blocked = tiled.reshape(n_tile_rows, n_tile_cols, tile_rows, tile_cols)
    transposed = blocked.transpose(0, 2, 1, 3)
    return transposed.reshape(padded_shape)


# --------------------------------------------------------------------------
# per-weight code assignment
# --------------------------------------------------------------------------


def _assign_codes(exponent_2d: numpy.ndarray, codebook: Codebook) -> tuple[numpy.ndarray, numpy.ndarray]:
    table = _code_lookup_table(codebook)
    code_2d = table[exponent_2d]
    escape_mask_2d = code_2d == ESCAPE_CODE
    return code_2d, escape_mask_2d


def _code_lookup_table(codebook: Codebook) -> numpy.ndarray:
    table = numpy.full(256, ESCAPE_CODE, dtype=numpy.uint8)
    exponents = codebook.exponents
    for code, exponent_value in enumerate(exponents):
        table[exponent_value] = code
    return table


# --------------------------------------------------------------------------
# ec: 4-bit codes, 2 per byte
# --------------------------------------------------------------------------


def _pack_ec(tiled_code: numpy.ndarray) -> numpy.ndarray:
    flat = tiled_code.reshape(-1).astype(numpy.uint8)
    low = flat[0::2]
    high = flat[1::2]
    return (low | (high << 4)).astype(numpy.uint8)


def _unpack_ec(ec_bytes: numpy.ndarray, grid: tuple[int, int], tile_size: int) -> numpy.ndarray:
    low = ec_bytes & 0x0F
    high = (ec_bytes >> 4) & 0x0F
    flat = numpy.empty(ec_bytes.size * 2, dtype=numpy.uint8)
    flat[0::2] = low
    flat[1::2] = high
    n_tile_rows, n_tile_cols = grid
    return flat.reshape(n_tile_rows, n_tile_cols, tile_size)


# --------------------------------------------------------------------------
# esc: per-tile escape exponent slots
# --------------------------------------------------------------------------


def _pack_esc(tiled_exponent: numpy.ndarray, tiled_escape: numpy.ndarray, escape_rank: numpy.ndarray, fallback_mask: numpy.ndarray, escape_budget: EscapeBudget) -> numpy.ndarray:
    n_tile_rows, n_tile_cols, _ = tiled_exponent.shape
    budget = escape_budget.value
    esc = numpy.zeros((n_tile_rows, n_tile_cols, budget), dtype=numpy.uint8)
    fits_budget = tiled_escape & (escape_rank < budget)
    not_fallback = ~fallback_mask[:, :, numpy.newaxis]
    slot_mask = fits_budget & not_fallback
    tile_row_index, tile_col_index, local_index = numpy.nonzero(slot_mask)
    ranks = escape_rank[tile_row_index, tile_col_index, local_index]
    values = tiled_exponent[tile_row_index, tile_col_index, local_index]
    esc[tile_row_index, tile_col_index, ranks] = values
    return esc.reshape(n_tile_rows * n_tile_cols, budget)


# --------------------------------------------------------------------------
# fallback bitmap and raw storage
# --------------------------------------------------------------------------


def _pack_bitmap(fallback_mask: numpy.ndarray) -> numpy.ndarray:
    flat = fallback_mask.reshape(-1)
    return numpy.packbits(flat)


def _unpack_bitmap(bitmap: numpy.ndarray, grid: tuple[int, int]) -> numpy.ndarray:
    n_tile_rows, n_tile_cols = grid
    total = n_tile_rows * n_tile_cols
    bits = numpy.unpackbits(bitmap, count=total)
    return bits.astype(bool).reshape(n_tile_rows, n_tile_cols)


def _pack_fallback_raw(tiled_raw: numpy.ndarray, fallback_mask: numpy.ndarray) -> numpy.ndarray:
    selected = tiled_raw[fallback_mask]
    return selected.reshape(-1)


def _restore_fallback_raw(tiled_decoded: numpy.ndarray, fallback_raw: numpy.ndarray, fallback_mask: numpy.ndarray) -> numpy.ndarray:
    fallback_count = int(fallback_mask.sum())
    if fallback_count == 0:
        return tiled_decoded
    tile_size = tiled_decoded.shape[-1]
    restored = fallback_raw.reshape(fallback_count, tile_size)
    result = tiled_decoded.copy()
    result[fallback_mask] = restored
    return result


def _decode_exponent(code_flat: numpy.ndarray, codebook_array: numpy.ndarray, esc_flat: numpy.ndarray, fallback_mask: numpy.ndarray, grid: tuple[int, int], tile_size: int, escape_budget: EscapeBudget) -> numpy.ndarray:
    n_tile_rows, n_tile_cols = grid
    budget = escape_budget.value
    esc = esc_flat.reshape(n_tile_rows, n_tile_cols, budget)
    escape_mask = code_flat == ESCAPE_CODE
    rank = numpy.cumsum(escape_mask.astype(numpy.int32), axis=-1) - 1
    safe_rank = numpy.clip(rank, 0, budget - 1)
    tile_row_index = numpy.arange(n_tile_rows).reshape(n_tile_rows, 1, 1)
    tile_col_index = numpy.arange(n_tile_cols).reshape(1, n_tile_cols, 1)
    gathered_escape = esc[tile_row_index, tile_col_index, safe_rank]
    non_escape = codebook_array[numpy.where(escape_mask, 0, code_flat)]
    exponent = numpy.where(escape_mask, gathered_escape, non_escape)
    return exponent
