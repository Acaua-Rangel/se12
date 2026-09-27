"""Byte-size bookkeeping and the compression-ratio check (AC-006).

Not in T-003's original file list (weight-codec tasks.md); split out of
container.py so each module stays small (calisthenics rule 7) and focused —
container.py is about the format's identity, this one is about its size.
"""

from __future__ import annotations

import dataclasses

from lws.domain.se12.codec import Se12Tensor

MAX_PACKED_RATIO = 0.78


@dataclasses.dataclass(frozen=True)
class ByteCount:
    value: int

    def __post_init__(self) -> None:
        value = self.value
        if value < 0:
            raise ValueError("ByteCount cannot be negative")


@dataclasses.dataclass(frozen=True)
class ByteSizes:
    original: ByteCount
    packed: ByteCount


@dataclasses.dataclass(frozen=True)
class CompressionRatio:
    value: float


@dataclasses.dataclass(frozen=True)
class CompressionVerdict:
    within_target: bool


def compression_ratio(sizes: ByteSizes) -> CompressionRatio:
    original = sizes.original
    packed = sizes.packed
    ratio = packed.value / original.value
    return CompressionRatio(ratio)


def meets_compression_target(ratio: CompressionRatio) -> CompressionVerdict:
    """AC-006: packed ÷ original, metadata included, must be at most 0.78."""
    value = ratio.value
    within_target = value <= MAX_PACKED_RATIO
    return CompressionVerdict(within_target=within_target)


def add_sizes(first: ByteSizes, second: ByteSizes) -> ByteSizes:
    first_original = first.original
    first_packed = first.packed
    second_original = second.original
    second_packed = second.packed
    original_total = first_original.value + second_original.value
    packed_total = first_packed.value + second_packed.value
    return ByteSizes(original=ByteCount(original_total), packed=ByteCount(packed_total))


def zero_sizes() -> ByteSizes:
    return ByteSizes(original=ByteCount(0), packed=ByteCount(0))


def se12_tensor_byte_size(encoded: Se12Tensor) -> ByteCount:
    """Every byte an SE12-encoded tensor occupies: the sm/ec/esc/fallback
    streams plus the codebook (design.md, "Container"). Shared by pack_model
    (AC-006's ratio) and the entropy survey (AC-034's projected bits/weight),
    so the two features can never silently disagree on what "packed" means."""
    streams = encoded.streams
    primary = streams.primary
    escape_handling = streams.escape_handling
    fallback = escape_handling.fallback
    metadata = encoded.metadata
    codebook = metadata.codebook
    stream_total = _stream_bytes(primary.sm) + _stream_bytes(primary.ec) + _stream_bytes(escape_handling.esc)
    stream_total += _stream_bytes(fallback.bitmap) + _stream_bytes(fallback.raw)
    codebook_exponents = codebook.exponents
    codebook_total = len(codebook_exponents)
    return ByteCount(stream_total + codebook_total)


def _stream_bytes(stream) -> int:
    array = stream.value
    size = array.nbytes
    return int(size)
