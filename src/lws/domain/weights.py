"""Shared value objects for BF16 weight data, used across domain features."""

from __future__ import annotations

import dataclasses

import numpy


@dataclasses.dataclass(frozen=True)
class ElementCount:
    """How many weights a tensor (or a whole model) has."""

    value: int

    def __post_init__(self) -> None:
        value = self.value
        if value < 0:
            raise ValueError("ElementCount cannot be negative")


@dataclasses.dataclass(frozen=True)
class Bf16Weights:
    """A first-class collection: the int16 bit pattern of a flattened BF16 tensor."""

    bit_pattern: numpy.ndarray

    def __post_init__(self) -> None:
        pattern = self.bit_pattern
        if pattern.dtype != numpy.int16:
            raise ValueError("Bf16Weights requires an int16 bit-pattern view")
        if pattern.ndim != 1:
            raise ValueError("Bf16Weights requires a flattened 1-D array")

    def element_count(self) -> ElementCount:
        pattern = self.bit_pattern
        return ElementCount(int(pattern.size))


@dataclasses.dataclass(frozen=True)
class ExponentBytes:
    """First-class collection: one 8-bit BF16 exponent per weight (bits 14..7)."""

    value: numpy.ndarray


@dataclasses.dataclass(frozen=True)
class SignMantissaBytes:
    """First-class collection: one `(sign << 7) | mantissa` byte per weight
    (weight-codec design.md's `sm` stream)."""

    value: numpy.ndarray


def split_bit_pattern(weights: Bf16Weights) -> tuple[ExponentBytes, SignMantissaBytes]:
    """Splits a BF16 bit pattern into its exponent byte and sign+mantissa byte.

    exponent byte: bits 14..7 of the original 16-bit value.
    sign+mantissa byte: the sign bit (original bit 15) moved next to the 7
    mantissa bits (bits 6..0) — exactly SE12's `sm` stream.
    """
    pattern = weights.bit_pattern
    unsigned = pattern.astype(numpy.uint16)
    exponent = ((unsigned >> 7) & 0xFF).astype(numpy.uint8)
    sign_mantissa = (((unsigned >> 8) & 0x80) | (unsigned & 0x7F)).astype(numpy.uint8)
    return ExponentBytes(value=exponent), SignMantissaBytes(value=sign_mantissa)


def join_bit_pattern(exponent: ExponentBytes, sign_mantissa: SignMantissaBytes) -> Bf16Weights:
    """The exact inverse of split_bit_pattern (weight-codec design.md, "Decode")."""
    exponent_array = exponent.value
    sign_mantissa_array = sign_mantissa.value
    exponent_wide = exponent_array.astype(numpy.uint16)
    sign_mantissa_wide = sign_mantissa_array.astype(numpy.uint16)
    sign = (sign_mantissa_wide & 0x80) << 8
    mantissa = sign_mantissa_wide & 0x7F
    unsigned = sign | (exponent_wide << 7) | mantissa
    bit_pattern = unsigned.astype(numpy.int16)
    return Bf16Weights(bit_pattern=bit_pattern)
