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
