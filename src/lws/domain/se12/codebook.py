"""Exponent codebook: the 15 most frequent 8-bit exponent values in a tensor
(weight-codec design.md). A weight whose exponent matches one of these 15
gets a 4-bit code 0..14; any other exponent is the ESCAPE code (15) and its
full byte goes into that tile's `esc` slots instead.
"""

from __future__ import annotations

import dataclasses

import numpy

from lws.domain.se12.tile import CODEBOOK_SIZE, ValidMask
from lws.domain.weights import ExponentBytes

EXPONENT_ALPHABET_SIZE = 256


@dataclasses.dataclass(frozen=True)
class Codebook:
    """First-class collection: the 15 exponent byte values, most frequent first."""

    exponents: tuple[int, ...]

    def __post_init__(self) -> None:
        exponents = self.exponents
        length = len(exponents)
        if length != CODEBOOK_SIZE:
            raise ValueError(f"Codebook needs exactly {CODEBOOK_SIZE} entries, got {length}")


@dataclasses.dataclass(frozen=True)
class MaskedExponents:
    """The tensor's exponent bytes, together with which positions are real
    (as opposed to padding — weight-codec, tile.py's padding note)."""

    exponent: ExponentBytes
    valid: ValidMask


def build_codebook(masked: MaskedExponents) -> Codebook:
    """The tensor-wide codebook, built only from real (non-padding) weights.

    Argsort over the full 256-entry histogram always yields exactly 15
    entries, even when a tensor uses fewer than 15 distinct exponents: the
    remaining slots point at unused (zero-count) exponent values, which no
    weight's code will ever reference. `kind="stable"` makes the choice
    deterministic on ties (principle P-010: packing is deterministic).
    """
    exponent = masked.exponent
    valid = masked.valid
    exponent_byte = exponent.value
    valid_mask = valid.value
    real_exponents = exponent_byte[valid_mask]
    counts = numpy.bincount(real_exponents, minlength=EXPONENT_ALPHABET_SIZE)
    order = numpy.argsort(counts, kind="stable")[::-1]
    top = order[:CODEBOOK_SIZE]
    values = tuple(int(value) for value in top)
    return Codebook(exponents=values)
