"""Tiling geometry (weight-codec design.md, "Tiles (random access)").

The matrix is cut into fixed TILE_ROWS x TILE_COLS blocks so that every
tile has the same byte size and its address is pure arithmetic. Padding
choice made here (design.md does not spell this out): a matrix whose rows or
columns are not an exact multiple of the tile shape is padded with +0.0
(bit pattern 0x0000) up to the next multiple before tiling; the decoder always
knows the true (unpadded) shape from metadata and slices the padding back off.
This keeps every tile — including boundary tiles — exactly TILE_ROWS x
TILE_COLS weights, with no ragged special case for the kernel to branch on.
"""

from __future__ import annotations

import dataclasses

import numpy

DEFAULT_TILE_ROWS = 16
DEFAULT_TILE_COLS = 256
DEFAULT_ESCAPE_BUDGET = 16
CODEBOOK_SIZE = 15
ESCAPE_CODE = 15
ZERO_BIT_PATTERN = 0


@dataclasses.dataclass(frozen=True)
class TileShape:
    """One tile's dimensions, e.g. 16 rows x 256 columns = 4096 weights."""

    rows: int
    cols: int


@dataclasses.dataclass(frozen=True)
class EscapeBudget:
    """How many escaped exponents a tile may hold before it falls back to raw storage."""

    value: int


@dataclasses.dataclass(frozen=True)
class MatrixShape:
    """The tensor's true (unpadded) 2-D shape."""

    rows: int
    cols: int


@dataclasses.dataclass(frozen=True)
class ValidMask:
    """First-class collection: True where a (possibly padded) position holds a real weight."""

    value: numpy.ndarray


def default_tile_shape() -> TileShape:
    return TileShape(rows=DEFAULT_TILE_ROWS, cols=DEFAULT_TILE_COLS)


def default_escape_budget() -> EscapeBudget:
    return EscapeBudget(value=DEFAULT_ESCAPE_BUDGET)
