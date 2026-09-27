"""Port: decodes SE12 tensors back to exact BF16 (US-004, AC-009).

Whole tensor, or just a set of matrix rows — the latter is for the
compressed embedding lookup (model-integration, later): decoding only the
tile-rows that cover the looked-up token rows, not the whole embedding
table (weight-codec Q-002).
"""

from __future__ import annotations

import dataclasses
import typing

from lws.domain.se12.codec import Se12Tensor
from lws.domain.weights import Bf16Weights


@dataclasses.dataclass(frozen=True)
class RowIndices:
    """First-class collection: matrix row indices to decode, in request order
    (not sorted, not deduplicated — the caller's lookup order is preserved)."""

    values: tuple[int, ...]


class WeightDecoder(typing.Protocol):
    def decode(self, tensor: Se12Tensor) -> Bf16Weights:
        ...

    def decode_rows(self, tensor: Se12Tensor, rows: RowIndices) -> Bf16Weights:
        """The decoded bit pattern of exactly `len(rows.values)` full rows,
        one after another, in `rows.values`' order."""
        ...
