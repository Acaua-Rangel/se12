"""Port: writes a packed model, one shard at a time (US-003).

A shard's entries are either SE12-encoded (eligible: BF16 and 2-D) or a
pass-through copy (everything else — ASM-002/Q-002), so a store only needs
to handle these two shapes; which one applies to a given tensor is decided
once, in the pack_model use case (weight-codec tasks.md, T-003 notes).
"""

from __future__ import annotations

import dataclasses
import typing

from lws.domain.se12.codec import Se12Tensor
from lws.domain.weights import RawTensor


@dataclasses.dataclass(frozen=True)
class Se12PackedTensor:
    name: str
    encoded: Se12Tensor


@dataclasses.dataclass(frozen=True)
class PassThroughTensor:
    name: str
    raw: RawTensor


PackedEntry = Se12PackedTensor | PassThroughTensor


@dataclasses.dataclass(frozen=True)
class PackedEntries:
    """First-class collection: every tensor entry destined for one shard."""

    items: tuple[PackedEntry, ...]


@dataclasses.dataclass(frozen=True)
class PackedShard:
    shard_name: str
    entries: PackedEntries


class PackedStore(typing.Protocol):
    def write_shard(self, shard: PackedShard) -> None:
        ...
