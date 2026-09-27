"""In-memory fakes for every application port, used by tests instead of real
adapters (constitution P-013: "every port has ... one in-memory fake used by
the tests"). This module is under tests/, so it is exempt from the
calisthenics rules that govern src/lws.
"""

from __future__ import annotations

import typing

from lws.application.ports.codec.pack_source import ShardOfTensors
from lws.application.ports.codec.packed_store import PackedShard
from lws.application.ports.codec.weight_source import SourceTensor


class FakeWeightSource:
    """A WeightSource backed by a fixed, in-memory list of tensors."""

    def __init__(self, items: typing.Sequence[SourceTensor]) -> None:
        self._items = list(items)

    def tensors(self) -> typing.Iterator[SourceTensor]:
        return iter(self._items)


class FakePackSource:
    """A PackSource backed by a fixed, in-memory mapping of shard name -> tensors."""

    def __init__(self, shards: typing.Sequence[tuple[str, typing.Sequence[SourceTensor]]]) -> None:
        self._shards = list(shards)

    def shards(self) -> typing.Iterator[ShardOfTensors]:
        for name, items in self._shards:
            yield ShardOfTensors(shard_name=name, tensors=iter(items))


class FakePackedStore:
    """A PackedStore that keeps every written shard in memory instead of on disk."""

    def __init__(self) -> None:
        self.written_shards: list[PackedShard] = []

    def write_shard(self, shard: PackedShard) -> None:
        self.written_shards.append(shard)
