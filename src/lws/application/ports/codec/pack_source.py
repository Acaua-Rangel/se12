"""Port: weight tensors grouped by their original shard.

The packer needs this shard boundary to mirror it in the output (weight-codec
design.md, "Container": one .safetensors file per original shard); the
entropy analyzer's WeightSource does not, so it stays the simpler port.
"""

from __future__ import annotations

import typing

from lws.application.ports.codec.weight_source import SourceTensor


class ShardOfTensors(typing.NamedTuple):
    shard_name: str
    tensors: typing.Iterator[SourceTensor]


class PackSource(typing.Protocol):
    """Streams a model's tensors one shard at a time, in a single pass per shard."""

    def shards(self) -> typing.Iterator[ShardOfTensors]:
        ...
