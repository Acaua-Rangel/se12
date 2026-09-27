"""Port: a source of named weight tensors, streamed one at a time.

An adapter (e.g. lws.adapters.safetensors.weight_source) reads shards lazily
from disk; a test fake (tests/fakes.py) serves tensors held in memory. Either
way, the domain only ever sees a Bf16Weights value object per eligible tensor
(design.md, weight-codec) — never a raw file handle.
"""

from __future__ import annotations

import typing

from lws.domain.weights import RawTensor, ShapedWeights


class EligibleTensor(typing.NamedTuple):
    """A BF16 tensor, ready for entropy measurement or encoding."""

    name: str
    tensor: ShapedWeights


class IneligibleSourceTensor(typing.NamedTuple):
    """A tensor that is not BF16 — reported (AC-002) and, for packing, copied
    through unchanged (AC-007), never crashed on."""

    name: str
    tensor: RawTensor


SourceTensor = EligibleTensor | IneligibleSourceTensor


class WeightSource(typing.Protocol):
    """Streams a model's tensors, one at a time, in a single pass.

    A single method (rather than one generator per class) matters here: two
    separate generators over the same shard files would either read every
    shard twice or force buffering the whole stream (`itertools.tee`) to keep
    them in lock-step — both defeat the point of streaming a multi-gigabyte
    model tensor by tensor.
    """

    def tensors(self) -> typing.Iterator[SourceTensor]:
        ...
