"""Adapter: WeightSource over one or more .safetensors shards on disk.

Uses the safetensors "pt" (torch) framework, not "np": numpy has no BF16
dtype, so reading a BF16 tensor needs torch to view its exact 16-bit pattern
(`tensor.view(torch.int16)`) before handing the domain a plain numpy array.
Only whichever ONE tensor is being read is ever materialized — safe_open()
memory-maps the shard and get_tensor() reads a single tensor from it.
"""

from __future__ import annotations

import pathlib
import typing

import safetensors
import torch

from lws.application.ports.codec.pack_source import ShardOfTensors
from lws.application.ports.codec.weight_source import EligibleTensor, IneligibleSourceTensor, SourceTensor
from lws.domain.weights import Bf16Weights, RawBytes, RawTensor, RawTensorPayload, ShapedWeights, TensorShape

SHARD_GLOB = "*.safetensors"


class SafetensorsWeightSource:
    """Streams every tensor of every shard in a model directory.

    Implements two ports from the same underlying read logic: `tensors()`
    (WeightSource, one flat pass — the entropy analyzer) and `shards()`
    (PackSource, grouped by shard — the packer, which must mirror shard
    boundaries in its output, weight-codec design.md "Container").
    """

    def __init__(self, model_directory: pathlib.Path) -> None:
        self.model_directory = model_directory

    def tensors(self) -> typing.Iterator[SourceTensor]:
        for shard_path in _shard_paths(self.model_directory):
            yield from _read_shard(shard_path)

    def shards(self) -> typing.Iterator[ShardOfTensors]:
        for shard_path in _shard_paths(self.model_directory):
            yield ShardOfTensors(shard_name=shard_path.name, tensors=_read_shard(shard_path))


def _shard_paths(model_directory: pathlib.Path) -> list[pathlib.Path]:
    return sorted(model_directory.glob(SHARD_GLOB))


def _read_shard(shard_path: pathlib.Path) -> typing.Iterator[SourceTensor]:
    # A single statement inside the `with` (rather than a nested `for`) keeps
    # this function at one level of indentation (calisthenics rule 1); the
    # loop itself lives in _read_all.
    with safetensors.safe_open(str(shard_path), framework="pt") as shard:
        yield from _read_all(shard)


def _read_all(shard: typing.Any) -> typing.Iterator[SourceTensor]:
    keys = shard.keys()
    for name in keys:
        yield _read_one(shard, name)


def _read_one(shard: typing.Any, name: str) -> SourceTensor:
    tensor = shard.get_tensor(name)
    dtype = tensor.dtype
    if dtype == torch.bfloat16:
        return _as_eligible(name, tensor)
    return _as_ineligible(name, tensor)


def _as_eligible(name: str, tensor: torch.Tensor) -> EligibleTensor:
    shape = TensorShape(dims=tuple(tensor.shape))
    flattened = tensor.reshape(-1)
    bit_view = flattened.view(torch.int16)
    array = bit_view.numpy().copy()
    weights = Bf16Weights(bit_pattern=array)
    shaped = ShapedWeights(weights=weights, shape=shape)
    return EligibleTensor(name=name, tensor=shaped)


def _as_ineligible(name: str, tensor: torch.Tensor) -> IneligibleSourceTensor:
    shape = TensorShape(dims=tuple(tensor.shape))
    dtype_name = str(tensor.dtype)
    flattened = tensor.reshape(-1)
    contiguous = flattened.contiguous()
    byte_view = contiguous.view(torch.uint8)
    raw_array = byte_view.numpy().copy()
    raw_bytes = RawBytes(value=raw_array)
    payload = RawTensorPayload(bytes_value=raw_bytes, shape=shape)
    raw_tensor = RawTensor(dtype_name=dtype_name, payload=payload)
    return IneligibleSourceTensor(name=name, tensor=raw_tensor)
