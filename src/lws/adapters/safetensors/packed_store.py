"""Adapter: PackedStore that writes SE12 shards as .safetensors files.

Per-tensor SE12 streams are written under `<name>.se12.<stream>` keys
(weight-codec design.md, "Container"); pass-through tensors are written
under their own original name, reconstructed to their exact original dtype
and shape so a plain safetensors reader sees a normal tensor. File-level
metadata is one JSON blob under the `"se12"` key (safetensors metadata is a
flat str -> str map, so a single key holds the whole per-tensor table).
"""

from __future__ import annotations

import json
import pathlib
import typing

import safetensors.torch
import torch

from lws.application.ports.codec.packed_store import PackedEntries, PackedShard, PassThroughTensor, Se12PackedTensor
from lws.domain.se12.container import CURRENT_FORMAT_VERSION, FORMAT_NAME
from lws.domain.weights import RawTensor

DTYPE_BY_NAME = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float64": torch.float64,
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.int64": torch.int64,
    "torch.int32": torch.int32,
    "torch.int16": torch.int16,
    "torch.int8": torch.int8,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
}


class SafetensorsPackedStore:
    """Writes one .safetensors file per shard into the output directory."""

    def __init__(self, output_directory: pathlib.Path) -> None:
        self.output_directory = output_directory

    def write_shard(self, shard: PackedShard) -> None:
        directory = self.output_directory
        directory.mkdir(parents=True, exist_ok=True)
        shard_name = shard.shard_name
        output_path = directory / shard_name
        tensors, tensor_metadata = _build_tensors(shard.entries)
        file_metadata = _file_metadata(tensor_metadata)
        safetensors.torch.save_file(tensors, str(output_path), metadata=file_metadata)


def _build_tensors(entries: PackedEntries) -> tuple[dict, dict]:
    items = entries.items
    tensors: dict = {}
    tensor_metadata: dict = {}
    for entry in items:
        _add_entry(entry, tensors, tensor_metadata)
    return tensors, tensor_metadata


def _add_entry(entry: typing.Any, tensors: dict, tensor_metadata: dict) -> None:
    if isinstance(entry, Se12PackedTensor):
        _add_se12_entry(entry, tensors, tensor_metadata)
        return
    _add_passthrough_entry(entry, tensors, tensor_metadata)


def _add_se12_entry(entry: Se12PackedTensor, tensors: dict, tensor_metadata: dict) -> None:
    name = entry.name
    encoded = entry.encoded
    streams = encoded.streams
    primary = streams.primary
    escape_handling = streams.escape_handling
    fallback = escape_handling.fallback
    tensors[f"{name}.se12.sm"] = _stream_tensor(primary.sm)
    tensors[f"{name}.se12.ec"] = _stream_tensor(primary.ec)
    tensors[f"{name}.se12.esc"] = _stream_tensor(escape_handling.esc)
    tensors[f"{name}.se12.fallback_bitmap"] = _stream_tensor(fallback.bitmap)
    tensors[f"{name}.se12.fallback_raw"] = _stream_tensor(fallback.raw)
    metadata = encoded.metadata
    tensor_metadata[name] = _se12_tensor_metadata(metadata)


def _stream_tensor(stream: typing.Any) -> torch.Tensor:
    array = stream.value
    return torch.from_numpy(array)


def _se12_tensor_metadata(metadata: typing.Any) -> dict:
    geometry = metadata.geometry
    shape = geometry.shape
    tiling = geometry.tiling
    tile_shape = tiling.tile_shape
    escape_budget = tiling.escape_budget
    codebook = metadata.codebook
    exponents = codebook.exponents
    return {
        "eligible": True,
        "shape": [shape.rows, shape.cols],
        "dtype": "torch.bfloat16",
        "tile_rows": tile_shape.rows,
        "tile_cols": tile_shape.cols,
        "escape_budget": escape_budget.value,
        "codebook": list(exponents),
    }


def _add_passthrough_entry(entry: PassThroughTensor, tensors: dict, tensor_metadata: dict) -> None:
    name = entry.name
    raw = entry.raw
    tensors[name] = _tensor_from_raw(raw)
    tensor_metadata[name] = _passthrough_tensor_metadata(raw)


def _tensor_from_raw(raw_tensor: RawTensor) -> torch.Tensor:
    dtype_name = raw_tensor.dtype_name
    payload = raw_tensor.payload
    bytes_value = payload.bytes_value
    array = bytes_value.value
    shape = payload.shape
    dims = shape.dims
    torch_dtype = DTYPE_BY_NAME[dtype_name]
    byte_tensor = torch.from_numpy(array)
    typed_view = byte_tensor.view(torch_dtype)
    reshaped = typed_view.reshape(dims)
    return reshaped.contiguous()


def _passthrough_tensor_metadata(raw: RawTensor) -> dict:
    dtype_name = raw.dtype_name
    payload = raw.payload
    shape = payload.shape
    dims = shape.dims
    return {"eligible": False, "shape": list(dims), "dtype": dtype_name}


def _file_metadata(tensor_metadata: dict) -> dict:
    payload = {"format": FORMAT_NAME, "format_version": CURRENT_FORMAT_VERSION, "tensors": tensor_metadata}
    return {"se12": json.dumps(payload)}
