"""Tests for the packer (US-003).

@spec:AC-006 @spec:AC-007 @spec:AC-008 @principle:P-003 @principle:P-010
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy
import pytest
import safetensors
import safetensors.torch
import torch

from lws.adapters.safetensors.packed_store import SafetensorsPackedStore
from lws.adapters.safetensors.weight_source import SafetensorsWeightSource
from lws.application.codec.pack_model import pack_model
from lws.domain.se12.container import (
    CURRENT_FORMAT_VERSION,
    FORMAT_NAME,
    FormatVersion,
    UnsupportedFormatVersionError,
    validate_format_version,
)
from lws.domain.se12.summary import compression_ratio, meets_compression_target
from lws.domain.weights import Bf16Weights
from tests.fakes import FakePackSource, FakePackedStore


def _concentrated_bf16(rows: int, cols: int, seed: int = 0, exponents: tuple[int, int] = (120, 135)) -> torch.Tensor:
    """Weights with a concentrated exponent distribution, like a trained model
    (ASM-001) — needed to reach the AC-006 ratio; uniform-random data would
    make most tiles fall back to raw storage instead."""
    rng = numpy.random.default_rng(seed)
    low, high = exponents
    n = rows * cols
    exponent_bytes = rng.choice(numpy.arange(low, high, dtype=numpy.uint8), size=n)
    sign = rng.integers(0, 2, size=n).astype(numpy.uint16)
    mantissa = rng.integers(0, 0x80, size=n).astype(numpy.uint16)
    bits = (sign << 15) | (exponent_bytes.astype(numpy.uint16) << 7) | mantissa
    pattern = bits.astype(numpy.uint16).view(numpy.int16).reshape(rows, cols)
    return torch.from_numpy(pattern.copy()).view(torch.bfloat16)


def _write_model(model_dir: pathlib.Path) -> dict[str, torch.Tensor]:
    tensors = {
        # Exact multiples of the default tile shape (16 x 256), like the real
        # target model's Linear shapes (weight-codec Q-001) — a shape that
        # forces padding is exercised directly in test_spec_se12_roundtrip.py;
        # mixing it in here would dilute the AC-006 ratio with padding
        # overhead the acceptance criterion is not about.
        "layer0.weight": _concentrated_bf16(64, 512, seed=1),
        "layer1.weight": _concentrated_bf16(32, 256, seed=2),
        "layer.bias": torch.randn(64, dtype=torch.bfloat16),  # BF16 but 1-D
        "norm.scale": torch.randn(64, dtype=torch.float32),  # not BF16
        "mask": torch.zeros(10, dtype=torch.bool),
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(tensors, str(model_dir / "model-00001-of-00001.safetensors"))
    return tensors


def _read_back_se12_tensor(shard_path: pathlib.Path, name: str):
    from lws.domain.se12.codebook import Codebook
    from lws.domain.se12.codec import EcBytes, EscapeHandling, EscBytes, FallbackBitmap, FallbackData, FallbackRaw, PrimaryStreams, Se12Geometry, Se12Metadata, Se12Streams, Se12Tensor, TilingParameters
    from lws.domain.se12.tile import EscapeBudget, MatrixShape, TileShape
    from lws.domain.weights import SignMantissaBytes

    with safetensors.safe_open(str(shard_path), framework="pt") as shard:
        file_metadata = shard.metadata()
        payload = json.loads(file_metadata["se12"])
        tensor_meta = payload["tensors"][name]
        sm = shard.get_tensor(f"{name}.se12.sm").numpy()
        ec = shard.get_tensor(f"{name}.se12.ec").numpy()
        esc = shard.get_tensor(f"{name}.se12.esc").numpy()
        fallback_bitmap = shard.get_tensor(f"{name}.se12.fallback_bitmap").numpy()
        fallback_raw = shard.get_tensor(f"{name}.se12.fallback_raw").numpy()
    codebook = Codebook(exponents=tuple(tensor_meta["codebook"]))
    shape = MatrixShape(rows=tensor_meta["shape"][0], cols=tensor_meta["shape"][1])
    tile_shape = TileShape(rows=tensor_meta["tile_rows"], cols=tensor_meta["tile_cols"])
    escape_budget = EscapeBudget(value=tensor_meta["escape_budget"])
    tiling = TilingParameters(tile_shape=tile_shape, escape_budget=escape_budget)
    geometry = Se12Geometry(shape=shape, tiling=tiling)
    metadata = Se12Metadata(codebook=codebook, geometry=geometry)
    primary = PrimaryStreams(sm=SignMantissaBytes(value=sm), ec=EcBytes(value=ec))
    fallback = FallbackData(bitmap=FallbackBitmap(value=fallback_bitmap), raw=FallbackRaw(value=fallback_raw))
    escape_handling = EscapeHandling(esc=EscBytes(value=esc), fallback=fallback)
    streams = Se12Streams(primary=primary, escape_handling=escape_handling)
    return Se12Tensor(streams=streams, metadata=metadata)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_packed_ratio_is_at_most_78_percent_of_original(tmp_path: pathlib.Path):
    """@spec:AC-006"""
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "packed"
    _write_model(model_dir)

    summary = pack_model(SafetensorsWeightSource(model_dir), SafetensorsPackedStore(out_dir))

    ratio = compression_ratio(summary.eligible_sizes)
    verdict = meets_compression_target(ratio)
    assert verdict.within_target, f"ratio {ratio.value} exceeds 0.78"
    counts = summary.tensor_counts
    assert counts.encoded.value == 2  # layer0.weight, layer1.weight
    assert counts.passed_through.value == 3  # bias, norm.scale, mask


def test_original_files_are_untouched_and_output_stays_in_out_dir(tmp_path: pathlib.Path):
    """@spec:AC-007 @principle:P-003"""
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "packed"
    _write_model(model_dir)
    original_path = model_dir / "model-00001-of-00001.safetensors"
    hash_before = _sha256(original_path)

    pack_model(SafetensorsWeightSource(model_dir), SafetensorsPackedStore(out_dir))

    hash_after = _sha256(original_path)
    assert hash_before == hash_after
    model_dir_entries = list(model_dir.iterdir())
    assert model_dir_entries == [original_path]  # nothing new written next to the original
    out_dir_entries = list(out_dir.iterdir())
    assert out_dir_entries == [out_dir / "model-00001-of-00001.safetensors"]


def test_ineligible_and_1d_bf16_tensors_are_copied_unchanged(tmp_path: pathlib.Path):
    """@spec:AC-007"""
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "packed"
    tensors = _write_model(model_dir)

    pack_model(SafetensorsWeightSource(model_dir), SafetensorsPackedStore(out_dir))

    packed_path = out_dir / "model-00001-of-00001.safetensors"
    with safetensors.safe_open(str(packed_path), framework="pt") as shard:
        for name in ("layer.bias", "norm.scale", "mask"):
            packed_tensor = shard.get_tensor(name)
            original_tensor = tensors[name]
            assert packed_tensor.dtype == original_tensor.dtype
            assert torch.equal(packed_tensor, original_tensor)


def test_packed_file_declares_format_and_per_tensor_metadata(tmp_path: pathlib.Path):
    """@spec:AC-008"""
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "packed"
    _write_model(model_dir)

    pack_model(SafetensorsWeightSource(model_dir), SafetensorsPackedStore(out_dir))

    packed_path = out_dir / "model-00001-of-00001.safetensors"
    with safetensors.safe_open(str(packed_path), framework="pt") as shard:
        metadata = shard.metadata()
    payload = json.loads(metadata["se12"])
    assert payload["format"] == FORMAT_NAME
    assert payload["format_version"] == CURRENT_FORMAT_VERSION
    tensor_meta = payload["tensors"]["layer0.weight"]
    assert tensor_meta["eligible"] is True
    assert tensor_meta["tile_rows"] == 16
    assert tensor_meta["tile_cols"] == 256
    assert len(tensor_meta["codebook"]) == 15
    validate_format_version(FormatVersion(payload["format_version"]))  # does not raise


def test_unknown_format_version_is_refused_with_found_and_supported():
    """@spec:AC-008"""
    with pytest.raises(UnsupportedFormatVersionError) as error:
        validate_format_version(FormatVersion(99))
    message = str(error.value)
    assert "99" in message
    assert str(CURRENT_FORMAT_VERSION) in message


def test_packing_is_deterministic_and_carries_no_gpu_specific_key(tmp_path: pathlib.Path):
    """@principle:P-010 — same input, same bytes; no device/arch/tuning key."""
    model_dir = tmp_path / "model"
    out_dir_a = tmp_path / "packed_a"
    out_dir_b = tmp_path / "packed_b"
    _write_model(model_dir)

    pack_model(SafetensorsWeightSource(model_dir), SafetensorsPackedStore(out_dir_a))
    pack_model(SafetensorsWeightSource(model_dir), SafetensorsPackedStore(out_dir_b))

    shard_name = "model-00001-of-00001.safetensors"
    hash_a = _sha256(out_dir_a / shard_name)
    hash_b = _sha256(out_dir_b / shard_name)
    assert hash_a == hash_b

    with safetensors.safe_open(str(out_dir_a / shard_name), framework="pt") as shard:
        metadata = shard.metadata()
    payload = json.dumps(metadata)
    forbidden = ("sm_6", "sm_7", "sm_8", "sm_9", "compute_", "gpu", "cuda", "P100", "T4", "H100", "A100", "RTX")
    for token in forbidden:
        assert token not in payload, f"packed metadata leaks a GPU-specific token: {token}"


def test_end_to_end_decode_of_the_written_file_is_bit_exact(tmp_path: pathlib.Path):
    """@principle:P-004 — the full pipeline (encode, write, read back, decode)
    reproduces the original bits, not just the in-memory Se12Tensor."""
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "packed"
    tensors = _write_model(model_dir)

    pack_model(SafetensorsWeightSource(model_dir), SafetensorsPackedStore(out_dir))

    from lws.domain.se12.codec import decode_tensor

    packed_path = out_dir / "model-00001-of-00001.safetensors"
    for name in ("layer0.weight", "layer1.weight"):
        se12_tensor = _read_back_se12_tensor(packed_path, name)
        decoded = decode_tensor(se12_tensor)
        original = tensors[name]
        original_flat = original.reshape(-1).view(torch.int16).numpy()
        assert numpy.array_equal(decoded.bit_pattern, original_flat), f"mismatch decoding {name} from the packed file"


def test_pack_model_use_case_with_fakes_needs_no_disk():
    """@spec:AC-006 — the use case's byte accounting works against fakes alone."""
    from lws.application.ports.codec.weight_source import EligibleTensor, IneligibleSourceTensor
    from lws.domain.weights import RawBytes, RawTensor, RawTensorPayload, ShapedWeights, TensorShape

    pattern = _concentrated_bf16(16, 256, seed=9).reshape(-1).view(torch.int16).numpy()
    weights = Bf16Weights(bit_pattern=pattern)
    shaped = ShapedWeights(weights=weights, shape=TensorShape(dims=(16, 256)))
    eligible = EligibleTensor(name="w", tensor=shaped)
    raw = RawTensor(dtype_name="torch.float32", payload=RawTensorPayload(bytes_value=RawBytes(value=numpy.zeros(4, dtype=numpy.uint8)), shape=TensorShape(dims=(1,))))
    ineligible = IneligibleSourceTensor(name="b", tensor=raw)

    source = FakePackSource([("shard-0.safetensors", [eligible, ineligible])])
    store = FakePackedStore()

    summary = pack_model(source, store)

    assert summary.tensor_counts.encoded.value == 1
    assert summary.tensor_counts.passed_through.value == 1
    assert len(store.written_shards) == 1
    assert store.written_shards[0].shard_name == "shard-0.safetensors"
