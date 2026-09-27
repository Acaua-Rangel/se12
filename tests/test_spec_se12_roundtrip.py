"""Tests for the SE12 reference codec (US-002).

@spec:AC-003 @spec:AC-004 @spec:AC-005 @principle:P-004
"""

from __future__ import annotations

import os

import numpy
import pytest
import safetensors.torch
import torch

from lws.domain.se12.codec import decode_tensor, encode_tensor, fallback_tile_count
from lws.domain.se12.tile import MatrixShape
from lws.domain.weights import Bf16Weights


def _bits(sign: int, exponent: int, mantissa: int) -> int:
    return (sign << 15) | (exponent << 7) | mantissa


def _pattern(values: list[int]) -> numpy.ndarray:
    return numpy.array(values, dtype=numpy.uint16).astype(numpy.int16)


def _round_trip(pattern: numpy.ndarray, rows: int, cols: int):
    weights = Bf16Weights(bit_pattern=pattern)
    shape = MatrixShape(rows=rows, cols=cols)
    encoded = encode_tensor(weights, shape)
    decoded = decode_tensor(encoded)
    return encoded, decoded


def test_round_trip_is_bit_exact_on_float_edge_cases():
    """@spec:AC-003 @principle:P-004

    +0, -0, subnormals, +-inf, NaNs with different payloads, the largest and
    smallest exponents, and a run of random normal values.
    """
    edge_cases = [
        _bits(0, 0, 0),  # +0
        _bits(1, 0, 0),  # -0
        _bits(0, 0, 1),  # smallest positive subnormal
        _bits(1, 0, 1),  # smallest negative subnormal
        _bits(0, 0, 0x7F),  # largest subnormal
        _bits(0, 0xFF, 0),  # +inf
        _bits(1, 0xFF, 0),  # -inf
        _bits(0, 0xFF, 0x40),  # NaN, payload A
        _bits(0, 0xFF, 0x7F),  # NaN, payload B (different payload)
        _bits(1, 0xFF, 0x01),  # -NaN, payload C
        _bits(0, 1, 0),  # smallest positive normal
        _bits(1, 1, 0),  # smallest negative normal
        _bits(0, 0xFE, 0x7F),  # largest finite normal
        _bits(1, 0xFE, 0x7F),  # largest-magnitude negative finite normal
    ]
    rng = numpy.random.default_rng(7)
    random_normals = [_bits(int(rng.integers(0, 2)), int(rng.integers(1, 0xFE)), int(rng.integers(0, 0x80))) for _ in range(200)]
    values = edge_cases + random_normals
    pattern = _pattern(values)
    encoded, decoded = _round_trip(pattern, rows=1, cols=len(values))
    assert numpy.array_equal(decoded.bit_pattern, pattern)
    assert decoded.bit_pattern.dtype == numpy.int16


def test_round_trip_is_bit_exact_across_shapes_with_and_without_padding():
    """@spec:AC-003 @principle:P-004"""
    rng = numpy.random.default_rng(11)
    shapes = [(16, 256), (32, 256), (16, 512), (32, 512), (33, 300), (1, 1), (36, 1), (160, 2560)]
    for rows, cols in shapes:
        bits = rng.integers(0, 65536, size=rows * cols, dtype=numpy.uint32).astype(numpy.uint16)
        pattern = bits.astype(numpy.int16)
        _encoded, decoded = _round_trip(pattern, rows, cols)
        assert numpy.array_equal(decoded.bit_pattern, pattern), f"mismatch at shape ({rows}, {cols})"


def test_fallback_tile_is_raw_and_still_bit_exact():
    """@spec:AC-005

    One tile is built so it holds more rare exponents than the escape budget
    (default 16): 15 dominant exponents fill most of the tile, plus 20
    distinct single-occurrence exponents that cannot all fit the codebook or
    the escape budget.
    """
    rng = numpy.random.default_rng(3)
    tile_weights = 16 * 256
    escape_overflow = 20
    exponents = numpy.zeros(tile_weights, dtype=numpy.uint8)
    dominant = numpy.arange(15, dtype=numpy.uint8)
    dominant_fill = tile_weights - escape_overflow
    exponents[:dominant_fill] = numpy.tile(dominant, dominant_fill // 15 + 1)[:dominant_fill]
    exponents[dominant_fill:] = numpy.arange(100, 100 + escape_overflow, dtype=numpy.uint8)
    sign = rng.integers(0, 2, size=tile_weights).astype(numpy.uint16)
    mantissa = rng.integers(0, 0x80, size=tile_weights).astype(numpy.uint16)
    bits = (sign << 15) | (exponents.astype(numpy.uint16) << 7) | mantissa
    pattern = bits.astype(numpy.int16)

    encoded, decoded = _round_trip(pattern, rows=16, cols=256)

    assert numpy.array_equal(decoded.bit_pattern, pattern)
    assert fallback_tile_count(encoded).value == 1


def test_fallback_tile_count_is_zero_when_every_tile_fits_its_budget():
    """@spec:AC-005 — the encoder's fallback-tile count is itself checkable."""
    rng = numpy.random.default_rng(5)
    rows, cols = 160, 2560  # 10 x 10 tiles
    n = rows * cols
    exponents = rng.choice(numpy.arange(120, 135, dtype=numpy.uint8), size=n)
    sign = rng.integers(0, 2, size=n).astype(numpy.uint16)
    mantissa = rng.integers(0, 0x80, size=n).astype(numpy.uint16)
    bits = (sign << 15) | (exponents.astype(numpy.uint16) << 7) | mantissa
    pattern = bits.astype(numpy.int16)

    encoded, decoded = _round_trip(pattern, rows, cols)

    assert numpy.array_equal(decoded.bit_pattern, pattern)
    assert fallback_tile_count(encoded).value == 0


@pytest.mark.model
def test_round_trip_every_tensor_of_the_real_model():
    """@spec:AC-004 @principle:P-004 — skips (not proof) without LWS_MODEL_DIR."""
    import pathlib

    model_dir = os.environ.get("LWS_MODEL_DIR")
    if not model_dir:
        pytest.skip("LWS_MODEL_DIR not set")
    checked_tensors = 0
    checked_weights = 0
    for shard_path in sorted(pathlib.Path(model_dir).glob("*.safetensors")):
        with safetensors.safe_open(str(shard_path), framework="pt") as shard:
            for name in shard.keys():
                tensor = shard.get_tensor(name)
                if tensor.dtype != torch.bfloat16:
                    continue
                if tensor.ndim != 2:
                    continue
                rows, cols = tensor.shape
                flat = tensor.reshape(-1).view(torch.int16).numpy().copy()
                weights = Bf16Weights(bit_pattern=flat)
                shape = MatrixShape(rows=rows, cols=cols)
                encoded = encode_tensor(weights, shape)
                decoded = decode_tensor(encoded)
                assert numpy.array_equal(decoded.bit_pattern, flat), f"mismatch in tensor {name}"
                checked_tensors += 1
                checked_weights += flat.size
    print(f"AC-004: {checked_tensors} tensors, {checked_weights} weights checked bit-exact")
    assert checked_tensors > 0
