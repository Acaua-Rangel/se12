"""Entropy survey across model families (AC-034): reuses the entropy domain
(measure_tensor) for the information-theoretic numbers and the reference
codec (encode_tensor, T-002) for the REAL SE12 bits/weight and fallback
share — not design.md's back-of-envelope 12.03-bit estimate.

Tensor role classification lives here, in one place, kept deliberately
architecture-agnostic (T-017 notes call for "per architecture", but Gemma 2,
Llama, Qwen, Mistral and OLMoE all follow the same Hugging Face `transformers`
naming conventions for a GPT-style decoder, so one pattern table covers all
five target families without special-casing any of them).
"""

from __future__ import annotations

import dataclasses
import enum
import typing

from lws.domain.entropy.report import Bits, ByteEntropyPair, EntropyStatistics, ExponentStatistics, Share, measure_tensor
from lws.domain.se12.codec import encode_tensor, fallback_tile_count
from lws.domain.se12.summary import se12_tensor_byte_size
from lws.domain.se12.tile import MatrixShape, default_tile_shape
from lws.domain.weights import Bf16Weights, ElementCount, ShapedWeights, TensorShape


class TensorRole(enum.Enum):
    EMBEDDING = "embedding"
    ATTENTION = "attention"
    MLP_OR_EXPERT = "mlp_or_expert"
    OTHER = "other"


ALL_ROLES = (TensorRole.EMBEDDING, TensorRole.ATTENTION, TensorRole.MLP_OR_EXPERT, TensorRole.OTHER)

_ROLE_PATTERNS = (
    ("embed_tokens", TensorRole.EMBEDDING),
    ("lm_head", TensorRole.EMBEDDING),
    ("wte", TensorRole.EMBEDDING),
    ("self_attn", TensorRole.ATTENTION),
    ("attention", TensorRole.ATTENTION),
    ("q_proj", TensorRole.ATTENTION),
    ("k_proj", TensorRole.ATTENTION),
    ("v_proj", TensorRole.ATTENTION),
    ("o_proj", TensorRole.ATTENTION),
    ("experts", TensorRole.MLP_OR_EXPERT),
    ("mlp", TensorRole.MLP_OR_EXPERT),
    ("feed_forward", TensorRole.MLP_OR_EXPERT),
    ("gate_proj", TensorRole.MLP_OR_EXPERT),
    ("up_proj", TensorRole.MLP_OR_EXPERT),
    ("down_proj", TensorRole.MLP_OR_EXPERT),
)


def _classify_tensor_role(name: str) -> TensorRole:
    matched = next((role for pattern, role in _ROLE_PATTERNS if pattern in name), None)
    if matched is None:
        return TensorRole.OTHER
    return matched


class SurveyInput(typing.NamedTuple):
    """One tensor to survey: its name (for role classification) and its
    weights+shape (ShapedWeights already composes those two, T-001)."""

    name: str
    shaped: ShapedWeights


@dataclasses.dataclass(frozen=True)
class TileCounts:
    total: ElementCount
    fallback: ElementCount


@dataclasses.dataclass(frozen=True)
class Se12TensorProjection:
    packed_bits: ElementCount
    tiles: TileCounts


@dataclasses.dataclass(frozen=True)
class TensorMeasurement:
    entropy: EntropyStatistics
    element_count: ElementCount


@dataclasses.dataclass(frozen=True)
class CombinedMeasurement:
    base: TensorMeasurement
    se12: Se12TensorProjection | None  # None when the tensor is not 2-D (not SE12-eligible)


@dataclasses.dataclass(frozen=True)
class TensorSurveyResult:
    role: TensorRole
    measurement: CombinedMeasurement


@dataclasses.dataclass(frozen=True)
class Se12AggregateProjection:
    bits_per_weight: Bits
    fallback_share: Share


@dataclasses.dataclass(frozen=True)
class SurveyStatistics:
    """The 5 numbers AC-034 asks for, for one model or one role within it."""

    entropy: EntropyStatistics
    se12: Se12AggregateProjection


@dataclasses.dataclass(frozen=True)
class RoleSurveyEntry:
    role: TensorRole
    statistics: SurveyStatistics


@dataclasses.dataclass(frozen=True)
class RoleSurveyBreakdown:
    """First-class collection: one SurveyStatistics per tensor role present."""

    items: tuple[RoleSurveyEntry, ...]


@dataclasses.dataclass(frozen=True)
class ModelSurveyReport:
    overall: SurveyStatistics
    by_role: RoleSurveyBreakdown


def measure_tensor_for_survey(item: SurveyInput) -> TensorSurveyResult:
    name = item.name
    shaped = item.shaped
    role = _classify_tensor_role(name)
    weights = shaped.weights
    report = measure_tensor(weights)
    element_count = report.element_count
    statistics = report.statistics
    base = TensorMeasurement(entropy=statistics, element_count=element_count)
    shape = shaped.shape
    se12 = _project_se12(weights, shape)
    combined = CombinedMeasurement(base=base, se12=se12)
    return TensorSurveyResult(role=role, measurement=combined)


def _project_se12(weights: Bf16Weights, shape: TensorShape) -> Se12TensorProjection | None:
    dims = shape.dims
    if len(dims) != 2:
        return None
    matrix_shape = MatrixShape(rows=dims[0], cols=dims[1])
    encoded = encode_tensor(weights, matrix_shape)
    packed_bytes = se12_tensor_byte_size(encoded)
    packed_bits = ElementCount(packed_bytes.value * 8)
    total_tiles = _tile_count(matrix_shape)
    fallback_tiles = fallback_tile_count(encoded)
    tiles = TileCounts(total=ElementCount(total_tiles), fallback=fallback_tiles)
    return Se12TensorProjection(packed_bits=packed_bits, tiles=tiles)


def _tile_count(matrix_shape: MatrixShape) -> int:
    tile_shape = default_tile_shape()
    tile_rows = tile_shape.rows
    tile_cols = tile_shape.cols
    rows = matrix_shape.rows
    cols = matrix_shape.cols
    tile_row_count = -(-rows // tile_rows)
    tile_col_count = -(-cols // tile_cols)
    return tile_row_count * tile_col_count


def survey_model(items: typing.Iterable[SurveyInput]) -> ModelSurveyReport:
    """`items` must be consumed lazily (a generator, not a pre-built list): a
    real model's tensors are streamed one at a time, each reduced to a small
    TensorSurveyResult before the next one is read, so only one tensor's raw
    weights are ever in memory (AC-034: this must run with 16 GB of RAM)."""
    results = tuple(measure_tensor_for_survey(item) for item in items)
    overall = _aggregate(results)
    by_role = tuple(_role_entry(results, role) for role in ALL_ROLES)
    present = tuple(entry for entry in by_role if entry is not None)
    return ModelSurveyReport(overall=overall, by_role=RoleSurveyBreakdown(items=present))


def _role_entry(results: tuple[TensorSurveyResult, ...], role: TensorRole) -> RoleSurveyEntry | None:
    matching = tuple(result for result in results if result.role is role)
    if not matching:
        return None
    return RoleSurveyEntry(role=role, statistics=_aggregate(matching))


def _aggregate(results: tuple[TensorSurveyResult, ...]) -> SurveyStatistics:
    total_elements = sum(_element_count(result) for result in results)
    exponent_entropy = _weighted_average(results, total_elements, _exponent_entropy)
    top_coverage = _weighted_average(results, total_elements, _top_coverage)
    sign_mantissa_entropy = _weighted_average(results, total_elements, _sign_mantissa_entropy)
    full_entropy = _weighted_average(results, total_elements, _full_entropy)
    entropy_statistics = _build_entropy_statistics(exponent_entropy, top_coverage, sign_mantissa_entropy, full_entropy)
    se12_projection = _aggregate_se12(results)
    return SurveyStatistics(entropy=entropy_statistics, se12=se12_projection)


def _weighted_average(results: tuple[TensorSurveyResult, ...], total_elements: int, extract) -> float:
    if total_elements == 0:
        return 0.0
    weighted_total = sum(extract(result) * _element_count(result) for result in results)
    return weighted_total / total_elements


def _build_entropy_statistics(exponent_entropy: float, top_coverage: float, sign_mantissa_entropy: float, full_entropy: float) -> EntropyStatistics:
    exponent = ExponentStatistics(entropy=Bits(exponent_entropy), top_coverage=Share(top_coverage))
    byte_pair = ByteEntropyPair(sign_mantissa_entropy=Bits(sign_mantissa_entropy), full_entropy=Bits(full_entropy))
    return EntropyStatistics(exponent=exponent, byte_pair=byte_pair)


def _aggregate_se12(results: tuple[TensorSurveyResult, ...]) -> Se12AggregateProjection:
    eligible = tuple(result for result in results if _se12_of(result) is not None)
    total_bits = sum(_packed_bits(result) for result in eligible)
    total_se12_elements = sum(_element_count(result) for result in eligible)
    total_tiles = sum(_total_tiles(result) for result in eligible)
    total_fallback = sum(_fallback_tiles(result) for result in eligible)
    bits_per_weight = total_bits / total_se12_elements if total_se12_elements else 0.0
    fallback_share = total_fallback / total_tiles if total_tiles else 0.0
    return Se12AggregateProjection(bits_per_weight=Bits(bits_per_weight), fallback_share=Share(fallback_share))


def _element_count(result: TensorSurveyResult) -> int:
    measurement = result.measurement
    base = measurement.base
    element_count = base.element_count
    return element_count.value


def _exponent_entropy(result: TensorSurveyResult) -> float:
    measurement = result.measurement
    base = measurement.base
    entropy = base.entropy
    exponent = entropy.exponent
    entropy_value = exponent.entropy
    return entropy_value.value


def _top_coverage(result: TensorSurveyResult) -> float:
    measurement = result.measurement
    base = measurement.base
    entropy = base.entropy
    exponent = entropy.exponent
    coverage = exponent.top_coverage
    return coverage.value


def _sign_mantissa_entropy(result: TensorSurveyResult) -> float:
    measurement = result.measurement
    base = measurement.base
    entropy = base.entropy
    byte_pair = entropy.byte_pair
    entropy_value = byte_pair.sign_mantissa_entropy
    return entropy_value.value


def _full_entropy(result: TensorSurveyResult) -> float:
    measurement = result.measurement
    base = measurement.base
    entropy = base.entropy
    byte_pair = entropy.byte_pair
    entropy_value = byte_pair.full_entropy
    return entropy_value.value


def _se12_of(result: TensorSurveyResult) -> Se12TensorProjection | None:
    measurement = result.measurement
    return measurement.se12


def _packed_bits(result: TensorSurveyResult) -> int:
    se12 = _se12_of(result)
    packed_bits = se12.packed_bits
    return packed_bits.value


def _total_tiles(result: TensorSurveyResult) -> int:
    se12 = _se12_of(result)
    tiles = se12.tiles
    total = tiles.total
    return total.value


def _fallback_tiles(result: TensorSurveyResult) -> int:
    se12 = _se12_of(result)
    tiles = se12.tiles
    fallback = tiles.fallback
    return fallback.value
