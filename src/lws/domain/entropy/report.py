"""BF16 anatomy (constitution weight-codec design.md):

    bit 15   14 ........ 7   6 ........ 0
     sign     exponent(8)     mantissa(7)

This module measures, for one tensor's bit pattern, how many bits the
exponent byte and the sign+mantissa byte really carry (Shannon entropy), the
full 16-bit entropy, and how much of the tensor the 15 most frequent
exponents cover — the numbers AC-001 asks the analyzer to report, and the
evidence ASM-001 (weight-codec spec.md) is checked against.
"""

from __future__ import annotations

import dataclasses

import numpy

from lws.domain.weights import Bf16Weights, ElementCount

TOP_EXPONENTS_TRACKED = 15
EXPONENT_ALPHABET_SIZE = 256
SIGN_MANTISSA_ALPHABET_SIZE = 256
FULL_ALPHABET_SIZE = 65536


@dataclasses.dataclass(frozen=True)
class Bits:
    """A Shannon entropy measurement, in bits."""

    value: float

    def __post_init__(self) -> None:
        value = self.value
        if value < 0:
            raise ValueError("Bits cannot be negative")


@dataclasses.dataclass(frozen=True)
class Share:
    """A fraction of a whole, between 0 and 1."""

    value: float

    def __post_init__(self) -> None:
        value = self.value
        if not 0.0 <= value <= 1.0:
            raise ValueError("Share must be between 0 and 1")


@dataclasses.dataclass(frozen=True)
class ExponentStatistics:
    entropy: Bits
    top_coverage: Share


@dataclasses.dataclass(frozen=True)
class ByteEntropyPair:
    sign_mantissa_entropy: Bits
    full_entropy: Bits


@dataclasses.dataclass(frozen=True)
class EntropyStatistics:
    exponent: ExponentStatistics
    byte_pair: ByteEntropyPair


@dataclasses.dataclass(frozen=True)
class TensorEntropyReport:
    element_count: ElementCount
    statistics: EntropyStatistics


def measure_tensor(weights: Bf16Weights) -> TensorEntropyReport:
    """The entropy report of one BF16 tensor (AC-001's per-tensor row)."""
    element_count = weights.element_count()
    exponent_counts, sign_mantissa_counts, full_counts = _split_histograms(weights)
    exponent = ExponentStatistics(
        entropy=Bits(_entropy_bits(exponent_counts, element_count)),
        top_coverage=Share(_top_coverage(exponent_counts, element_count, TOP_EXPONENTS_TRACKED)),
    )
    byte_pair = ByteEntropyPair(
        sign_mantissa_entropy=Bits(_entropy_bits(sign_mantissa_counts, element_count)),
        full_entropy=Bits(_entropy_bits(full_counts, element_count)),
    )
    return TensorEntropyReport(element_count=element_count, statistics=EntropyStatistics(exponent=exponent, byte_pair=byte_pair))


def _split_histograms(weights: Bf16Weights) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    pattern = weights.bit_pattern
    unsigned = pattern.astype(numpy.uint16)
    exponent_byte = ((unsigned >> 7) & 0xFF).astype(numpy.uint8)
    sign_mantissa_byte = (((unsigned >> 8) & 0x80) | (unsigned & 0x7F)).astype(numpy.uint8)
    exponent_counts = numpy.bincount(exponent_byte, minlength=EXPONENT_ALPHABET_SIZE)
    sign_mantissa_counts = numpy.bincount(sign_mantissa_byte, minlength=SIGN_MANTISSA_ALPHABET_SIZE)
    full_counts = numpy.bincount(unsigned, minlength=FULL_ALPHABET_SIZE)
    return exponent_counts, sign_mantissa_counts, full_counts


def _entropy_bits(counts: numpy.ndarray, element_count: ElementCount) -> float:
    total = element_count.value
    nonzero_counts = counts[counts > 0]
    probabilities = nonzero_counts / total
    return float(-(probabilities * numpy.log2(probabilities)).sum())


def _top_coverage(counts: numpy.ndarray, element_count: ElementCount, top: int) -> float:
    total = element_count.value
    sorted_counts = numpy.sort(counts)[::-1]
    covered = int(sorted_counts[:top].sum())
    return covered / total


@dataclasses.dataclass(frozen=True)
class NamedTensorEntropyReport:
    name: str
    report: TensorEntropyReport


@dataclasses.dataclass(frozen=True)
class PerTensorEntropyReports:
    """First-class collection: one entropy report per eligible tensor."""

    items: tuple[NamedTensorEntropyReport, ...]


@dataclasses.dataclass(frozen=True)
class IneligibleTensor:
    name: str
    dtype: str


@dataclasses.dataclass(frozen=True)
class IneligibleTensorReports:
    """First-class collection: tensors skipped because they are not BF16 (AC-002)."""

    items: tuple[IneligibleTensor, ...]


@dataclasses.dataclass(frozen=True)
class EntropySurveySummary:
    weighted_average: EntropyStatistics
    ineligible: IneligibleTensorReports


@dataclasses.dataclass(frozen=True)
class ModelEntropyReport:
    per_tensor: PerTensorEntropyReports
    summary: EntropySurveySummary


def summarize(per_tensor: PerTensorEntropyReports, ineligible: IneligibleTensorReports) -> ModelEntropyReport:
    """Element-weighted, model-wide average of every per-tensor statistic (AC-001)."""
    weighted_average = _weighted_average(per_tensor)
    summary = EntropySurveySummary(weighted_average=weighted_average, ineligible=ineligible)
    return ModelEntropyReport(per_tensor=per_tensor, summary=summary)


def _weighted_average(per_tensor: PerTensorEntropyReports) -> EntropyStatistics:
    named_reports = per_tensor.items
    weights = numpy.array([_element_weight(named) for named in named_reports], dtype=numpy.float64)
    weight_total = weights.sum()
    total_weight = float(weight_total) if weights.size else 1.0
    exponent_entropy = _weighted_mean([_exponent_entropy(named) for named in named_reports], weights, total_weight)
    top_coverage = _weighted_mean([_top_coverage_value(named) for named in named_reports], weights, total_weight)
    sign_mantissa_entropy = _weighted_mean([_sign_mantissa_entropy(named) for named in named_reports], weights, total_weight)
    full_entropy = _weighted_mean([_full_entropy(named) for named in named_reports], weights, total_weight)
    exponent = ExponentStatistics(entropy=Bits(exponent_entropy), top_coverage=Share(top_coverage))
    byte_pair = ByteEntropyPair(sign_mantissa_entropy=Bits(sign_mantissa_entropy), full_entropy=Bits(full_entropy))
    return EntropyStatistics(exponent=exponent, byte_pair=byte_pair)


def _element_weight(named: NamedTensorEntropyReport) -> float:
    report = named.report
    element_count = report.element_count
    return float(element_count.value)


def _exponent_entropy(named: NamedTensorEntropyReport) -> float:
    report = named.report
    statistics = report.statistics
    exponent = statistics.exponent
    entropy = exponent.entropy
    return entropy.value


def _top_coverage_value(named: NamedTensorEntropyReport) -> float:
    report = named.report
    statistics = report.statistics
    exponent = statistics.exponent
    coverage = exponent.top_coverage
    return coverage.value


def _sign_mantissa_entropy(named: NamedTensorEntropyReport) -> float:
    report = named.report
    statistics = report.statistics
    byte_pair = statistics.byte_pair
    entropy = byte_pair.sign_mantissa_entropy
    return entropy.value


def _full_entropy(named: NamedTensorEntropyReport) -> float:
    report = named.report
    statistics = report.statistics
    byte_pair = statistics.byte_pair
    entropy = byte_pair.full_entropy
    return entropy.value


def _weighted_mean(values: list[float], weights: numpy.ndarray, total_weight: float) -> float:
    if not values:
        return 0.0
    values_array = numpy.array(values, dtype=numpy.float64)
    return float((values_array * weights).sum() / total_weight)
