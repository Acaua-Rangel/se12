"""Use case: entropy report of a whole model (US-001, AC-001, AC-002)."""

from __future__ import annotations

from lws.application.ports.codec.weight_source import EligibleTensor, IneligibleSourceTensor, SourceTensor, WeightSource
from lws.domain.entropy.report import (
    IneligibleTensor,
    IneligibleTensorReports,
    ModelEntropyReport,
    NamedTensorEntropyReport,
    PerTensorEntropyReports,
    measure_tensor,
    summarize,
)


def analyze_model(source: WeightSource) -> ModelEntropyReport:
    """Streams every tensor out of `source`, once, and returns the entropy report.

    Eligible tensors are reduced to a small report immediately, so at most one
    tensor's raw weights are ever held in memory at a time (see weight-codec
    ASM-015, written for the packer but just as true here for large models).
    """
    eligible_reports: list[NamedTensorEntropyReport] = []
    ineligible_reports: list[IneligibleTensor] = []
    for item in source.tensors():
        _accumulate(item, eligible_reports, ineligible_reports)
    per_tensor = PerTensorEntropyReports(items=tuple(eligible_reports))
    ineligible = IneligibleTensorReports(items=tuple(ineligible_reports))
    return summarize(per_tensor, ineligible)


def _accumulate(
    item: SourceTensor,
    eligible_reports: list[NamedTensorEntropyReport],
    ineligible_reports: list[IneligibleTensor],
) -> None:
    if isinstance(item, EligibleTensor):
        eligible_reports.append(_measure_named(item))
        return
    ineligible_reports.append(_as_ineligible(item))


def _measure_named(item: EligibleTensor) -> NamedTensorEntropyReport:
    name = item.name
    weights = item.weights
    report = measure_tensor(weights)
    return NamedTensorEntropyReport(name=name, report=report)


def _as_ineligible(item: IneligibleSourceTensor) -> IneligibleTensor:
    name = item.name
    dtype = item.dtype
    return IneligibleTensor(name=name, dtype=dtype)
