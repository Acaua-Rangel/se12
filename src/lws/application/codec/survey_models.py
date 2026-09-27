"""Use case: entropy survey across several models (US-001, AC-034).

Each model's tensors are streamed lazily straight from its WeightSource into
the domain's survey_model — never pre-built into a list — so only one
tensor's raw weights are ever held in memory at a time, whatever the model's
total size (AC-034: this must run with 16 GB of RAM).
"""

from __future__ import annotations

import dataclasses
import typing

from lws.application.ports.codec.weight_source import EligibleTensor, WeightSource
from lws.domain.entropy.survey import ModelSurveyReport, SurveyInput, survey_model


@dataclasses.dataclass(frozen=True)
class ModelIdentity:
    label: str
    revision: str | None


class NamedModelSource(typing.NamedTuple):
    identity: ModelIdentity
    source: WeightSource


@dataclasses.dataclass(frozen=True)
class ModelSurveyEntry:
    identity: ModelIdentity
    report: ModelSurveyReport


@dataclasses.dataclass(frozen=True)
class MultiModelSurveyReport:
    """First-class collection: one survey entry per model given to survey_models."""

    entries: tuple[ModelSurveyEntry, ...]


def survey_models(sources: typing.Iterable[NamedModelSource]) -> MultiModelSurveyReport:
    entries = tuple(_survey_one(named_source) for named_source in sources)
    return MultiModelSurveyReport(entries=entries)


def _survey_one(named_source: NamedModelSource) -> ModelSurveyEntry:
    identity = named_source.identity
    source = named_source.source
    items = (_as_survey_input(item) for item in source.tensors() if isinstance(item, EligibleTensor))
    report = survey_model(items)
    return ModelSurveyEntry(identity=identity, report=report)


def _as_survey_input(item: EligibleTensor) -> SurveyInput:
    name = item.name
    shaped = item.tensor
    return SurveyInput(name=name, shaped=shaped)
