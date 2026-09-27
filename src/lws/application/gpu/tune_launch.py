"""Use case: tune launch configurations for the raw and fused kernels, per
(shape, batch), and save the result (US-013, AC-024).

Fully testable with fake evaluators (tests/fakes.py + tests/test_spec_tune.py):
this file never touches a GPU itself — it only orchestrates the domain's
search-space rules over whatever `CandidateEvaluator` the caller supplies.
"""

from __future__ import annotations

import dataclasses
import typing

from lws.application.ports.gpu.candidate_evaluator import CandidateEvaluator
from lws.application.ports.gpu.tuning_cache import (
    BatchSize,
    ConfigurationSet,
    KernelConfigurations,
    ShapeKey,
    TunedEntries,
    TunedEntry,
    TuningCache,
    TuningFile,
    WeightShape,
)
from lws.domain.device.properties import GpuSlug
from lws.domain.launch.heuristic import GemvShape, LaunchConfiguration
from lws.domain.launch.search_space import TimedCandidate, best_configuration, candidate_configurations, order_matched_configuration


@dataclasses.dataclass(frozen=True)
class TuneRequest:
    weight_shape: WeightShape
    batch: BatchSize


@dataclasses.dataclass(frozen=True)
class EvaluatorPair:
    raw: CandidateEvaluator
    fused: CandidateEvaluator


def tune_launch(gpu_slug: GpuSlug, requests: typing.Sequence[TuneRequest], evaluators: typing.Callable[[TuneRequest], EvaluatorPair], cache: TuningCache) -> TuningFile:
    maybe_entries = tuple(_tune_one(request, evaluators(request)) for request in requests)
    entries = tuple(entry for entry in maybe_entries if entry is not None)
    tuning_file = TuningFile(gpu_slug=gpu_slug, entries=TunedEntries(items=entries))
    cache.save(tuning_file)
    return tuning_file


def _tune_one(request: TuneRequest, evaluator_pair: EvaluatorPair) -> TunedEntry | None:
    weight_shape = request.weight_shape
    batch = request.batch
    gemv_shape = _as_gemv_shape(weight_shape, batch)
    candidates = candidate_configurations(gemv_shape)
    raw_timed = _time_all(candidates, evaluator_pair.raw)
    fused_timed = _time_all(candidates, evaluator_pair.fused)
    best_raw = best_configuration(raw_timed)
    best_fused = best_configuration(fused_timed)
    matched = order_matched_configuration(raw_timed, fused_timed)
    configurations = _build_configuration_set(best_raw, best_fused, matched)
    if configurations is None:
        return None
    key = ShapeKey(weight_shape=weight_shape, batch=batch)
    return TunedEntry(key=key, configurations=configurations)


def _as_gemv_shape(weight_shape: WeightShape, batch: BatchSize) -> GemvShape:
    output_rows = weight_shape.output_rows
    batch_value = batch.value
    return GemvShape(output_rows=output_rows, batch_size=batch_value)


def _time_all(candidates: tuple[LaunchConfiguration, ...], evaluator: CandidateEvaluator) -> tuple[TimedCandidate, ...]:
    return tuple(_time_one(candidate, evaluator) for candidate in candidates)


def _time_one(configuration: LaunchConfiguration, evaluator: CandidateEvaluator) -> TimedCandidate:
    measurement = evaluator.evaluate(configuration)
    return TimedCandidate(configuration=configuration, measurement=measurement)


def _build_configuration_set(best_raw: LaunchConfiguration | None, best_fused: LaunchConfiguration | None, matched: LaunchConfiguration | None) -> ConfigurationSet | None:
    all_present = best_raw is not None and best_fused is not None and matched is not None
    if not all_present:
        return None
    independent = KernelConfigurations(raw=best_raw, fused=best_fused)
    return ConfigurationSet(independent=independent, order_matched=matched)
