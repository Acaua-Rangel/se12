"""Tests for the per-GPU launch tuner (US-013).

@spec:AC-024
"""

from __future__ import annotations

import warnings

import pytest

from lws.adapters.filesystem.tuning_cache import FilesystemTuningCache
from lws.application.gpu.tune_launch import EvaluatorPair, TuneRequest, tune_launch
from lws.application.ports.gpu.tuning_cache import BatchSize, WeightShape
from lws.domain.device.properties import GpuSlug
from lws.domain.launch.heuristic import GemvShape, default_launch_configuration
from lws.domain.device.properties import SmCount
from lws.domain.launch.search_space import TimingAndAccuracy, candidate_configurations
from tests.fakes import FakeCandidateEvaluator, FakeTuningCache


def test_default_configuration_needs_no_tuning_file():
    """@spec:AC-024 — heuristic over queried SM count and shape, no GPU name."""
    shape = GemvShape(output_rows=2304, batch_size=1)
    configuration = default_launch_configuration(SmCount(80), shape)
    assert configuration.threads_per_block.value > 0
    assert configuration.blocks.value > 0


def test_tune_launch_saves_the_fastest_passing_configuration_per_kernel():
    """@spec:AC-024 — grid search, same space for raw and fused, each keeps its own best."""
    weight_shape = WeightShape(output_rows=2304, k=9216)
    batch = BatchSize(value=1)
    request = TuneRequest(weight_shape=weight_shape, batch=batch)
    shape = GemvShape(output_rows=weight_shape.output_rows, batch_size=batch.value)
    candidates = candidate_configurations(shape)

    # Raw kernel: fastest at the SMALLEST threads_per_block; fused: fastest
    # at the LARGEST — distinct optima, so we can check each independently.
    raw_results = {c: TimingAndAccuracy(median_latency_microseconds=float(c.threads_per_block.value), passes_accuracy_check=True) for c in candidates}
    fused_results = {c: TimingAndAccuracy(median_latency_microseconds=float(-c.threads_per_block.value), passes_accuracy_check=True) for c in candidates}

    def evaluators(_request):
        return EvaluatorPair(raw=FakeCandidateEvaluator(raw_results), fused=FakeCandidateEvaluator(fused_results))

    cache = FakeTuningCache()
    gpu_slug = GpuSlug(value="test-gpu-sm80")

    tuning_file = tune_launch(gpu_slug, [request], evaluators, cache)

    assert tuning_file.gpu_slug == gpu_slug
    entries = tuning_file.entries.items
    assert len(entries) == 1
    entry = entries[0]
    assert entry.key.weight_shape == weight_shape
    assert entry.key.batch == batch
    configurations = entry.configurations
    independent = configurations.independent
    smallest = min(candidates, key=lambda c: c.threads_per_block.value)
    largest = max(candidates, key=lambda c: c.threads_per_block.value)
    assert independent.raw == smallest
    assert independent.fused == largest
    assert cache.stored is tuning_file


def test_tune_launch_skips_a_shape_where_nothing_passes_accuracy():
    """@spec:AC-024 — only configurations that pass the accuracy check are saved."""
    weight_shape = WeightShape(output_rows=256, k=512)
    batch = BatchSize(value=1)
    request = TuneRequest(weight_shape=weight_shape, batch=batch)
    shape = GemvShape(output_rows=weight_shape.output_rows, batch_size=batch.value)
    candidates = candidate_configurations(shape)
    failing_results = {c: TimingAndAccuracy(median_latency_microseconds=1.0, passes_accuracy_check=False) for c in candidates}

    def evaluators(_request):
        return EvaluatorPair(raw=FakeCandidateEvaluator(failing_results), fused=FakeCandidateEvaluator(failing_results))

    cache = FakeTuningCache()
    tuning_file = tune_launch(GpuSlug(value="test-gpu"), [request], evaluators, cache)

    assert tuning_file.entries.items == ()


def test_filesystem_tuning_cache_round_trips(tmp_path):
    """@spec:AC-024"""
    from lws.application.ports.gpu.tuning_cache import ConfigurationSet, KernelConfigurations, ShapeKey, TunedEntries, TunedEntry, TuningFile
    from lws.domain.launch.heuristic import BlockCount, LaunchConfiguration, ThreadsPerBlock

    gpu_slug = GpuSlug(value="tesla-t4-sm75")
    configuration = LaunchConfiguration(threads_per_block=ThreadsPerBlock(128), blocks=BlockCount(64))
    key = ShapeKey(weight_shape=WeightShape(output_rows=2304, k=9216), batch=BatchSize(value=1))
    entry = TunedEntry(key=key, configurations=ConfigurationSet(independent=KernelConfigurations(raw=configuration, fused=configuration), order_matched=configuration))
    tuning_file = TuningFile(gpu_slug=gpu_slug, entries=TunedEntries(items=(entry,)))

    cache = FilesystemTuningCache(tmp_path, gpu_slug)
    cache.save(tuning_file)

    reloaded_cache = FilesystemTuningCache(tmp_path, gpu_slug)
    reloaded = reloaded_cache.load()

    assert reloaded == tuning_file


def test_filesystem_tuning_cache_ignores_a_file_for_a_different_gpu(tmp_path):
    """@spec:AC-024 — a tuning file whose GPU slug differs is ignored with a warning."""
    from lws.application.ports.gpu.tuning_cache import ConfigurationSet, KernelConfigurations, ShapeKey, TunedEntries, TunedEntry, TuningFile
    from lws.domain.launch.heuristic import BlockCount, LaunchConfiguration, ThreadsPerBlock

    written_for = GpuSlug(value="tesla-p100-sm60")
    configuration = LaunchConfiguration(threads_per_block=ThreadsPerBlock(128), blocks=BlockCount(64))
    key = ShapeKey(weight_shape=WeightShape(output_rows=2304, k=9216), batch=BatchSize(value=1))
    entry = TunedEntry(key=key, configurations=ConfigurationSet(independent=KernelConfigurations(raw=configuration, fused=configuration), order_matched=configuration))
    tuning_file = TuningFile(gpu_slug=written_for, entries=TunedEntries(items=(entry,)))
    writer = FilesystemTuningCache(tmp_path, written_for)
    writer.save(tuning_file)

    reader = FilesystemTuningCache(tmp_path, GpuSlug(value="nvidia-h100-sm90"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = reader.load()

    assert result is None
    assert len(caught) == 1
    assert "tesla-p100-sm60" in str(caught[0].message)
    assert "nvidia-h100-sm90" in str(caught[0].message)


def test_filesystem_tuning_cache_returns_none_with_an_empty_cache_dir(tmp_path):
    """@spec:AC-024 — "the heuristic-only test runs with an empty cache dir"."""
    cache = FilesystemTuningCache(tmp_path, GpuSlug(value="tesla-t4-sm75"))
    assert cache.load() is None


@pytest.mark.gpu
def test_deleting_the_tuning_file_changes_speed_only_not_results():
    """@spec:AC-024 — needs a real GPU to compare tuned vs default results."""
    pytest.skip("requires real GPU hardware to execute — see fused-decode-gemv tests for the same caveat")
