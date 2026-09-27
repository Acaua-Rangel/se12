"""Adapter: TuningCache as one JSON file per GPU slug (US-013, AC-024).

A file whose recorded GPU slug does not match the current one is ignored
with a warning — never applied to the wrong GPU.
"""

from __future__ import annotations

import json
import pathlib
import warnings

from lws.application.ports.gpu.tuning_cache import (
    BatchSize,
    ConfigurationSet,
    KernelConfigurations,
    ShapeKey,
    TunedEntries,
    TunedEntry,
    TuningFile,
    WeightShape,
)
from lws.domain.device.properties import GpuSlug
from lws.domain.launch.heuristic import BlockCount, LaunchConfiguration, ThreadsPerBlock

FILENAME_PREFIX = "tune-"
FILENAME_SUFFIX = ".json"


class FilesystemTuningCache:
    """Reads/writes `<directory>/tune-<current gpu slug>.json`.

    `load()` globs for every `tune-*.json` present rather than checking only
    the exact expected filename: a directory can hold a file left over from
    a DIFFERENT GPU (a shared/synced cache directory, or simply switching
    GPUs on the same machine) — that file must still be found, recognized by
    its OWN recorded slug, and ignored with a warning (AC-024), not silently
    treated as "no cache" the way an exact-filename check alone would.
    """

    def __init__(self, directory: pathlib.Path, gpu_slug: GpuSlug) -> None:
        self.directory = directory
        self.gpu_slug = gpu_slug

    def load(self) -> TuningFile | None:
        directory = self.directory
        gpu_slug = self.gpu_slug
        pattern = f"{FILENAME_PREFIX}*{FILENAME_SUFFIX}"
        candidates = sorted(directory.glob(pattern))
        return _load_first_match(candidates, gpu_slug)

    def save(self, tuning_file: TuningFile) -> None:
        directory = self.directory
        directory.mkdir(parents=True, exist_ok=True)
        path = self._path()
        payload = _to_json(tuning_file)
        path.write_text(json.dumps(payload, indent=2))

    def _path(self) -> pathlib.Path:
        directory = self.directory
        gpu_slug = self.gpu_slug
        slug_value = gpu_slug.value
        return directory / f"{FILENAME_PREFIX}{slug_value}{FILENAME_SUFFIX}"


def _load_first_match(candidates: list[pathlib.Path], expected_slug: GpuSlug) -> TuningFile | None:
    mismatched = tuple(_check_one(path, expected_slug) for path in candidates)
    matching = tuple(result for result in mismatched if result is not None)
    if matching:
        return matching[0]
    if candidates:
        _warn_about_mismatch(candidates[0], expected_slug)
    return None


def _check_one(path: pathlib.Path, expected_slug: GpuSlug) -> TuningFile | None:
    payload = json.loads(path.read_text())
    found_slug = payload.get("gpu_slug")
    expected_value = expected_slug.value
    if found_slug != expected_value:
        return None
    return _from_json(payload)


def _warn_about_mismatch(path: pathlib.Path, expected_slug: GpuSlug) -> None:
    payload = json.loads(path.read_text())
    found_slug = payload.get("gpu_slug")
    expected_value = expected_slug.value
    warnings.warn(f"ignoring tuning cache {path}: it was recorded for GPU '{found_slug}', current GPU is '{expected_value}'")


def _to_json(tuning_file: TuningFile) -> dict:
    gpu_slug = tuning_file.gpu_slug
    entries = tuning_file.entries
    items = entries.items
    return {"gpu_slug": gpu_slug.value, "entries": [_entry_to_json(entry) for entry in items]}


def _entry_to_json(entry: TunedEntry) -> dict:
    key = entry.key
    configurations = entry.configurations
    return {"key": _key_to_json(key), "configurations": _configurations_to_json(configurations)}


def _key_to_json(key: ShapeKey) -> dict:
    weight_shape = key.weight_shape
    batch = key.batch
    return {"output_rows": weight_shape.output_rows, "k": weight_shape.k, "batch": batch.value}


def _configurations_to_json(configurations: ConfigurationSet) -> dict:
    independent = configurations.independent
    order_matched = configurations.order_matched
    return {
        "raw": _configuration_to_json(independent.raw),
        "fused": _configuration_to_json(independent.fused),
        "order_matched": _configuration_to_json(order_matched),
    }


def _configuration_to_json(configuration: LaunchConfiguration) -> dict:
    threads_per_block = configuration.threads_per_block
    blocks = configuration.blocks
    return {"threads_per_block": threads_per_block.value, "blocks": blocks.value}


def _from_json(payload: dict) -> TuningFile:
    gpu_slug = GpuSlug(value=payload["gpu_slug"])
    raw_entries = payload["entries"]
    entries = tuple(_entry_from_json(raw_entry) for raw_entry in raw_entries)
    return TuningFile(gpu_slug=gpu_slug, entries=TunedEntries(items=entries))


def _entry_from_json(raw_entry: dict) -> TunedEntry:
    key = _key_from_json(raw_entry["key"])
    configurations = _configurations_from_json(raw_entry["configurations"])
    return TunedEntry(key=key, configurations=configurations)


def _key_from_json(raw_key: dict) -> ShapeKey:
    weight_shape = WeightShape(output_rows=raw_key["output_rows"], k=raw_key["k"])
    batch = BatchSize(value=raw_key["batch"])
    return ShapeKey(weight_shape=weight_shape, batch=batch)


def _configurations_from_json(raw_configurations: dict) -> ConfigurationSet:
    raw = _configuration_from_json(raw_configurations["raw"])
    fused = _configuration_from_json(raw_configurations["fused"])
    order_matched = _configuration_from_json(raw_configurations["order_matched"])
    independent = KernelConfigurations(raw=raw, fused=fused)
    return ConfigurationSet(independent=independent, order_matched=order_matched)


def _configuration_from_json(raw_configuration: dict) -> LaunchConfiguration:
    threads_per_block = ThreadsPerBlock(value=raw_configuration["threads_per_block"])
    blocks = BlockCount(value=raw_configuration["blocks"])
    return LaunchConfiguration(threads_per_block=threads_per_block, blocks=blocks)
