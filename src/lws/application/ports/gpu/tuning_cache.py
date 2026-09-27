"""Port: reads and writes the per-GPU launch-tuning cache (US-013, AC-024).

One file per GPU slug (`tune-<gpu-slug>.json`); a file whose slug does not
match the current GPU is ignored with a warning (the adapter's job, not
this port's).
"""

from __future__ import annotations

import dataclasses
import typing

from lws.domain.device.properties import GpuSlug
from lws.domain.launch.heuristic import LaunchConfiguration


@dataclasses.dataclass(frozen=True)
class BatchSize:
    value: int


@dataclasses.dataclass(frozen=True)
class WeightShape:
    """A weight matrix's (N, K) — NOT lws.domain.launch.heuristic.GemvShape,
    which deliberately holds only what the launch heuristic needs
    (output_rows, batch_size) and has no K. A tuning cache key needs all
    three dimensions (N, K, batch), so it gets its own type rather than
    reusing or extending GemvShape for a purpose it was not designed for."""

    output_rows: int
    k: int


@dataclasses.dataclass(frozen=True)
class ShapeKey:
    weight_shape: WeightShape
    batch: BatchSize


@dataclasses.dataclass(frozen=True)
class KernelConfigurations:
    raw: LaunchConfiguration
    fused: LaunchConfiguration


@dataclasses.dataclass(frozen=True)
class ConfigurationSet:
    independent: KernelConfigurations
    order_matched: LaunchConfiguration


@dataclasses.dataclass(frozen=True)
class TunedEntry:
    key: ShapeKey
    configurations: ConfigurationSet


@dataclasses.dataclass(frozen=True)
class TunedEntries:
    """First-class collection: one tuned entry per (shape, batch) searched."""

    items: tuple[TunedEntry, ...]


@dataclasses.dataclass(frozen=True)
class TuningFile:
    gpu_slug: GpuSlug
    entries: TunedEntries


class TuningCache(typing.Protocol):
    def load(self) -> TuningFile | None:
        """None when there is no cache file for the CURRENT GPU slug — either
        none exists, or one exists for a different GPU (ignored with a
        warning by the adapter, not treated as an error)."""
        ...

    def save(self, tuning_file: TuningFile) -> None:
        ...
