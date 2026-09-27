"""Adapter: ReportStore as plain JSON files (US-006, US-007, AC-012, AC-014,
AC-023). Output names use the GPU slug: `micro-<slug>.json`,
`verdict-<slug>.json`.
"""

from __future__ import annotations

import json
import pathlib

from lws.domain.device.properties import GpuSlug

MICRO_PREFIX = "micro-"
VERDICT_PREFIX = "verdict-"
SUFFIX = ".json"


class FilesystemReportStore:
    def __init__(self, directory: pathlib.Path) -> None:
        self.directory = directory

    def write_micro_report(self, gpu_slug: GpuSlug, payload: dict) -> None:
        self._write(MICRO_PREFIX, gpu_slug, payload)

    def write_verdict_report(self, gpu_slug: GpuSlug, payload: dict) -> None:
        self._write(VERDICT_PREFIX, gpu_slug, payload)

    def read_all_micro_reports(self) -> tuple[dict, ...]:
        directory = self.directory
        pattern = f"{MICRO_PREFIX}*{SUFFIX}"
        paths = sorted(directory.glob(pattern))
        return tuple(json.loads(path.read_text()) for path in paths)

    def _write(self, prefix: str, gpu_slug: GpuSlug, payload: dict) -> None:
        directory = self.directory
        directory.mkdir(parents=True, exist_ok=True)
        slug_value = gpu_slug.value
        path = directory / f"{prefix}{slug_value}{SUFFIX}"
        path.write_text(json.dumps(payload, indent=2))
