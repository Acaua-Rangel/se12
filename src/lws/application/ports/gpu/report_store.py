"""Port: writes/reads benchmark and verdict reports as JSON, named by GPU
slug (US-006, US-007, AC-012, AC-014, AC-023). Payloads are plain dicts —
the report shape is inherently a JSON blob (principle P-006's ~29 metadata
fields plus per-shape timings), so a fully domain-typed wrapper for every
field would only duplicate the JSON schema without adding a rule this port
needs to enforce; validity (AC-023) is checked separately, by
lws.domain.benchmark.summary.check_metadata, over whatever was read back.
"""

from __future__ import annotations

import typing

from lws.domain.device.properties import GpuSlug


class ReportStore(typing.Protocol):
    def write_micro_report(self, gpu_slug: GpuSlug, payload: dict) -> None:
        ...

    def write_verdict_report(self, gpu_slug: GpuSlug, payload: dict) -> None:
        ...

    def read_all_micro_reports(self) -> tuple[dict, ...]:
        ...
