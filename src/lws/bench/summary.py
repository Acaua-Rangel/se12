"""Entrypoint: `python -m lws.bench.summary reports/ --out reports/summary.md`
(AC-023). Composition root: wires the filesystem ReportStore into the
summarize_reports use case and renders the classified reports as markdown.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from lws.adapters.filesystem.report_store import FilesystemReportStore
from lws.application.gpu.summarize_reports import summarize_reports
from lws.domain.benchmark.summary import ClassifiedReports, RejectedReport, ValidReport


def main(arguments: list[str] | None = None) -> int:  # calisthenics: allow 3 — POSIX exit code convention
    parsed = _parse_arguments(arguments)
    store = FilesystemReportStore(parsed.reports_dir)
    classified = summarize_reports(store)
    text = _render(classified)
    _write(parsed.out, text)
    print(text)
    return 0


def _parse_arguments(arguments: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m lws.bench.summary")
    parser.add_argument("reports_dir", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("reports/summary.md"))
    return parser.parse_args(arguments)


def _render(classified: ClassifiedReports) -> str:
    valid = classified.valid
    rejected = classified.rejected
    valid_items = valid.items
    rejected_items = rejected.items
    lines = ["# Cross-GPU summary", "", _render_valid_table(valid_items), "", _render_rejected_section(rejected_items)]
    return "\n".join(lines)


def _render_valid_table(valid_items: tuple[ValidReport, ...]) -> str:
    header = "| GPU | Compute capability | Peak bandwidth (GB/s) | Raw kernel (% of peak) | Verdict |"
    separator = "|---|---|---|---|---|"
    rows = tuple(_render_row(item) for item in valid_items)
    return "\n".join((header, separator, *rows))


def _render_row(item: ValidReport) -> str:
    metadata = item.metadata
    fields = metadata.fields
    name = fields.get("gpu_name", "?")
    capability = fields.get("compute_capability", "?")
    peak = fields.get("peak_copy_bandwidth_gbps", "?")
    raw_fraction = fields.get("raw_bandwidth_fraction_of_peak", "?")
    verdict = fields.get("verdict", "?")
    return f"| {name} | {capability} | {peak} | {raw_fraction} | {verdict} |"


def _render_rejected_section(rejected_items: tuple[RejectedReport, ...]) -> str:
    if not rejected_items:
        return "No rejected reports."
    lines = ["## Rejected reports (missing principle P-006 metadata)", ""]
    lines.extend(_render_rejected_line(item) for item in rejected_items)
    return "\n".join(lines)


def _render_rejected_line(item: RejectedReport) -> str:
    metadata = item.metadata
    fields = metadata.fields
    name = fields.get("gpu_name", "unknown")
    missing = item.missing
    names = missing.names
    joined_names = ", ".join(names)
    return f"- {name}: missing {joined_names}"


def _write(out_path: pathlib.Path, text: str) -> None:
    parent = out_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)


if __name__ == "__main__":
    sys.exit(main())
