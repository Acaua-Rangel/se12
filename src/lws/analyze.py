"""Entrypoint: `python -m lws.analyze <model_dir> --out reports/entropy.json`.

Composition root only: parses arguments, wires the safetensors adapter into
the analyze_model use case, and serializes the resulting report as JSON
(US-001, AC-001, AC-002).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

from lws.adapters.safetensors.weight_source import SafetensorsWeightSource
from lws.application.codec.analyze_model import analyze_model
from lws.domain.entropy.report import ModelEntropyReport


def main(arguments: list[str] | None = None) -> int:  # calisthenics: allow 3 — POSIX exit code convention
    parsed = _parse_arguments(arguments)
    source = SafetensorsWeightSource(parsed.model_dir)
    report = analyze_model(source)
    _write_report(report, parsed.out)
    return 0


def _parse_arguments(arguments: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m lws.analyze")
    parser.add_argument("model_dir", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("reports/entropy.json"))
    return parser.parse_args(arguments)


def _write_report(report: ModelEntropyReport, out_path: pathlib.Path) -> None:
    parent = out_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    payload = dataclasses.asdict(report)
    text = json.dumps(payload, indent=2)
    out_path.write_text(text)


if __name__ == "__main__":
    sys.exit(main())
