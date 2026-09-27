"""Entrypoint: `python -m lws.survey <repo_id or dir>... --out reports/entropy-survey.json` (AC-034).

Composition root only: for each argument, decides whether it is a local
directory or a Hugging Face repo id, wires the matching adapter, then calls
the survey_models use case and writes the JSON report. Runs entirely on CPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

from lws.adapters.huggingface.shard_stream_source import HuggingFaceShardStreamSource, resolve_revision
from lws.adapters.safetensors.weight_source import SafetensorsWeightSource
from lws.application.codec.survey_models import ModelIdentity, MultiModelSurveyReport, NamedModelSource, survey_models
from lws.domain.entropy.survey import TensorRole


def main(arguments: list[str] | None = None) -> int:  # calisthenics: allow 3 — POSIX exit code convention
    parsed = _parse_arguments(arguments)
    sources = tuple(_resolve_source(model_ref) for model_ref in parsed.models)
    report = survey_models(sources)
    _write_report(report, parsed.out)
    return 0


def _parse_arguments(arguments: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m lws.survey")
    parser.add_argument("models", nargs="+", help="local model directories or Hugging Face repo ids")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("reports/entropy-survey.json"))
    return parser.parse_args(arguments)


def _resolve_source(model_ref: str) -> NamedModelSource:
    path = pathlib.Path(model_ref)
    if path.is_dir():
        return _local_source(path)
    return _huggingface_source(model_ref)


def _local_source(path: pathlib.Path) -> NamedModelSource:
    identity = ModelIdentity(label=str(path), revision=None)
    source = SafetensorsWeightSource(path)
    return NamedModelSource(identity=identity, source=source)


def _huggingface_source(repo_id: str) -> NamedModelSource:
    revision = resolve_revision(repo_id)
    identity = ModelIdentity(label=repo_id, revision=revision)
    source = HuggingFaceShardStreamSource(repo_id, revision)
    return NamedModelSource(identity=identity, source=source)


def _write_report(report: MultiModelSurveyReport, out_path: pathlib.Path) -> None:
    parent = out_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    payload = dataclasses.asdict(report)
    text = json.dumps(payload, indent=2, default=_json_default)
    out_path.write_text(text)


def _json_default(value: object) -> str:
    if isinstance(value, TensorRole):
        return value.value
    return str(value)


if __name__ == "__main__":
    sys.exit(main())
