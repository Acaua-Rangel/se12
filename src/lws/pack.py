"""Entrypoint: `python -m lws.pack <model_dir> <out_dir>` (US-003).

Composition root only: parses arguments, wires the safetensors adapters into
the pack_model use case, prints the summary.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from lws.adapters.safetensors.packed_store import SafetensorsPackedStore
from lws.adapters.safetensors.weight_source import SafetensorsWeightSource
from lws.application.codec.pack_model import PackSummary, pack_model
from lws.domain.se12.summary import compression_ratio, meets_compression_target


def main(arguments: list[str] | None = None) -> int:  # calisthenics: allow 3 — POSIX exit code convention
    parsed = _parse_arguments(arguments)
    source = SafetensorsWeightSource(parsed.model_dir)
    store = SafetensorsPackedStore(parsed.out_dir)
    summary = pack_model(source, store)
    _print_summary(summary)
    return 0


def _parse_arguments(arguments: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m lws.pack")
    parser.add_argument("model_dir", type=pathlib.Path)
    parser.add_argument("out_dir", type=pathlib.Path)
    return parser.parse_args(arguments)


def _print_summary(summary: PackSummary) -> None:
    sizes = summary.eligible_sizes
    original = sizes.original
    packed = sizes.packed
    ratio = compression_ratio(sizes)
    verdict = meets_compression_target(ratio)
    counts = summary.tensor_counts
    encoded_count = counts.encoded
    passed_through_count = counts.passed_through
    print(f"eligible tensors encoded: {encoded_count.value}")
    print(f"tensors copied through unchanged: {passed_through_count.value}")
    print(f"original bytes (eligible only): {original.value}")
    print(f"packed bytes (eligible only): {packed.value}")
    print(f"ratio: {ratio.value:.4f} (target <= 0.78, met: {verdict.within_target})")


if __name__ == "__main__":
    sys.exit(main())
