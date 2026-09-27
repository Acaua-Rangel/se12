"""Entrypoint: `python -m lws.device` (optionally `LWS_DEVICE=<index>`, AC-021).

Composition root only: wires the CUDA GpuProbe into the diagnose_device use
case, prints and writes `reports/device-<gpu-slug>.json`, and exits 0 when
the GPU is supported, non-zero with an actionable message otherwise.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys

import torch

from lws.adapters.cuda.gpu_probe import CudaGpuProbe
from lws.adapters.cuda.nvrtc_compiler import force_ptx_requested
from lws.application.gpu.diagnose_device import DeviceDiagnosis, UnsupportedDiagnosis, diagnose_device, diagnosis_is_supported
from lws.domain.device.decisions import ForcePtx

DEFAULT_REPORT_DIRECTORY = pathlib.Path("reports")
NO_GPU_MESSAGE = (
    "UNSUPPORTED: no CUDA-capable GPU was found (torch.cuda.is_available() is "
    "False). Either no NVIDIA GPU is present, the driver is not installed, or "
    "this torch build has no CUDA support (reinstall from "
    "https://pytorch.org/get-started/locally/ with a CUDA index)."
)


def main() -> int:  # calisthenics: allow 3 — POSIX exit code convention
    """AC-021 takes no CLI arguments — only $LWS_DEVICE and $LWS_FORCE_PTX."""
    if not torch.cuda.is_available():
        print(NO_GPU_MESSAGE, file=sys.stderr)
        return 1
    probe = CudaGpuProbe()
    force_ptx = ForcePtx(force_ptx_requested())
    diagnosis = diagnose_device(probe, force_ptx)
    verdict = diagnosis_is_supported(diagnosis)
    _print_and_write(diagnosis)
    return _exit_code(verdict)


def _exit_code(verdict) -> int:
    if verdict.supported:
        return 0
    return 1


def _print_and_write(diagnosis: DeviceDiagnosis) -> None:
    payload = dataclasses.asdict(diagnosis)
    text = json.dumps(payload, indent=2, default=str)
    print(text)
    _write_report(diagnosis, text)
    _print_actionable_message(diagnosis)


def _write_report(diagnosis: DeviceDiagnosis, text: str) -> None:
    properties = diagnosis.properties
    identity_and_compute = properties.identity_and_compute
    identity = identity_and_compute.identity
    slug = identity.slug
    slug_value = slug.value
    directory = DEFAULT_REPORT_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    out_path = directory / f"device-{slug_value}.json"
    out_path.write_text(text)


def _print_actionable_message(diagnosis: DeviceDiagnosis) -> None:
    result = diagnosis.result
    outcome = result.outcome
    if not isinstance(outcome, UnsupportedDiagnosis):
        return
    reason = outcome.reason
    print(f"UNSUPPORTED: {reason}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
