"""Entrypoint: `python -m lws.bench.calibrate --out reports/calibration-<gpu-slug>.json`
(AC-026). Composition root: wires the CUDA CalibrationProbe into
calibrate_device, assembles principle P-006 metadata, writes the report.

UNVERIFIED ON REAL HARDWARE — see se12_common.cuh's note; every GPU-touching
call in this file shares that caveat.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from lws.adapters.cuda.benchmark_timer import CudaBenchmarkTimer
from lws.adapters.cuda.calibration_probe import CudaCalibrationProbe
from lws.adapters.cuda.gpu_probe import CudaGpuProbe
from lws.adapters.cuda.nvrtc_compiler import NvrtcKernelCompiler
from lws.application.gpu.calibrate_device import DeviceCalibration, calibrate_device
from lws.domain.device.decisions import ForcePtx, choose_compile_target

WARMUP_RUNS = 25
TIMED_RUNS = 100


def main(arguments: list[str] | None = None) -> int:  # calisthenics: allow 3 — POSIX exit code convention
    parsed = _parse_arguments(arguments)
    probe = CudaGpuProbe()
    properties = probe.properties()
    identity_and_compute = properties.identity_and_compute
    identity = identity_and_compute.identity
    compute = identity_and_compute.compute
    slug = identity.slug
    capability = compute.capability
    sm_count = compute.sm_count
    memory_and_nvrtc = properties.memory_and_nvrtc
    nvrtc_targets = memory_and_nvrtc.nvrtc_targets
    memory = memory_and_nvrtc.memory
    characteristics = memory.characteristics
    l2_size = characteristics.l2_size
    target = choose_compile_target(capability, nvrtc_targets, ForcePtx(False))

    compiler = NvrtcKernelCompiler()
    timer = CudaBenchmarkTimer(0, l2_size)
    calibration_probe = CudaCalibrationProbe(compiler, target, 0, sm_count, timer)

    calibration = calibrate_device(calibration_probe)
    out_path = parsed.out or pathlib.Path(f"reports/calibration-{slug.value}.json")
    payload = _to_payload(slug, capability, calibration)
    _write(out_path, payload)
    print(f"calibration written to {out_path}")
    return 0


def _parse_arguments(arguments: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m lws.bench.calibrate")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    return parser.parse_args(arguments)


def _to_payload(slug, capability, calibration: DeviceCalibration) -> dict:
    bandwidth = calibration.bandwidth
    per_batch = calibration.per_batch
    items = per_batch.items
    return {
        "gpu_slug": slug.value,
        "compute_capability": f"{capability.major}.{capability.minor}",
        "bandwidth_gbps": bandwidth.value,
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "per_batch": [_batch_to_payload(item) for item in items],
    }


def _batch_to_payload(item) -> dict:
    batch = item.batch
    costs = item.costs
    raw = costs.raw
    se12 = costs.se12
    return {"batch_size": batch.value, "raw_cost_per_weight_us": raw.value, "se12_cost_per_weight_us": se12.value}


def _write(out_path: pathlib.Path, payload: dict) -> None:
    parent = out_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    sys.exit(main())
