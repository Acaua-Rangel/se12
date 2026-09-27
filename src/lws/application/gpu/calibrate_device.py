"""Use case: calibrate the roofline model's inputs for batch sizes 1, 4, 16
(US-015, AC-026)."""

from __future__ import annotations

import dataclasses

from lws.application.ports.gpu.calibration_probe import CalibrationProbe
from lws.application.ports.gpu.tuning_cache import BatchSize
from lws.domain.device.properties import GigabytesPerSecond
from lws.domain.performance.roofline import FormatCosts, GpuCalibration

REFERENCE_BATCH_SIZES = (1, 4, 16)


@dataclasses.dataclass(frozen=True)
class BatchCalibration:
    batch: BatchSize
    costs: FormatCosts


@dataclasses.dataclass(frozen=True)
class PerBatchCalibrations:
    """First-class collection: one BatchCalibration per reference batch size."""

    items: tuple[BatchCalibration, ...]


@dataclasses.dataclass(frozen=True)
class DeviceCalibration:
    bandwidth: GigabytesPerSecond
    per_batch: PerBatchCalibrations


def calibrate_device(probe: CalibrationProbe) -> DeviceCalibration:
    bandwidth = probe.measure_bandwidth()
    batches = tuple(_calibrate_one_batch(probe, size) for size in REFERENCE_BATCH_SIZES)
    return DeviceCalibration(bandwidth=bandwidth, per_batch=PerBatchCalibrations(items=batches))


def _calibrate_one_batch(probe: CalibrationProbe, size: int) -> BatchCalibration:
    batch = BatchSize(value=size)
    costs = probe.measure_compute_bound_costs(batch)
    return BatchCalibration(batch=batch, costs=costs)


def gpu_calibration_for_batch(device_calibration: DeviceCalibration, batch: BatchSize) -> GpuCalibration | None:
    per_batch = device_calibration.per_batch
    items = per_batch.items
    matching = tuple(item for item in items if item.batch == batch)
    if not matching:
        return None
    match = matching[0]
    costs = match.costs
    bandwidth = device_calibration.bandwidth
    return GpuCalibration(bandwidth=bandwidth, costs=costs)
