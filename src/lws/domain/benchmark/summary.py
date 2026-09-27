"""Cross-GPU summary: which reports are valid vs rejected for missing
principle P-006 metadata (AC-023). Table rendering is a presentation
concern (the entrypoint); this module only decides validity.
"""

from __future__ import annotations

import dataclasses

# Every field constitution P-006 requires a benchmark report to carry.
REQUIRED_METADATA_FIELDS = (
    "gpu_name",
    "gpu_slug",
    "compute_capability",
    "sm_count",
    "vram_bytes",
    "l2_size_bytes",
    "peak_copy_bandwidth_gbps",
    "driver_version",
    "cuda_runtime_version",
    "nvrtc_version",
    "torch_version",
    "cupy_version",
    "compile_target",
    "compute_profile",
    "sm_clock_mhz",
    "memory_clock_mhz",
    "temperature_celsius",
    "power_limit_watts",
    "ecc_enabled",
    "mig_enabled",
    "display_attached",
    "cuda_graphs_enabled",
    "shape",
    "batch_size",
    "warmup_runs",
    "timed_runs",
    "median_microseconds",
    "p10_microseconds",
    "p90_microseconds",
)


@dataclasses.dataclass(frozen=True)
class ReportMetadata:
    """First-class collection: the raw metadata fields found in one report."""

    fields: dict


@dataclasses.dataclass(frozen=True)
class MissingFields:
    """First-class collection: which required fields a report is missing."""

    names: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class MetadataCheck:
    is_valid: bool
    missing: MissingFields


def check_metadata(metadata: ReportMetadata) -> MetadataCheck:
    fields = metadata.fields
    missing_names = tuple(name for name in REQUIRED_METADATA_FIELDS if name not in fields)
    is_valid = len(missing_names) == 0
    return MetadataCheck(is_valid=is_valid, missing=MissingFields(names=missing_names))


@dataclasses.dataclass(frozen=True)
class ValidReport:
    metadata: ReportMetadata


@dataclasses.dataclass(frozen=True)
class RejectedReport:
    metadata: ReportMetadata
    missing: MissingFields


@dataclasses.dataclass(frozen=True)
class ClassifiedReports:
    valid: ValidReportList
    rejected: RejectedReportList


@dataclasses.dataclass(frozen=True)
class ValidReportList:
    """First-class collection."""

    items: tuple[ValidReport, ...]


@dataclasses.dataclass(frozen=True)
class RejectedReportList:
    """First-class collection."""

    items: tuple[RejectedReport, ...]


def classify_reports(reports: tuple[ReportMetadata, ...]) -> ClassifiedReports:
    checked = tuple((report, check_metadata(report)) for report in reports)
    valid = tuple(ValidReport(metadata=report) for report, check in checked if check.is_valid)
    rejected = tuple(RejectedReport(metadata=report, missing=check.missing) for report, check in checked if not check.is_valid)
    return ClassifiedReports(valid=ValidReportList(items=valid), rejected=RejectedReportList(items=rejected))
