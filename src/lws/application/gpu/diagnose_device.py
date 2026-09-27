"""Use case: diagnose the GPU at hand (US-012, AC-021).

Every decision is delegated to lws.domain.device — this use case only
orchestrates: query the probe, run the domain rules in order (capability
floor, smoke test, compile target, compute profile, VRAM fit), and shape the
result. Fully testable with a fake GpuProbe (tests/fakes.py); no real GPU
needed to exercise this file.
"""

from __future__ import annotations

import dataclasses

from lws.application.ports.gpu.gpu_probe import GpuProbe
from lws.domain.device.decisions import (
    REFERENCE_VRAM_PROFILES,
    CompileTarget,
    ComputeProfileKind,
    ForcePtx,
    SupportVerdict,
    UnsupportedDeviceError,
    VramFitCheck,
    check_vram_fit,
    choose_compile_target,
    choose_compute_profile,
    compute_hardware_supports_execution,
)
from lws.domain.device.properties import ComputeCapability, DeviceProperties, NvrtcTargets, SoftwareVersions


@dataclasses.dataclass(frozen=True)
class VramFitChecks:
    """First-class collection: one fit check per reference VRAM profile."""

    items: tuple[VramFitCheck, ...]


@dataclasses.dataclass(frozen=True)
class ProfileAndFit:
    compute_profile: ComputeProfileKind
    vram_fit: VramFitChecks


@dataclasses.dataclass(frozen=True)
class SupportedDiagnosis:
    compile_target: CompileTarget
    profile_and_fit: ProfileAndFit


@dataclasses.dataclass(frozen=True)
class UnsupportedDiagnosis:
    reason: str


DiagnosisOutcome = SupportedDiagnosis | UnsupportedDiagnosis


@dataclasses.dataclass(frozen=True)
class DiagnosisResult:
    outcome: DiagnosisOutcome
    software: SoftwareVersions


@dataclasses.dataclass(frozen=True)
class DeviceDiagnosis:
    properties: DeviceProperties
    result: DiagnosisResult


def diagnose_device(probe: GpuProbe, force_ptx: ForcePtx) -> DeviceDiagnosis:
    properties = probe.properties()
    software = probe.software_versions()
    outcome = _build_outcome(properties, probe, force_ptx)
    result = DiagnosisResult(outcome=outcome, software=software)
    return DeviceDiagnosis(properties=properties, result=result)


def diagnosis_is_supported(diagnosis: DeviceDiagnosis) -> SupportVerdict:
    result = diagnosis.result
    outcome = result.outcome
    supported = isinstance(outcome, SupportedDiagnosis)
    return SupportVerdict(supported=supported)


def _build_outcome(properties: DeviceProperties, probe: GpuProbe, force_ptx: ForcePtx) -> DiagnosisOutcome:
    identity_and_compute = properties.identity_and_compute
    compute = identity_and_compute.compute
    capability_check = compute_hardware_supports_execution(compute)
    if not capability_check.supported:
        capability = compute.capability
        major = capability.major
        minor = capability.minor
        return UnsupportedDiagnosis(reason=f"compute capability {major}.{minor} is below the minimum supported 6.0")
    smoke_result = probe.smoke_test()
    if not smoke_result.supported:
        return UnsupportedDiagnosis(reason="the installed torch cannot run a kernel on this device")
    return _build_supported(properties, force_ptx)


def _build_supported(properties: DeviceProperties, force_ptx: ForcePtx) -> DiagnosisOutcome:
    identity_and_compute = properties.identity_and_compute
    compute = identity_and_compute.compute
    capability = compute.capability
    memory_and_nvrtc = properties.memory_and_nvrtc
    nvrtc_targets = memory_and_nvrtc.nvrtc_targets
    compile_target = _try_compile_target(capability, nvrtc_targets, force_ptx)
    if isinstance(compile_target, UnsupportedDiagnosis):
        return compile_target
    compute_profile = choose_compute_profile(capability)
    memory = memory_and_nvrtc.memory
    vram = memory.vram
    fit_checks = check_vram_fit(REFERENCE_VRAM_PROFILES, vram)
    profile_and_fit = ProfileAndFit(compute_profile=compute_profile, vram_fit=VramFitChecks(items=fit_checks))
    return SupportedDiagnosis(compile_target=compile_target, profile_and_fit=profile_and_fit)


def _try_compile_target(capability: ComputeCapability, nvrtc_targets: NvrtcTargets, force_ptx: ForcePtx) -> CompileTarget | UnsupportedDiagnosis:
    try:
        return choose_compile_target(capability, nvrtc_targets, force_ptx)
    except UnsupportedDeviceError as error:
        message = str(error)
        return UnsupportedDiagnosis(reason=message)
