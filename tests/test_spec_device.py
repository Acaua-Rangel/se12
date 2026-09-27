"""Tests for the device doctor (US-012).

@spec:AC-021 @spec:AC-022
"""

from __future__ import annotations

import pytest

from lws.application.gpu.diagnose_device import (
    SupportedDiagnosis,
    UnsupportedDiagnosis,
    diagnose_device,
    diagnosis_is_supported,
)
from lws.domain.device.decisions import (
    BF16_NATIVE_MINIMUM_ARCH,
    ComputeProfileKind,
    ForcePtx,
    SupportVerdict,
    UnsupportedDeviceError,
    check_vram_fit,
    choose_compile_target,
    choose_compute_profile,
    compute_hardware_supports_execution,
    gpu_slug,
    CompileTargetKind,
    REFERENCE_VRAM_PROFILES,
)
from lws.domain.device.properties import (
    Bytes,
    ComputeCapability,
    ComputeHardware,
    DeviceIdentity,
    DeviceName,
    DeviceProperties,
    GigabytesPerSecond,
    IdentityAndCompute,
    MemoryAndNvrtc,
    MemoryCharacteristics,
    MemoryProfile,
    NvrtcInfo,
    NvrtcTargets,
    SmCount,
    SoftwareVersions,
    ToolchainVersions,
    TorchAndCupy,
    TorchArchList,
    TorchInfo,
    VersionsCore,
    VramSizes,
)
from tests.fakes import FakeGpuProbe

# NVRTC targets a reasonably current toolchain supports (used across tests).
NVRTC_TARGETS = NvrtcTargets(values=(60, 61, 70, 75, 80, 86, 89, 90))

# The six architectures AC-021 explicitly asks the decision logic to cover.
SM_CASES = (
    (6, 0, "Tesla P100-PCIE-16GB"),
    (7, 5, "Tesla T4"),
    (8, 6, "NVIDIA GeForce RTX 3060"),
    (9, 0, "NVIDIA H100 80GB HBM3"),
    (12, 0, "NVIDIA RTX 5090"),
    (5, 2, "NVIDIA GeForce GTX 970"),  # below sm_60: unsupported
)


def _capability(major: int, minor: int) -> ComputeCapability:
    return ComputeCapability(major=major, minor=minor)


def _properties(major: int, minor: int, name: str, free_bytes: int = 8 * 1024**3, nvrtc_targets: NvrtcTargets = NVRTC_TARGETS) -> DeviceProperties:
    capability = _capability(major, minor)
    device_name = DeviceName(name)
    slug = gpu_slug(device_name, capability)
    identity = DeviceIdentity(name=device_name, slug=slug)
    compute = ComputeHardware(capability=capability, sm_count=SmCount(80))
    identity_and_compute = IdentityAndCompute(identity=identity, compute=compute)
    vram = VramSizes(total=Bytes(16 * 1024**3), free=Bytes(free_bytes))
    characteristics = MemoryCharacteristics(l2_size=Bytes(4 * 1024**2), bandwidth=GigabytesPerSecond(700.0))
    memory = MemoryProfile(vram=vram, characteristics=characteristics)
    memory_and_nvrtc = MemoryAndNvrtc(memory=memory, nvrtc_targets=nvrtc_targets)
    return DeviceProperties(identity_and_compute=identity_and_compute, memory_and_nvrtc=memory_and_nvrtc)


def _software() -> SoftwareVersions:
    toolchain = ToolchainVersions(driver="550.90.07", cuda_runtime="12.6")
    torch_info = TorchInfo(version="2.5.0", arch_list=TorchArchList(values=("sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90")))
    torch_and_cupy = TorchAndCupy(torch=torch_info, cupy_version="13.4.0")
    core = VersionsCore(toolchain=toolchain, torch_and_cupy=torch_and_cupy)
    nvrtc = NvrtcInfo(version="12.6", targets=NVRTC_TARGETS)
    return SoftwareVersions(core=core, nvrtc=nvrtc)


@pytest.mark.parametrize("major,minor,name", SM_CASES)
def test_gpu_slug_is_deterministic_and_arch_suffixed(major, minor, name):
    """@spec:AC-021"""
    capability = _capability(major, minor)
    slug = gpu_slug(DeviceName(name), capability)
    assert slug.value.endswith(f"-sm{major}{minor}")
    assert slug.value == slug.value.lower()
    assert " " not in slug.value


@pytest.mark.parametrize("major,minor,name", SM_CASES)
def test_compile_target_matches_supported_architectures(major, minor, name):
    """@spec:AC-021 @spec:AC-022 — sm_60/75/86/90 supported natively (SASS);
    sm_120 falls back to the highest PTX arch at or below it; sm_52 is refused."""
    capability = _capability(major, minor)
    if major * 10 + minor < 60:
        with pytest.raises(UnsupportedDeviceError) as error:
            choose_compile_target(capability, NVRTC_TARGETS, ForcePtx(False))
        assert "60" in str(error.value) or "6.0" in str(error.value) or "sm_" in str(error.value)
        return
    target = choose_compile_target(capability, NVRTC_TARGETS, ForcePtx(False))
    device_arch = major * 10 + minor
    if device_arch in NVRTC_TARGETS.values:
        assert target.kind is CompileTargetKind.SASS
        assert target.arch.value == device_arch
        return
    assert target.kind is CompileTargetKind.PTX
    assert target.arch.value <= device_arch
    assert target.arch.value in NVRTC_TARGETS.values


def test_force_ptx_uses_ptx_even_when_sass_is_supported():
    """@spec:AC-022"""
    capability = _capability(7, 5)  # sm_75, natively supported
    target = choose_compile_target(capability, NVRTC_TARGETS, ForcePtx(True))
    assert target.kind is CompileTargetKind.PTX
    assert target.arch.value == 75


def test_no_supported_target_at_all_raises_with_found_and_supported():
    """@spec:AC-021 @spec:AC-022 — NVRTC too old to target this device at all."""
    capability = _capability(9, 0)  # sm_90
    empty_nvrtc = NvrtcTargets(values=())  # nothing at all -> no SASS, no PTX fallback possible
    with pytest.raises(UnsupportedDeviceError) as error:
        choose_compile_target(capability, empty_nvrtc, ForcePtx(False))
    message = str(error.value)
    assert "90" in message


@pytest.mark.parametrize(
    "major,minor,expected",
    [
        (6, 0, ComputeProfileKind.FP32_ACT),
        (7, 5, ComputeProfileKind.FP32_ACT),
        (7, 9, ComputeProfileKind.FP32_ACT),
        (8, 0, ComputeProfileKind.BF16_NATIVE),
        (8, 6, ComputeProfileKind.BF16_NATIVE),
        (9, 0, ComputeProfileKind.BF16_NATIVE),
        (12, 0, ComputeProfileKind.BF16_NATIVE),
    ],
)
def test_compute_profile_switches_at_sm80(major, minor, expected):
    """@spec:AC-021 — BF16 native only from sm_80 (fused-decode-gemv design.md)."""
    capability = _capability(major, minor)
    profile = choose_compute_profile(capability)
    assert profile is expected
    assert (major * 10 + minor >= BF16_NATIVE_MINIMUM_ARCH) == (profile is ComputeProfileKind.BF16_NATIVE)


def test_vram_fit_checks_reference_profiles():
    """@spec:AC-021 — which features and model profiles fit in the free VRAM."""
    vram = VramSizes(total=Bytes(16 * 1024**3), free=Bytes(7 * 1024**3))
    checks = check_vram_fit(REFERENCE_VRAM_PROFILES, vram)
    fits_by_label = {check.profile.label: check.fits for check in checks}
    assert fits_by_label["kernel-level (codec + kernels + microbenchmark)"] is True
    assert fits_by_label["packed model (SE12), Gemma 2 2B"] is True
    assert fits_by_label["original BF16 model, Gemma 2 2B"] is False


def test_below_sm60_is_never_supported():
    """@spec:AC-021"""
    compute = ComputeHardware(capability=_capability(5, 2), sm_count=SmCount(10))
    verdict = compute_hardware_supports_execution(compute)
    assert verdict.supported is False


@pytest.mark.parametrize("major,minor,name", SM_CASES)
def test_diagnose_device_end_to_end_with_a_fake_probe(major, minor, name):
    """@spec:AC-021 — the full use case, exercised with a fake GpuProbe (no GPU needed)."""
    properties = _properties(major, minor, name)
    software = _software()
    probe = FakeGpuProbe(properties=properties, software=software, smoke_test_result=SupportVerdict(supported=True))

    diagnosis = diagnose_device(probe, ForcePtx(False))
    verdict = diagnosis_is_supported(diagnosis)

    device_arch = major * 10 + minor
    if device_arch < 60:
        assert verdict.supported is False
        assert isinstance(diagnosis.result.outcome, UnsupportedDiagnosis)
        return
    assert verdict.supported is True
    outcome = diagnosis.result.outcome
    assert isinstance(outcome, SupportedDiagnosis)
    profile_and_fit = outcome.profile_and_fit
    assert len(profile_and_fit.vram_fit.items) == len(REFERENCE_VRAM_PROFILES)


def test_diagnose_device_reports_unsupported_when_the_smoke_test_fails():
    """@spec:AC-021 — a capability-eligible GPU whose installed torch cannot actually run a kernel."""
    properties = _properties(7, 5, "Tesla T4")
    software = _software()
    probe = FakeGpuProbe(properties=properties, software=software, smoke_test_result=SupportVerdict(supported=False))

    diagnosis = diagnose_device(probe, ForcePtx(False))

    verdict = diagnosis_is_supported(diagnosis)
    assert verdict.supported is False
    outcome = diagnosis.result.outcome
    assert isinstance(outcome, UnsupportedDiagnosis)
    assert "torch" in outcome.reason
