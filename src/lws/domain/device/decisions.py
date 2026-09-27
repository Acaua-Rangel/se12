"""Pure decisions over DeviceProperties: GPU slug, compile target (SASS or
PTX fallback), compute profile, VRAM fit (AC-021, AC-022). Every function
here is tested on CPU with hand-built DeviceProperties — no GPU needed,
matching constitution P-012 and AC-021's "tested on CPU with fake device
properties covering sm_60, sm_75, sm_86, sm_90, sm_120 and sm_52".
"""

from __future__ import annotations

import dataclasses
import enum
import re

from lws.domain.device.properties import (
    Bytes,
    ComputeCapability,
    ComputeHardware,
    DeviceName,
    GpuSlug,
    NvrtcTargets,
    VramSizes,
)

MINIMUM_SUPPORTED_ARCH = 60
BF16_NATIVE_MINIMUM_ARCH = 80
GIGABYTE = 1024**3


class UnsupportedDeviceError(ValueError):
    """The device cannot be targeted at all — below sm_60, or NVRTC has
    nothing usable for it (AC-021's "non-zero with an actionable message")."""


@dataclasses.dataclass(frozen=True)
class ForcePtx:
    value: bool


@dataclasses.dataclass(frozen=True)
class ComputeArch:
    """A compute_XY / sm_XY number, e.g. 75 for sm_75."""

    value: int


class CompileTargetKind(enum.Enum):
    SASS = "sass"
    PTX = "ptx"


@dataclasses.dataclass(frozen=True)
class CompileTarget:
    kind: CompileTargetKind
    arch: ComputeArch


def arch_number(capability: ComputeCapability) -> ComputeArch:
    major = capability.major
    minor = capability.minor
    return ComputeArch(major * 10 + minor)


def gpu_slug(name: DeviceName, capability: ComputeCapability) -> GpuSlug:
    raw_name = name.value
    lowered = raw_name.lower()
    slug_base = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    arch = arch_number(capability)
    arch_value = arch.value
    slug_text = f"{slug_base}-sm{arch_value}"
    return GpuSlug(value=slug_text)


def choose_compile_target(capability: ComputeCapability, nvrtc_targets: NvrtcTargets, force_ptx: ForcePtx) -> CompileTarget:
    """SASS for the device's own arch when NVRTC supports it (unless
    force_ptx), else PTX for the highest supported arch at or below the
    device; raises when neither is possible (AC-022, AC-021's refusal)."""
    device_arch = arch_number(capability)
    _require_supported_device(device_arch)
    sass_target = _sass_if_wanted(device_arch, nvrtc_targets, force_ptx)
    if sass_target is not None:
        return sass_target
    ptx_target = _best_ptx_target(device_arch, nvrtc_targets)
    if ptx_target is not None:
        return ptx_target
    device_value = device_arch.value
    raise UnsupportedDeviceError(f"NVRTC can target neither sm_{device_value} natively nor any PTX architecture at or below it")


def _require_supported_device(device_arch: ComputeArch) -> None:
    value = device_arch.value
    if value >= MINIMUM_SUPPORTED_ARCH:
        return
    raise UnsupportedDeviceError(f"compute capability sm_{value} is below the minimum supported sm_{MINIMUM_SUPPORTED_ARCH}")


def _sass_if_wanted(device_arch: ComputeArch, nvrtc_targets: NvrtcTargets, force_ptx: ForcePtx) -> CompileTarget | None:
    wants_ptx = force_ptx.value
    if wants_ptx:
        return None
    device_value = device_arch.value
    targets = nvrtc_targets.values
    if device_value not in targets:
        return None
    return CompileTarget(kind=CompileTargetKind.SASS, arch=device_arch)


def _best_ptx_target(device_arch: ComputeArch, nvrtc_targets: NvrtcTargets) -> CompileTarget | None:
    device_value = device_arch.value
    targets = nvrtc_targets.values
    candidates = tuple(target for target in targets if target <= device_value)
    if not candidates:
        return None
    best = max(candidates)
    return CompileTarget(kind=CompileTargetKind.PTX, arch=ComputeArch(best))


class ComputeProfileKind(enum.Enum):
    BF16_NATIVE = "bf16-native"
    FP32_ACT = "fp32-act"


def choose_compute_profile(capability: ComputeCapability) -> ComputeProfileKind:
    """BF16 activations on sm_80+ (native BF16 arithmetic), FP32 below it —
    fused-decode-gemv design.md, "Compute profiles"."""
    device_arch = arch_number(capability)
    value = device_arch.value
    if value >= BF16_NATIVE_MINIMUM_ARCH:
        return ComputeProfileKind.BF16_NATIVE
    return ComputeProfileKind.FP32_ACT


@dataclasses.dataclass(frozen=True)
class VramProfile:
    label: str
    required: Bytes


@dataclasses.dataclass(frozen=True)
class VramFitCheck:
    profile: VramProfile
    fits: bool


REFERENCE_VRAM_PROFILES = (
    VramProfile(label="kernel-level (codec + kernels + microbenchmark)", required=Bytes(4 * GIGABYTE)),
    VramProfile(label="packed model (SE12), Gemma 2 2B", required=Bytes(6 * GIGABYTE)),
    VramProfile(label="original BF16 model, Gemma 2 2B", required=Bytes(8 * GIGABYTE)),
)


def check_vram_fit(profiles: tuple[VramProfile, ...], vram: VramSizes) -> tuple[VramFitCheck, ...]:
    return tuple(_fit_one(profile, vram) for profile in profiles)


def _fit_one(profile: VramProfile, vram: VramSizes) -> VramFitCheck:
    required = profile.required
    free = vram.free
    fits = required.value <= free.value
    return VramFitCheck(profile=profile, fits=fits)


@dataclasses.dataclass(frozen=True)
class SupportVerdict:
    supported: bool


def compute_hardware_supports_execution(compute: ComputeHardware) -> SupportVerdict:
    """A cheap pre-check (below sm_60 is never supported) — the definitive
    check (does the installed torch actually run a kernel here) needs the
    adapter's smoke test, which is not a pure decision."""
    capability = compute.capability
    device_arch = arch_number(capability)
    value = device_arch.value
    supported = value >= MINIMUM_SUPPORTED_ARCH
    return SupportVerdict(supported=supported)
