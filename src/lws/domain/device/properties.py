"""Value objects for what is queried from a GPU (constitution P-012: the
GpuProbe adapter is the only code that reads these facts from real
hardware; everything else — including every test — works with a
DeviceProperties value built by hand or by a fake)."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class DeviceName:
    value: str


@dataclasses.dataclass(frozen=True)
class GpuSlug:
    value: str


@dataclasses.dataclass(frozen=True)
class ComputeCapability:
    major: int
    minor: int


@dataclasses.dataclass(frozen=True)
class SmCount:
    value: int


@dataclasses.dataclass(frozen=True)
class Bytes:
    value: int

    def __post_init__(self) -> None:
        value = self.value
        if value < 0:
            raise ValueError("Bytes cannot be negative")


@dataclasses.dataclass(frozen=True)
class GigabytesPerSecond:
    value: float


@dataclasses.dataclass(frozen=True)
class NvrtcTargets:
    """First-class collection: the compute_XY numbers this NVRTC can target,
    e.g. (60, 61, 70, 75, 80, 86, 89, 90) — queried once per process, never
    hard-coded (P-012)."""

    values: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class DeviceIdentity:
    name: DeviceName
    slug: GpuSlug


@dataclasses.dataclass(frozen=True)
class ComputeHardware:
    capability: ComputeCapability
    sm_count: SmCount


@dataclasses.dataclass(frozen=True)
class IdentityAndCompute:
    identity: DeviceIdentity
    compute: ComputeHardware


@dataclasses.dataclass(frozen=True)
class VramSizes:
    total: Bytes
    free: Bytes


@dataclasses.dataclass(frozen=True)
class MemoryCharacteristics:
    l2_size: Bytes
    bandwidth: GigabytesPerSecond


@dataclasses.dataclass(frozen=True)
class MemoryProfile:
    vram: VramSizes
    characteristics: MemoryCharacteristics


@dataclasses.dataclass(frozen=True)
class MemoryAndNvrtc:
    memory: MemoryProfile
    nvrtc_targets: NvrtcTargets


@dataclasses.dataclass(frozen=True)
class DeviceProperties:
    """Everything a domain decision (compile target, compute profile, VRAM
    fit, slug) needs. Toolchain version strings (driver, CUDA runtime, torch,
    cupy) are report-only (AC-021) and deliberately kept out of this type —
    no decision function needs them, so they never need to flow through the
    decision call chain (see SoftwareVersions in the same package)."""

    identity_and_compute: IdentityAndCompute
    memory_and_nvrtc: MemoryAndNvrtc


@dataclasses.dataclass(frozen=True)
class TorchInfo:
    version: str
    arch_list: TorchArchList


@dataclasses.dataclass(frozen=True)
class TorchArchList:
    """First-class collection: the CUDA architectures this torch build ships kernels for."""

    values: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class NvrtcInfo:
    version: str
    targets: NvrtcTargets


@dataclasses.dataclass(frozen=True)
class ToolchainVersions:
    driver: str
    cuda_runtime: str


@dataclasses.dataclass(frozen=True)
class TorchAndCupy:
    torch: TorchInfo
    cupy_version: str


@dataclasses.dataclass(frozen=True)
class VersionsCore:
    toolchain: ToolchainVersions
    torch_and_cupy: TorchAndCupy


@dataclasses.dataclass(frozen=True)
class SoftwareVersions:
    """Report-only version strings (AC-021); never consumed by a decision."""

    core: VersionsCore
    nvrtc: NvrtcInfo
