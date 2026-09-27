"""Adapter: GpuProbe over a real NVIDIA GPU, via torch + cupy.

The ONLY module allowed to read hardware facts (constitution P-012). Every
number is queried at the moment `properties()` / `software_versions()` /
`smoke_test()` is called — nothing here is a constant, a GPU name literal or
a per-model lookup table.

UNVERIFIED ON REAL HARDWARE: this environment has no NVIDIA driver, so this
file could not be exercised while writing it (confirmed: even NVRTC itself
fails to load here with `cudaErrorInsufficientDriver` / a missing
`libnvrtc.so`). Written carefully against the torch/cupy APIs, with
defensive fallbacks where a given cupy/torch version might not expose a
field, but it must be treated as a first draft until a `gpu`-marked test run
on Kaggle or another real GPU (scripts/verify.sh) confirms every field.
"""

from __future__ import annotations

import os

import cupy
import torch

from lws.domain.device.decisions import SupportVerdict, gpu_slug
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

DEVICE_INDEX_ENV_VAR = "LWS_DEVICE"
DDR_FACTOR = 2  # GDDR/HBM transfer data on both clock edges
KILOHERTZ_TO_HERTZ = 1000
BITS_PER_BYTE = 8
BYTES_PER_GIGABYTE = 1000**3  # GB/s uses decimal giga, matching GPU spec sheets
SMOKE_TEST_INPUT = 8
SMOKE_TEST_MULTIPLIER = 2
SMOKE_TEST_EXPECTED = SMOKE_TEST_INPUT * SMOKE_TEST_MULTIPLIER


class CudaGpuProbe:
    """Queries device `device_index` (default: $LWS_DEVICE, else 0)."""

    def __init__(self, device_index: int | None = None) -> None:
        self.device_index = _resolve_device_index(device_index)

    def properties(self) -> DeviceProperties:
        index = self.device_index
        identity_and_compute = _identity_and_compute(index)
        memory_and_nvrtc = _memory_and_nvrtc(index)
        return DeviceProperties(identity_and_compute=identity_and_compute, memory_and_nvrtc=memory_and_nvrtc)

    def software_versions(self) -> SoftwareVersions:
        toolchain = _toolchain_versions()
        torch_and_cupy = _torch_and_cupy()
        core = VersionsCore(toolchain=toolchain, torch_and_cupy=torch_and_cupy)
        nvrtc = _nvrtc_info()
        return SoftwareVersions(core=core, nvrtc=nvrtc)

    def smoke_test(self) -> SupportVerdict:
        index = self.device_index
        return _run_smoke_test(index)


def _resolve_device_index(device_index: int | None) -> int:
    if device_index is not None:
        return device_index
    raw = os.environ.get(DEVICE_INDEX_ENV_VAR)
    if raw is None:
        return 0
    return int(raw)


def _identity_and_compute(index: int) -> IdentityAndCompute:
    torch_properties = torch.cuda.get_device_properties(index)
    raw_name = torch_properties.name
    name = DeviceName(raw_name)
    major = torch_properties.major
    minor = torch_properties.minor
    capability = ComputeCapability(major=major, minor=minor)
    slug = gpu_slug(name, capability)
    identity = DeviceIdentity(name=name, slug=slug)
    processor_count = torch_properties.multi_processor_count
    sm_count = SmCount(processor_count)
    compute = ComputeHardware(capability=capability, sm_count=sm_count)
    return IdentityAndCompute(identity=identity, compute=compute)


def _memory_and_nvrtc(index: int) -> MemoryAndNvrtc:
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    vram = VramSizes(total=Bytes(total_bytes), free=Bytes(free_bytes))
    attributes = _device_attributes(index)
    l2_size = Bytes(attributes.get("L2CacheSize", 0))
    bandwidth = _estimate_bandwidth(attributes)
    characteristics = MemoryCharacteristics(l2_size=l2_size, bandwidth=bandwidth)
    memory = MemoryProfile(vram=vram, characteristics=characteristics)
    nvrtc_targets = _nvrtc_targets()
    return MemoryAndNvrtc(memory=memory, nvrtc_targets=nvrtc_targets)


def _device_attributes(index: int) -> dict:
    device = cupy.cuda.Device(index)
    try:
        return device.attributes
    except Exception:  # defensive: attribute query shape can vary by cupy version
        return {}


def _estimate_bandwidth(attributes: dict) -> GigabytesPerSecond:
    memory_clock_khz = attributes.get("MemoryClockRate", 0)
    bus_width_bits = attributes.get("GlobalMemoryBusWidth", 0)
    hertz = memory_clock_khz * KILOHERTZ_TO_HERTZ
    bytes_per_transfer = bus_width_bits / BITS_PER_BYTE
    bytes_per_second = hertz * DDR_FACTOR * bytes_per_transfer
    gigabytes_per_second = bytes_per_second / BYTES_PER_GIGABYTE
    return GigabytesPerSecond(gigabytes_per_second)


def _nvrtc_targets() -> NvrtcTargets:
    values = _query_nvrtc_supported_archs()
    return NvrtcTargets(values=values)


def _query_nvrtc_supported_archs() -> tuple[int, ...]:
    from cupy.cuda import nvrtc

    try:
        archs = nvrtc.getSupportedArchs()
    except Exception:  # defensive: older/newer cupy may expose this differently
        return ()
    return tuple(int(arch) for arch in archs)


def _run_smoke_test(index: int) -> SupportVerdict:
    try:
        return _run_smoke_test_unsafe(index)
    except Exception:
        return SupportVerdict(supported=False)


def _run_smoke_test_unsafe(index: int) -> SupportVerdict:
    device = torch.device("cuda", index)
    tensor = torch.full((SMOKE_TEST_INPUT,), 1.0, device=device)
    doubled = tensor * SMOKE_TEST_MULTIPLIER
    total = doubled.sum()
    result = total.item()
    supported = result == SMOKE_TEST_EXPECTED
    return SupportVerdict(supported=supported)


def _toolchain_versions() -> ToolchainVersions:
    driver_version = _driver_version_string()
    cuda_runtime_version = _cuda_runtime_version_string()
    return ToolchainVersions(driver=driver_version, cuda_runtime=cuda_runtime_version)


def _driver_version_string() -> str:
    try:
        raw = cupy.cuda.runtime.driverGetVersion()
    except Exception:
        return "unknown"
    return _decode_cuda_version(raw)


def _cuda_runtime_version_string() -> str:
    try:
        raw = cupy.cuda.runtime.runtimeGetVersion()
    except Exception:
        return "unknown"
    return _decode_cuda_version(raw)


def _decode_cuda_version(raw: int) -> str:
    # CUDA's own encoding: raw = major * 1000 + minor * 10
    major = raw // 1000
    minor = (raw % 1000) // 10
    return f"{major}.{minor}"


def _torch_and_cupy() -> TorchAndCupy:
    torch_info = _torch_info()
    cupy_version = cupy.__version__
    return TorchAndCupy(torch=torch_info, cupy_version=cupy_version)


def _torch_info() -> TorchInfo:
    version = torch.__version__
    arch_list = torch.cuda.get_arch_list()
    return TorchInfo(version=version, arch_list=TorchArchList(values=tuple(arch_list)))


def _nvrtc_info() -> NvrtcInfo:
    from cupy.cuda import nvrtc

    version_string = _nvrtc_version_string(nvrtc)
    targets = _nvrtc_targets()
    return NvrtcInfo(version=version_string, targets=targets)


def _nvrtc_version_string(nvrtc: object) -> str:
    try:
        major, minor = nvrtc.getVersion()
    except Exception:
        return "unknown"
    return f"{major}.{minor}"
