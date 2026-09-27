"""In-memory fakes for every application port, used by tests instead of real
adapters (constitution P-013: "every port has ... one in-memory fake used by
the tests"). This module is under tests/, so it is exempt from the
calisthenics rules that govern src/lws.
"""

from __future__ import annotations

import typing

from lws.application.ports.codec.pack_source import ShardOfTensors
from lws.application.ports.codec.packed_store import PackedShard
from lws.application.ports.codec.weight_source import SourceTensor
from lws.application.ports.gpu.kernel_compiler import CompiledKernelHandle, CudaSource
from lws.domain.device.decisions import CompileTarget, SupportVerdict
from lws.domain.device.properties import DeviceProperties, SoftwareVersions


class FakeWeightSource:
    """A WeightSource backed by a fixed, in-memory list of tensors."""

    def __init__(self, items: typing.Sequence[SourceTensor]) -> None:
        self._items = list(items)

    def tensors(self) -> typing.Iterator[SourceTensor]:
        return iter(self._items)


class FakePackSource:
    """A PackSource backed by a fixed, in-memory mapping of shard name -> tensors."""

    def __init__(self, shards: typing.Sequence[tuple[str, typing.Sequence[SourceTensor]]]) -> None:
        self._shards = list(shards)

    def shards(self) -> typing.Iterator[ShardOfTensors]:
        for name, items in self._shards:
            yield ShardOfTensors(shard_name=name, tensors=iter(items))


class FakePackedStore:
    """A PackedStore that keeps every written shard in memory instead of on disk."""

    def __init__(self) -> None:
        self.written_shards: list[PackedShard] = []

    def write_shard(self, shard: PackedShard) -> None:
        self.written_shards.append(shard)


class FakeGpuProbe:
    """A GpuProbe that reports a fixed, hand-built device — no GPU needed."""

    def __init__(self, properties: DeviceProperties, software: SoftwareVersions, smoke_test_result: SupportVerdict) -> None:
        self._properties = properties
        self._software = software
        self._smoke_test_result = smoke_test_result

    def properties(self) -> DeviceProperties:
        return self._properties

    def software_versions(self) -> SoftwareVersions:
        return self._software

    def smoke_test(self) -> SupportVerdict:
        return self._smoke_test_result


class FakeCompiledKernelHandle:
    """A CompiledKernelHandle carrying nothing but the target it was 'compiled' for."""

    def __init__(self, target: CompileTarget) -> None:
        self.target = target


class FakeKernelCompiler:
    """A KernelCompiler that records what it was asked to compile instead of calling NVRTC."""

    def __init__(self) -> None:
        self.compiled: list[tuple[CudaSource, CompileTarget]] = []

    def compile(self, source: CudaSource, target: CompileTarget) -> CompiledKernelHandle:
        self.compiled.append((source, target))
        return FakeCompiledKernelHandle(target)
