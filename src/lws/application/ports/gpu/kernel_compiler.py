"""Port: compiles CUDA C source for a chosen target (AC-022).

`CompiledKernelHandle` is a structural marker only — the domain never
inspects what an adapter returns from `compile`; only that same adapter's
launch code (fused-decode-gemv, later tasks) uses it.
"""

from __future__ import annotations

import dataclasses
import typing

from lws.domain.device.decisions import CompileTarget


@dataclasses.dataclass(frozen=True)
class CudaSource:
    value: str


class CompiledKernelHandle(typing.Protocol):
    pass


class KernelCompiler(typing.Protocol):
    def compile(self, source: CudaSource, target: CompileTarget) -> CompiledKernelHandle:
        ...
