"""Adapter: KernelCompiler via CuPy RawModule + NVRTC (AC-022).

Compiled binaries are cached on disk via CuPy's own kernel cache
(`CUPY_CACHE_DIR`), which CuPy keys by source code, compile options (which
include the `-arch=` flag, i.e. the chosen target) and NVRTC version — the
exact three things AC-022 asks for — so a second process does not
recompile. `$LWS_CACHE_DIR` (default `~/.cache/lws`) points that cache
somewhere durable instead of CuPy's own default location.

UNVERIFIED ON REAL HARDWARE — see gpu_probe.py's note; NVRTC itself is
unavailable in this environment (no libnvrtc.so), so this file could not be
exercised at all while writing it.
"""

from __future__ import annotations

import os
import pathlib

import cupy

from lws.application.ports.gpu.kernel_compiler import CudaSource
from lws.domain.device.decisions import CompileTarget, CompileTargetKind

CACHE_DIR_ENV_VAR = "LWS_CACHE_DIR"
CUPY_CACHE_DIR_ENV_VAR = "CUPY_CACHE_DIR"
FORCE_PTX_ENV_VAR = "LWS_FORCE_PTX"
DEFAULT_CACHE_SUBDIRECTORY = pathlib.Path(".cache") / "lws"


class CupyCompiledKernelHandle:
    """Structural implementation of CompiledKernelHandle: a compiled RawModule."""

    def __init__(self, module: cupy.RawModule, target: CompileTarget) -> None:
        self.module = module
        self.target = target

    def function(self, name: str) -> cupy.RawKernel:  # calisthenics: allow 3 — CUDA C symbol name, passed straight to cupy
        module = self.module
        return module.get_function(name)


class NvrtcKernelCompiler:
    """Compiles CUDA C source with CuPy's NVRTC backend."""

    def __init__(self) -> None:
        directory = _cache_directory()
        self.cache_directory = directory
        _configure_cupy_cache(directory)

    def compile(self, source: CudaSource, target: CompileTarget) -> CupyCompiledKernelHandle:
        options = _compile_options(target)
        text = source.value
        module = cupy.RawModule(code=text, options=options)
        return CupyCompiledKernelHandle(module=module, target=target)


def force_ptx_requested() -> bool:  # calisthenics: allow 3 — a plain env-var flag, matches ForcePtx's own bare-bool field
    """Reads $LWS_FORCE_PTX (AC-022's testability clause) — a composition
    root turns this into a `ForcePtx` value before calling
    `choose_compile_target`; the compiler itself never reads it."""
    raw = os.environ.get(FORCE_PTX_ENV_VAR, "")
    return raw == "1"


def _cache_directory() -> pathlib.Path:
    raw = os.environ.get(CACHE_DIR_ENV_VAR)
    if raw:
        return pathlib.Path(raw)
    home = pathlib.Path.home()
    return home / DEFAULT_CACHE_SUBDIRECTORY


def _configure_cupy_cache(directory: pathlib.Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(CUPY_CACHE_DIR_ENV_VAR, str(directory))


def _compile_options(target: CompileTarget) -> tuple[str, ...]:
    flag = _arch_flag(target)
    return (flag, "-O3")


def _arch_flag(target: CompileTarget) -> str:
    kind = target.kind
    arch = target.arch
    arch_value = arch.value
    if kind is CompileTargetKind.SASS:
        return f"-arch=sm_{arch_value}"
    return f"-arch=compute_{arch_value}"
