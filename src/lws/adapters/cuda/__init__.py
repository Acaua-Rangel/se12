"""Adapters that touch a real NVIDIA GPU: query it (gpu_probe.py), compile
CUDA C for it (nvrtc_compiler.py), and — from fused-decode-gemv T-004
onward — launch kernels on it. This package cannot be exercised on a
machine without a GPU; its tests are marked `gpu` and skip otherwise (a
skip is never proof — see scripts/verify.sh)."""
