"""Port: computes y = x . W^T from a weight matrix already bound into the
kernel object — either raw BF16 or SE12 (US-005). Both share the same
skeleton (principle P-009: same grid, same reduction, only the load+decode
step differs), so the SAME comparison harness runs against either one.

Activations and the result are torch tensors in practice; the port keeps
them as `typing.Any` since ports never import torch (P-013) — the CUDA
adapter's own type hints are precise about it.
"""

from __future__ import annotations

import typing


class MatVecKernel(typing.Protocol):
    def multiply(self, activations: typing.Any) -> typing.Any:
        ...
