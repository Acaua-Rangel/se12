"""Port: reads hardware facts from the GPU at hand.

Constitution P-012: the adapter implementing this port is the ONLY code
allowed to read GPU facts (device name, SM count, VRAM, NVRTC targets, ...);
every decision over those facts is a pure domain function (lws.domain.device)
tested with a value built by hand or by this port's fake.
"""

from __future__ import annotations

import typing

from lws.domain.device.decisions import SupportVerdict
from lws.domain.device.properties import DeviceProperties, SoftwareVersions


class GpuProbe(typing.Protocol):
    def properties(self) -> DeviceProperties:
        ...

    def software_versions(self) -> SoftwareVersions:
        ...

    def smoke_test(self) -> SupportVerdict:
        """Does the installed torch actually run a kernel on this device —
        the one check no static property can answer (AC-021)."""
        ...
