"""SE12 packed-container format name and version rule (AC-008)."""

from __future__ import annotations

import dataclasses

FORMAT_NAME = "SE12"
CURRENT_FORMAT_VERSION = 1
SUPPORTED_FORMAT_VERSIONS = (1,)


class UnsupportedFormatVersionError(ValueError):
    """A packed file declares a format version this codec does not know."""


@dataclasses.dataclass(frozen=True)
class FormatVersion:
    value: int


def validate_format_version(found: FormatVersion) -> None:
    """Refuses an unknown format version, naming the found and supported ones (AC-008)."""
    value = found.value
    if value in SUPPORTED_FORMAT_VERSIONS:
        return
    supported = ", ".join(str(version) for version in SUPPORTED_FORMAT_VERSIONS)
    raise UnsupportedFormatVersionError(f"found format version {value}, supported: {supported}")
