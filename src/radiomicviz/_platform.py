"""Reference-platform provenance shared by extraction stages."""

from __future__ import annotations

import platform
from typing import Mapping

REFERENCE_PLATFORM = "Linux-x86_64"


def platform_identifier() -> str:
    """Return a stable operating-system and architecture identifier."""
    machine = platform.machine().lower()
    architecture = "x86_64" if machine in {"amd64", "x86_64"} else machine
    return f"{platform.system()}-{architecture}"


def reference_platform_deviation(software: Mapping[str, str]) -> str | None:
    """Describe a non-reference platform when a real software record includes one."""
    actual = software.get("platform")
    if actual is None or actual == REFERENCE_PLATFORM:
        return None
    return f"platform is {actual}; reference platform is {REFERENCE_PLATFORM}"
