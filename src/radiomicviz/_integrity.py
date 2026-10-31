"""Content fingerprints shared by resumable RadiomicsViz stages."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading the file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fingerprint(
    path: Path,
    *,
    recorded_path: Path | None = None,
    sha256: str | None = None,
) -> dict[str, object]:
    """Describe a file while using its content hash as the stable identity."""
    stat = path.stat()
    return {
        "path": str((recorded_path or path).resolve()),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "sha256": sha256 or sha256_file(path),
    }


def fingerprint_matches(
    recorded: object,
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> bool:
    """Validate file content while deliberately ignoring path and modification time."""
    if not isinstance(recorded, Mapping) or not path.is_file():
        return False
    digest = recorded.get("sha256")
    size = recorded.get("size_bytes")
    if not isinstance(digest, str) or len(digest) != 64:
        return False
    if expected_sha256 is not None and digest != expected_sha256:
        return False
    try:
        if not isinstance(size, int) or path.stat().st_size != size:
            return False
        return sha256_file(path) == digest
    except OSError:
        return False
