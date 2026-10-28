"""Commented starter template for ``project.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from importlib.resources import files
from pathlib import Path

TEMPLATE_NAMES = ("project",)


@dataclass(frozen=True)
class WrittenTemplate:
    """Result of writing a template.

    Attributes:
        path: The file that now holds the template.
        backup: Where the previous file was moved, or ``None`` if there was none.
    """

    path: Path
    backup: Path | None


def template_text(name: str) -> str:
    """Return the text of the built-in ``project`` template."""
    if name not in TEMPLATE_NAMES:
        raise ValueError(f"unknown template {name!r}; expected one of {', '.join(TEMPLATE_NAMES)}")
    return (files("radiomicviz") / "templates" / f"{name}.yaml").read_text(encoding="utf-8")


def write_template(name: str, destination: str | Path, *, force: bool = False) -> WrittenTemplate:
    """Write the template ``name`` to ``destination``.

    An existing file is never overwritten in place. Without ``force`` the call fails;
    with ``force`` the existing file is first renamed to ``<file name>.bak-<timestamp>``
    in the same directory.

    Raises:
        FileExistsError: ``destination`` exists and ``force`` is false.
        IsADirectoryError: ``destination`` is a directory.
        FileNotFoundError: The parent directory of ``destination`` does not exist.
    """
    text = template_text(name)
    destination = Path(destination)
    if destination.is_dir():
        raise IsADirectoryError(f"{destination} is a directory")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"directory {destination.parent} does not exist")

    backup = None
    if destination.exists():
        if not force:
            raise FileExistsError(f"{destination} already exists")
        backup = _backup_path(destination)
        destination.rename(backup)

    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return WrittenTemplate(path=destination, backup=backup)


def _backup_path(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    candidate = path.with_name(f"{path.name}.bak-{stamp}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.bak-{stamp}-{counter}")
        counter += 1
    return candidate
