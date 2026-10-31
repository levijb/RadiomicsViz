"""Read-only browser viewer for RadiomicsViz output trees.

Flask is imported only when an app is created, so ``radiomicviz --help`` and every
pipeline command work without the optional ``viewer`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["create_app", "serve"]


def create_app(
    manifest_path: str | Path,
    *,
    variant: str = "b32",
    cache_mb: int = 512,
    verify_integrity: bool = False,
    blind: bool = False,
    access_token: str | None = None,
    allowed_hosts: tuple[str, ...] = (),
) -> Any:
    """Build the Flask application for one output tree (requires the viewer extra)."""
    from radiomicviz.viewer.app import build_app

    return build_app(
        manifest_path,
        variant=variant,
        cache_mb=cache_mb,
        verify_integrity=verify_integrity,
        blind=blind,
        access_token=access_token,
        allowed_hosts=allowed_hosts,
    )


def serve(
    manifest_path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
    variant: str = "b32",
    cache_mb: int = 512,
    verify_integrity: bool = False,
    blind: bool = False,
    allow_remote: bool = False,
) -> None:
    """Serve the viewer until interrupted (requires the viewer extra)."""
    from radiomicviz.viewer.server import run_server

    run_server(
        manifest_path,
        host=host,
        port=port,
        open_browser=open_browser,
        variant=variant,
        cache_mb=cache_mb,
        verify_integrity=verify_integrity,
        blind=blind,
        allow_remote=allow_remote,
    )
