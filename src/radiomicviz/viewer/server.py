"""Run the viewer with Werkzeug's threaded server until interrupted."""

from __future__ import annotations

import logging
import secrets
import socket
import sys
import threading
import webbrowser
from pathlib import Path

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


class ViewerStartError(Exception):
    """The viewer cannot start; the message names the next step."""


def run_server(
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
    out=None,
) -> None:
    out = out or sys.stdout
    if host not in LOOPBACK and not allow_remote:
        raise ViewerStartError(
            f"refusing to bind to {host}: use the default 127.0.0.1 with SSH port forwarding, "
            "or add --allow-remote"
        )
    from werkzeug.serving import make_server

    from radiomicviz.viewer.app import build_app

    token = secrets.token_urlsafe(32)
    app = build_app(
        manifest_path,
        variant=variant,
        cache_mb=cache_mb,
        verify_integrity=verify_integrity,
        blind=blind,
        access_token=token,
        # An explicit remote bind may be reached through a hostname or any address on the
        # selected interface. The unguessable token still gates every request.
        allowed_hosts=("*",) if host not in LOOPBACK and allow_remote else (host,),
    )
    unavailable = ViewerStartError(
        f"port {port} is not available on {host}; choose another --port or use --port 0"
    )
    if port:
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((host, port))
            except OSError:
                raise unavailable from None
    try:
        server = make_server(host, port, app, threaded=True)
    except (OSError, SystemExit):  # Werkzeug exits the process itself when bind fails
        raise unavailable from None
    bound = int(server.server_port)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    url_host = "127.0.0.1" if host in {"127.0.0.1", "localhost"} else (f"[{host}]" if ":" in host else host)
    url = f"http://{url_host}:{bound}/?token={token}"
    print(f"RadiomicsViz viewer (read-only): {url}", file=out)
    print(
        f"On a remote machine, forward the port first: ssh -L {bound}:localhost:{bound} <cluster-login>",
        file=out,
    )
    print("Press Ctrl+C to stop.", file=out, flush=True)
    if host not in LOOPBACK:
        print(
            f"WARNING: listening on {host}; anyone who can reach this port and has the URL can read the data.",
            file=sys.stderr,
        )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    timer = None
    if open_browser:
        timer = threading.Timer(0.8, lambda: webbrowser.open(url))
        timer.daemon = True
        timer.start()
    try:
        while thread.is_alive():
            thread.join(timeout=0.5)
    except KeyboardInterrupt:
        print("Viewer stopped.", file=out)
    finally:
        if timer is not None:
            timer.cancel()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
