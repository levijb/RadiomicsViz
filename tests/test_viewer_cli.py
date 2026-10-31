"""Viewer command line (T8) and shipped static assets (T11)."""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

import _viewer_support as support
from radiomicviz.cli import main

VIEWER = Path(__import__("radiomicviz.viewer", fromlist=["x"]).__file__).parent


def test_help_and_pipeline_commands_do_not_import_flask():
    code = (
        "import sys; from click.testing import CliRunner; from radiomicviz.cli import main; "
        "r = CliRunner(); [r.invoke(main, args) for args in (['--help'], ['--version'], ['view', '--help'], "
        "['whole', '--help'])]; print('flask' in sys.modules, 'SimpleITK' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip().splitlines()[-1] == "False False"


def test_view_help_and_missing_flask_message(monkeypatch):
    help_text = CliRunner().invoke(main, ["view", "--help"]).output
    assert "--manifest" in help_text and "--blind" in help_text and "--no-browser" in help_text
    real_import = builtins.__import__

    def no_flask(name, *args, **kwargs):
        if name == "flask" or name.startswith("flask."):
            raise ImportError("No module named 'flask'")
        return real_import(name, *args, **kwargs)

    manifest_like = Path(__file__)
    monkeypatch.setattr(builtins, "__import__", no_flask)
    result = CliRunner().invoke(main, ["view", "--manifest", str(manifest_like)])
    assert result.exit_code == 1
    assert "pip install" in result.output and "flask" in result.output.lower()
    assert result.output.count("\n") <= 2


def test_bad_manifest_paths(tmp_path):
    pytest.importorskip("flask")
    missing = CliRunner().invoke(main, ["view", "--manifest", str(tmp_path / "nope.csv")])
    assert missing.exit_code == 2
    other = tmp_path / "other.csv"
    other.write_text("a\n", encoding="utf-8")
    wrong = CliRunner().invoke(main, ["view", "--manifest", str(other), "--no-browser"])
    assert wrong.exit_code == 1 and "regions_manifest.csv" in wrong.output and str(tmp_path) not in wrong.output


def test_busy_port_remote_host_and_printed_url(tmp_path, monkeypatch):
    pytest.importorskip("flask")
    from radiomicviz.viewer import server

    project = support.build_inputs(tmp_path)
    manifest = support.write_manifest(project)
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        result = CliRunner().invoke(main, ["view", "--manifest", str(manifest), "--no-browser", "--port", str(port)])
    assert result.exit_code == 1 and f"port {port} is not available" in result.output and "--port 0" in result.output
    remote = CliRunner().invoke(main, ["view", "--manifest", str(manifest), "--no-browser", "--host", "0.0.0.0"])
    assert remote.exit_code == 1 and "--allow-remote" in remote.output

    class FakeServer:
        server_port = 50123

        def serve_forever(self):
            return None

        def shutdown(self):
            return None

        def server_close(self):
            return None

    import werkzeug.serving

    monkeypatch.setattr(werkzeug.serving, "make_server", lambda *args, **kwargs: FakeServer())
    monkeypatch.setattr(server.webbrowser, "open", lambda url: pytest.fail("browser opened despite --no-browser"))
    out = io.StringIO()
    server.run_server(manifest, open_browser=False, out=out)
    printed = out.getvalue()
    assert re.search(r"http://127\.0\.0\.1:50123/\?token=[A-Za-z0-9_-]{20,}", printed)
    assert "ssh -L 50123:localhost:50123 <cluster-login>" in printed


def test_static_assets_are_shipped_and_safe():
    vendor = VIEWER / "static" / "vendor" / "niivue"
    record = json.loads((vendor / "VENDOR.json").read_text(encoding="utf-8"))
    bundle = vendor / record["file"]
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == record["file_sha256"]
    assert (vendor / "LICENSE").read_text(encoding="utf-8").startswith("BSD 2-Clause License")
    notices = (VIEWER / "THIRD_PARTY_NOTICES").read_text(encoding="utf-8")
    for name in ("NiiVue", "gl-matrix", "fflate", "nifti-reader-js", "zarrita", "numcodecs", "@lukeed/uuid",
                 "array-equal", "@ungap/structured-clone"):
        assert name in notices
    script = (VIEWER / "static" / "viewer.js").read_text(encoding="utf-8")
    assert "innerHTML" not in script and "insertAdjacentHTML" not in script and "eval(" not in script
    template = (VIEWER / "templates" / "index.html").read_text(encoding="utf-8")
    assert "<script>" not in template and "http://" not in template and "https://" not in template
