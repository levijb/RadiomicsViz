import importlib.metadata
import subprocess
import sys

import radiomicviz
from radiomicviz.config.templates import TEMPLATE_NAMES, template_text


def test_version_matches_distribution_metadata():
    assert radiomicviz.__version__ == importlib.metadata.version("radiomicsviz")


def test_python_dash_m_reports_version():
    result = subprocess.run(
        [sys.executable, "-B", "-m", "radiomicviz", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"radiomicviz {radiomicviz.__version__}"


def test_templates_ship_with_the_package():
    for name in TEMPLATE_NAMES:
        text = template_text(name)
        assert text.startswith("# ====")
        assert "\r" not in text
