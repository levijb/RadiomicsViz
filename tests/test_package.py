import importlib.metadata
import subprocess
import sys

import radiomicviz
from radiomicviz._platform import reference_platform_deviation
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


def test_reference_platform_is_explicit_but_optional_for_injected_test_records():
    assert reference_platform_deviation({}) is None
    assert reference_platform_deviation({"platform": "Linux-x86_64"}) is None
    assert "reference platform is Linux-x86_64" in reference_platform_deviation(
        {"platform": "Windows-x86_64"}
    )
