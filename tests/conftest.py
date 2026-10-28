import pytest
import yaml

from radiomicviz.config.templates import template_text


@pytest.fixture
def project_data():
    """A valid project mapping, freshly parsed from the shipped template."""
    return yaml.safe_load(template_text("project"))
