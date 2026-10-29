"""Narrow logging controls for expected PyRadiomics advisory messages."""

from __future__ import annotations

import logging


_SUPPRESSED_MESSAGES = (
    (
        "Fixed bin Count enabled! However, we recommend using a fixed bin Width. "
        "See http://pyradiomics.readthedocs.io/en/latest/faq.html"
        "#radiomics-fixed-bin-width for more details"
    ),
    (
        "GLCM is symmetrical, therefore Sum Average = 2 * Joint Average, only 1 "
        "needs to be calculated"
    ),
)

_SOURCE_LOGGERS = (
    "radiomics.featureextractor",
    "radiomics.glcm",
)


class _ExpectedAdvisoryFilter(logging.Filter):
    """Reject only the two expected advisory records named above."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() not in _SUPPRESSED_MESSAGES


def install_pyradiomics_advisory_filters() -> None:
    """Install idempotent exact-message filters on their emitting loggers.

    This leaves every other PyRadiomics log record and Python runtime warning visible.
    """
    for logger_name in _SOURCE_LOGGERS:
        logger = logging.getLogger(logger_name)
        if not any(isinstance(item, _ExpectedAdvisoryFilter) for item in logger.filters):
            logger.addFilter(_ExpectedAdvisoryFilter())
