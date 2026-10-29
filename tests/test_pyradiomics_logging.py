from __future__ import annotations

import logging

from radiomicviz._pyradiomics_logging import (
    _ExpectedAdvisoryFilter,
    _SOURCE_LOGGERS,
    _SUPPRESSED_MESSAGES,
    install_pyradiomics_advisory_filters,
)


def _record(logger_name: str, message: str) -> logging.LogRecord:
    return logging.LogRecord(
        logger_name,
        logging.WARNING,
        __file__,
        1,
        message,
        (),
        None,
    )


def test_filter_suppresses_only_the_two_expected_advisories():
    advisory_filter = _ExpectedAdvisoryFilter()

    for logger_name, message in zip(_SOURCE_LOGGERS, _SUPPRESSED_MESSAGES):
        assert not advisory_filter.filter(_record(logger_name, message))

    assert advisory_filter.filter(
        _record("radiomics.firstorder", "Mean of empty slice")
    )
    assert advisory_filter.filter(
        _record("radiomics.glcm", "A different GLCM warning")
    )


def test_filter_installation_is_idempotent():
    install_pyradiomics_advisory_filters()
    install_pyradiomics_advisory_filters()

    for logger_name in _SOURCE_LOGGERS:
        installed = [
            item
            for item in logging.getLogger(logger_name).filters
            if isinstance(item, _ExpectedAdvisoryFilter)
        ]
        assert len(installed) == 1
