from __future__ import annotations

import hashlib

import pytest

from radiomicviz.protocols import ProtocolError, load_protocol


def test_builtin_protocol_authenticates_frozen_roi_preset():
    protocol = load_protocol()
    preset = protocol.roi_preset("b32")

    assert protocol.protocol_id == "lesion-habitat-v1"
    assert protocol.status == "reference-verified"
    assert preset.parameter_path.stat().st_size == 3295
    assert hashlib.sha256(preset.parameter_path.read_bytes()).hexdigest() == (
        "cd2f3e1da1028daaec55b5573b7e2d986b7f34302617452103724d94ff0f942b"
    )
    assert preset.expected_feature_count == 1834
    assert preset.expected_shape_feature_count == 14
    assert preset.expected_diagnostics_count == 37
    assert preset.expected_feature_schema_sha256 == (
        "f4ea6da5d5191bfb24ad81f05cf7792d7c853294e460fcaf959621c00f02240e"
    )


def test_unknown_protocol_and_variant_are_explicit_errors():
    with pytest.raises(ProtocolError, match="unknown protocol"):
        load_protocol("other")
    with pytest.raises(ProtocolError, match="available: b32"):
        load_protocol().roi_preset("b10")
