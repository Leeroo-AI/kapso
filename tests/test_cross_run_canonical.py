import json

import pytest

from kapso.cross_run.canonical import (
    CANONICALIZER_VERSION,
    CanonicalizationError,
    canonical_json_bytes,
    content_id,
    normalize_utc_timestamp,
    parse_json_bytes,
    tree_or_blob_digest,
    verify_declared_content_id,
)


def test_canonical_json_has_stable_golden_bytes_and_identity():
    payload = {"unicode": "λ", "nested": {"z": 2, "a": [True, None]}, "a": 1}

    encoded = canonical_json_bytes(payload)

    assert CANONICALIZER_VERSION == "kapso.canonical_json.v1"
    assert encoded == (b'{"a":1,"nested":{"a":[true,null],"z":2},"unicode":"\xce\xbb"}')
    assert content_id("fixture", payload) == (
        "fixture:sha256:fe11db981673073d71c8eeb5e8fee76d00874e6fda8abc664460348c8e22ccd5"
    )
    assert tree_or_blob_digest(encoded) == (
        "sha256:fe11db981673073d71c8eeb5e8fee76d00874e6fda8abc664460348c8e22ccd5"
    )


def test_canonical_identity_ignores_input_mapping_order():
    left = {"outer": {"b": 2, "a": 1}, "name": "same"}
    right = {"name": "same", "outer": {"a": 1, "b": 2}}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert content_id("fixture", left) == content_id("fixture", right)


@pytest.mark.parametrize(
    "payload",
    [
        '{"key":1,"key":2}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"value":1e999}',
    ],
)
def test_strict_json_parser_rejects_duplicate_keys_and_non_finite_numbers(payload):
    with pytest.raises(CanonicalizationError):
        parse_json_bytes(payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {1: "bad"}, {"x"}])
def test_canonical_json_rejects_unsupported_values(value):
    with pytest.raises(CanonicalizationError):
        canonical_json_bytes(value)


def test_verify_declared_identity_excludes_envelope_fields():
    immutable = {"value": "scientific"}
    declared = content_id("fixture", immutable)
    first = {"record_id": declared, **immutable, "attestation": {"signer": "one"}}
    second = {"record_id": declared, **immutable, "attestation": {"signer": "two"}}

    verify_declared_content_id(
        first,
        namespace="fixture",
        identity_field="record_id",
        excluded_fields=("attestation",),
    )
    verify_declared_content_id(
        second,
        namespace="fixture",
        identity_field="record_id",
        excluded_fields=("attestation",),
    )

    changed = dict(second)
    changed["value"] = "different"
    with pytest.raises(CanonicalizationError):
        verify_declared_content_id(
            changed,
            namespace="fixture",
            identity_field="record_id",
            excluded_fields=("attestation",),
        )


def test_timestamp_requires_normalized_utc_not_naive_or_offset_time():
    assert normalize_utc_timestamp("2026-07-20T12:30:00Z", "timestamp") == (
        "2026-07-20T12:30:00Z"
    )
    for invalid in (
        "2026-07-20T12:30:00",
        "2026-07-20T12:30:00+00:00",
        "2026-07-20T13:30:00+01:00",
        "2026-07-20T12:30:00.000000Z",
        "2026-13-20T12:30:00Z",
        "2026-02-30T12:30:00Z",
        "2026-07-20T12:30:60Z",
    ):
        with pytest.raises(CanonicalizationError):
            normalize_utc_timestamp(invalid, "timestamp")


def test_python_json_default_duplicate_behavior_would_have_hidden_corruption():
    assert json.loads('{"same":1,"same":2}') == {"same": 2}
    with pytest.raises(CanonicalizationError):
        parse_json_bytes('{"same":1,"same":2}')
