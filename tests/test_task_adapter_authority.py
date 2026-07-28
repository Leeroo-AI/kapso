from __future__ import annotations

import pytest

from kapso.core.config import load_config
from kapso.cross_run.canonical import canonical_json_bytes
from kapso.cross_run.settings import CrossRunSettings
from kapso.cross_run.task_adapter_authority import (
    CanonicalTaskAdapterAuthority,
    TaskAdapterAuthorityError,
)
from test_task_adapter_store import _activation_envelope, _manifest, _package, _store

_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
_ADAPTER_SOURCE = b"def evaluate(value):\n    return value + 1\n"


def _authority() -> CanonicalTaskAdapterAuthority:
    settings = CrossRunSettings.from_dict(
        load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
    )
    return CanonicalTaskAdapterAuthority(settings.expert.task_adapters.active_authority)


def test_configured_authority_verifies_package_and_activation(tmp_path):
    authority = _authority()
    manifest = _manifest(_ADAPTER_SOURCE)
    store = _store(tmp_path, authority=authority)

    verified = store.publish(_package(tmp_path, manifest))
    activation = store.activate(
        scope_contract_id=manifest.scope_contract_id,
        task_family_id=manifest.task_family_id,
        task_adapter_id=manifest.task_adapter_id,
        verification_receipt_id=verified.verification_receipt.verification_receipt_id,
        expected_activation_id=None,
        authority_envelope=_activation_envelope(verified),
    )

    assert authority.authority_id == "kapso_task_adapter_authority"
    assert (
        store.resolve_active_binding(
            scope_contract_id=manifest.scope_contract_id,
            task_family_id=manifest.task_family_id,
            task_adapter_id=manifest.task_adapter_id,
        ).activation
        == activation
    )


def test_configured_authority_rejects_substituted_envelopes(tmp_path):
    authority = _authority()
    manifest = _manifest(_ADAPTER_SOURCE)
    package = _package(tmp_path, manifest)
    store = _store(tmp_path, authority=authority)
    verified = store.publish(package)
    activation = store.activate(
        scope_contract_id=manifest.scope_contract_id,
        task_family_id=manifest.task_family_id,
        task_adapter_id=manifest.task_adapter_id,
        verification_receipt_id=verified.verification_receipt.verification_receipt_id,
        expected_activation_id=None,
        authority_envelope=_activation_envelope(verified),
    )

    with pytest.raises(TaskAdapterAuthorityError, match="proof differs"):
        authority.verify_package(
            manifest=manifest,
            source_extraction_receipt=verified.source_extraction_receipt,
            proof_objects={
                key: canonical_json_bytes({"substituted": key})
                for key in package.proof_objects
            },
            publisher_verification=package.publisher_verification,
        )
    with pytest.raises(TaskAdapterAuthorityError, match="activation differs"):
        authority.verify_activation(
            activation=activation,
            authority_envelope=canonical_json_bytes({"substituted": True}),
        )
