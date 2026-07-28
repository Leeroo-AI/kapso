"""Configured evaluator trust roots authenticate persisted result envelopes."""

from __future__ import annotations

import base64
from dataclasses import replace

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kapso.core.config import load_effective_config
from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.contracts import (
    ExpertEvaluatorAttestation,
    ExpertEvaluatorAttestationEnvelope,
)
from kapso.cross_run.expert.attestation import ConfiguredExpertAttestationVerifier
from kapso.cross_run.settings import ExpertEvaluatorTrustRootSettings

_CONFIG_PATH = "src/kapso/config.yaml"


def test_configured_ed25519_attestation_verifies_exact_issuer_and_bytes():
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    configured = load_effective_config(_CONFIG_PATH, "GENERIC").cross_run
    validation = replace(
        configured.expert.validation,
        evaluator_trust_roots=(
            ExpertEvaluatorTrustRootSettings(
                trust_root_id="synthetic_evaluator_root",
                issuer_ids=("expert_contract_evaluator",),
                public_key_base64=base64.b64encode(public_key).decode("ascii"),
            ),
        ),
    )
    attestation = ExpertEvaluatorAttestation.mint(
        evaluator_run_id=content_id("expert-evaluator-run", {"run": 1}),
        issuer_id="expert_contract_evaluator",
        trust_root_id="synthetic_evaluator_root",
        predicate_digest=tree_or_blob_digest(b"synthetic evaluator result"),
    )
    envelope = ExpertEvaluatorAttestationEnvelope(
        attestation=attestation,
        signature=base64.b64encode(
            private_key.sign(attestation.to_json_bytes())
        ).decode("ascii"),
    )

    ConfiguredExpertAttestationVerifier(validation).verify(envelope)

    substituted = ExpertEvaluatorAttestationEnvelope(
        attestation=attestation,
        signature=base64.b64encode(b"0" * 64).decode("ascii"),
    )
    with pytest.raises(InvalidSignature):
        ConfiguredExpertAttestationVerifier(validation).verify(substituted)

    assert (
        validation.evaluator_trust_root_id("expert_contract_evaluator")
        == "synthetic_evaluator_root"
    )
    assert validation.evaluator_trust_root_id("expert_code_evaluator") is None
