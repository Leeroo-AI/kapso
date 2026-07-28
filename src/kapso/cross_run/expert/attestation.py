"""Configured Ed25519 verification for external expert evaluator results."""

from __future__ import annotations

import base64

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from kapso.cross_run.contracts import ExpertEvaluatorAttestationEnvelope
from kapso.cross_run.expert.validation import ExpertValidationError
from kapso.cross_run.settings import ExpertValidationSettings


class ConfiguredExpertAttestationVerifier:
    """Verify evaluator attestations against config-pinned public trust roots."""

    def __init__(self, settings: ExpertValidationSettings) -> None:
        if type(settings) is not ExpertValidationSettings:
            raise ExpertValidationError(
                "expert attestation verifier requires validation settings"
            )
        self.settings = settings
        self._roots_by_id = {
            root.trust_root_id: root for root in settings.evaluator_trust_roots
        }

    def verify(self, envelope: ExpertEvaluatorAttestationEnvelope) -> None:
        if type(envelope) is not ExpertEvaluatorAttestationEnvelope:
            raise ExpertValidationError(
                "expert evaluator attestation uses another contract"
            )
        attestation = envelope.attestation
        if attestation.trust_root_id is None:
            raise ExpertValidationError(
                "expert evaluator attestation has no trust root"
            )
        root = self._roots_by_id.get(attestation.trust_root_id)
        if root is None or attestation.issuer_id not in root.issuer_ids:
            raise ExpertValidationError(
                "expert evaluator attestation issuer is not configured"
            )
        public_key_bytes = base64.b64decode(root.public_key_base64, validate=True)
        signature = base64.b64decode(envelope.signature, validate=True)
        if len(public_key_bytes) != 32 or len(signature) != 64:
            raise ExpertValidationError(
                "expert evaluator Ed25519 key or signature length is invalid"
            )
        Ed25519PublicKey.from_public_bytes(public_key_bytes).verify(
            signature,
            attestation.to_json_bytes(),
        )
