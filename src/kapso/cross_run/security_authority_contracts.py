"""Neutral adapter-verifier and security-denylist observations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

from kapso.cross_run.canonical import (
    content_id,
    require_content_id,
    require_identifier,
)
from kapso.cross_run.contracts import StrictContract

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class SecurityAuthorityContractError(ValueError):
    """A shared verifier or denylist observation is invalid."""


def _require_sorted_content_ids(
    values: tuple[str, ...],
    name: str,
    *,
    allow_empty: bool = False,
) -> None:
    if (not values and not allow_empty) or values != tuple(sorted(set(values))):
        raise SecurityAuthorityContractError(f"{name} must be sorted and unique")
    for value in values:
        require_content_id(value, name)


@dataclass(frozen=True)
class TaskAdapterTrustObservation(StrictContract):
    observation_id: str
    task_adapter_manifest_id: str
    verification_receipt_id: str
    verifier_id: str
    verifier_version: str
    dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "task-adapter-trust-observation"
    IDENTITY_FIELD: ClassVar[str] = "observation_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.task_adapter_manifest_id,
                "task-adapter-manifest",
                "adapter trust manifest",
            ),
            (
                self.verification_receipt_id,
                "task-adapter-verification-receipt",
                "adapter trust verification receipt",
            ),
        ):
            require_content_id(value, name)
            if value.split(":sha256:", 1)[0] != namespace:
                raise SecurityAuthorityContractError(f"{name} uses the wrong namespace")
        for value, name in (
            (self.verifier_id, "adapter trust verifier ID"),
            (self.verifier_version, "adapter trust verifier version"),
        ):
            require_identifier(value, name)
        _require_sorted_content_ids(
            self.dependency_ids,
            "adapter trust dependencies",
        )
        if self.verification_receipt_id not in self.dependency_ids:
            raise SecurityAuthorityContractError(
                "adapter trust omits its verification receipt"
            )

    @property
    def verifier_authority_subject_id(self) -> str:
        return content_id(
            "task-adapter-verifier-authority",
            {
                "verifier_id": self.verifier_id,
                "verifier_version": self.verifier_version,
            },
        )


@dataclass(frozen=True)
class SecurityDenylistObservation(StrictContract):
    observation_id: str
    scope_id: str
    scope_contract_id: str
    scope_repository_binding_hash: str
    snapshot_id: str
    generation: int
    publication_id: str
    repository_full_name: str
    repository_node_id: str
    pointer_digest: str
    authority_commit_sha: str
    release_attestation_ref: str
    checked_subject_ids: tuple[str, ...]
    denied_subject_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "security-denylist-observation"
    IDENTITY_FIELD: ClassVar[str] = "observation_id"

    def _validate(self) -> None:
        require_identifier(self.scope_id, "denylist observation scope")
        require_content_id(
            self.scope_contract_id,
            "denylist observation scope contract",
        )
        if _DIGEST_PATTERN.fullmatch(self.scope_repository_binding_hash) is None:
            raise SecurityAuthorityContractError(
                "denylist observation repository binding is invalid"
            )
        require_content_id(self.snapshot_id, "denylist observation snapshot")
        if self.snapshot_id.split(":sha256:", 1)[0] != "security-denylist-snapshot":
            raise SecurityAuthorityContractError(
                "denylist observation snapshot uses the wrong namespace"
            )
        if type(self.generation) is not int or self.generation < 0:
            raise SecurityAuthorityContractError(
                "denylist observation generation must be non-negative"
            )
        require_content_id(
            self.publication_id,
            "denylist observation publication",
        )
        if self.publication_id.split(":sha256:", 1)[0] != "github-publication":
            raise SecurityAuthorityContractError(
                "denylist observation publication uses the wrong namespace"
            )
        if _REPOSITORY_PATTERN.fullmatch(self.repository_full_name) is None:
            raise SecurityAuthorityContractError(
                "denylist observation repository identity is invalid"
            )
        require_identifier(
            self.repository_node_id,
            "denylist observation repository node",
        )
        if _DIGEST_PATTERN.fullmatch(self.pointer_digest) is None:
            raise SecurityAuthorityContractError(
                "denylist observation pointer digest is invalid"
            )
        if _COMMIT_PATTERN.fullmatch(self.authority_commit_sha) is None:
            raise SecurityAuthorityContractError(
                "denylist observation authority commit is invalid"
            )
        if not isinstance(self.release_attestation_ref, str) or not (
            self.release_attestation_ref.strip()
        ):
            raise SecurityAuthorityContractError(
                "denylist observation release attestation is required"
            )
        _require_sorted_content_ids(
            self.checked_subject_ids,
            "denylist observation checked subjects",
        )
        _require_sorted_content_ids(
            self.denied_subject_ids,
            "denylist observation denied subjects",
            allow_empty=True,
        )
        if not set(self.denied_subject_ids).issubset(self.checked_subject_ids):
            raise SecurityAuthorityContractError(
                "denylist observation denied subjects were not checked"
            )
