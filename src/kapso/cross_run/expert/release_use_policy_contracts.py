"""Authenticated current policy observations for expert release use."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

from kapso.cross_run.canonical import require_content_id, require_identifier
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.record_contracts import ExpertReleaseUseRevocation

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_REPOSITORY_PATTERN = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?/[A-Za-z0-9._-]+$"
)


class ExpertReleaseUsePolicyContractError(ValueError):
    """A release-use policy observation is incomplete or inconsistent."""


def _require_namespaced_id(value: str, namespace: str, name: str) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise ExpertReleaseUsePolicyContractError(f"{name} uses the wrong namespace")


@dataclass(frozen=True)
class ExpertReleaseUsePolicyObservation(StrictContract):
    """Exact release-use matches from one authenticated knowledge CURRENT."""

    observation_id: str
    scope_id: str
    scope_contract_id: str
    scope_repository_binding_hash: str
    repository_full_name: str
    repository_node_id: str
    knowledge_snapshot_id: str
    catalog_generation: int
    knowledge_publication_id: str
    current_pointer_digest: str
    authority_commit_sha: str
    release_attestation_ref: str
    checked_release_ids: tuple[str, ...]
    matched_revocations: tuple[ExpertReleaseUseRevocation, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-release-use-policy-observation"
    IDENTITY_FIELD: ClassVar[str] = "observation_id"

    def _validate(self) -> None:
        require_identifier(self.scope_id, "release-use policy scope_id")
        _require_namespaced_id(
            self.scope_contract_id,
            "expert-scope-contract",
            "release-use policy scope_contract_id",
        )
        if _DIGEST_PATTERN.fullmatch(self.scope_repository_binding_hash) is None:
            raise ExpertReleaseUsePolicyContractError(
                "release-use policy repository binding is invalid"
            )
        if _REPOSITORY_PATTERN.fullmatch(self.repository_full_name) is None:
            raise ExpertReleaseUsePolicyContractError(
                "release-use policy repository identity is invalid"
            )
        require_identifier(
            self.repository_node_id,
            "release-use policy repository_node_id",
        )
        _require_namespaced_id(
            self.knowledge_snapshot_id,
            "knowledge-snapshot",
            "release-use policy knowledge_snapshot_id",
        )
        if type(self.catalog_generation) is not int or self.catalog_generation < 0:
            raise ExpertReleaseUsePolicyContractError(
                "release-use policy catalog_generation must be non-negative"
            )
        if self.catalog_generation == 0 and self.matched_revocations:
            raise ExpertReleaseUsePolicyContractError(
                "generation-zero release-use policy cannot contain matches"
            )
        _require_namespaced_id(
            self.knowledge_publication_id,
            "github-publication",
            "release-use policy knowledge_publication_id",
        )
        if _DIGEST_PATTERN.fullmatch(self.current_pointer_digest) is None:
            raise ExpertReleaseUsePolicyContractError(
                "release-use policy current pointer digest is invalid"
            )
        if _COMMIT_PATTERN.fullmatch(self.authority_commit_sha) is None:
            raise ExpertReleaseUsePolicyContractError(
                "release-use policy authority commit is invalid"
            )
        if (
            not isinstance(self.release_attestation_ref, str)
            or not self.release_attestation_ref.strip()
        ):
            raise ExpertReleaseUsePolicyContractError(
                "release-use policy release attestation is required"
            )
        if self.checked_release_ids != tuple(sorted(set(self.checked_release_ids))):
            raise ExpertReleaseUsePolicyContractError(
                "checked release IDs must be sorted and unique"
            )
        for release_id in self.checked_release_ids:
            _require_namespaced_id(
                release_id,
                "expert-base-release",
                "checked release ID",
            )
        revocation_ids = tuple(
            revocation.revocation_id for revocation in self.matched_revocations
        )
        if revocation_ids != tuple(sorted(set(revocation_ids))):
            raise ExpertReleaseUsePolicyContractError(
                "matched release-use revocations must be sorted and unique"
            )
        for revocation in self.matched_revocations:
            if (
                revocation.scope_id != self.scope_id
                or revocation.scope_contract_id != self.scope_contract_id
                or revocation.release_id not in self.checked_release_ids
            ):
                raise ExpertReleaseUsePolicyContractError(
                    "matched release-use revocation was not checked in this scope"
                )

    @property
    def matched_release_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted({revocation.release_id for revocation in self.matched_revocations})
        )


__all__ = [
    "ExpertReleaseUsePolicyContractError",
    "ExpertReleaseUsePolicyObservation",
]
