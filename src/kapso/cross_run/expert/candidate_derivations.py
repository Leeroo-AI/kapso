"""Closed provenance records for expert candidate construction."""

from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import ClassVar, Mapping

from kapso.cross_run.canonical import (
    require_content_id,
    require_identifier,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    CodingAgentWorkspaceDelta,
    ExpertCandidateOperationRecord,
    StrictContract,
)
from kapso.cross_run.expert.proposal_contract import ExpertCandidateAncestorInput
from kapso.cross_run.expert.triggers import (
    ExpertEvolutionTriggerDecision,
    ExpertTriggerEvidencePacket,
)


class ExpertCandidateDerivationError(ValueError):
    """Candidate derivation provenance is incomplete or inconsistent."""


def _require_namespaced_id(value: str, namespace: str, name: str) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise ExpertCandidateDerivationError(f"{name} uses the wrong namespace")


@dataclass(frozen=True)
class ExpertAgentProposalDerivationRecord(StrictContract):
    """Stable identity of one complete coding-agent proposal derivation."""

    derivation_id: str
    trigger_evidence_packet_id: str
    trigger_decision_id: str
    operation_record_id: str
    workspace_delta_id: str
    ancestor_candidate_ids: tuple[str, ...]
    origin_principal_ids: tuple[str, ...]
    source_dependency_ids: tuple[str, ...]
    operation_artifact_checksums: Mapping[str, str]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-agent-proposal-derivation"
    IDENTITY_FIELD: ClassVar[str] = "derivation_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.trigger_evidence_packet_id,
                "expert-trigger-evidence-packet",
                "agent derivation trigger packet",
            ),
            (
                self.trigger_decision_id,
                "expert-trigger-decision",
                "agent derivation trigger decision",
            ),
            (
                self.operation_record_id,
                "expert-candidate-operation",
                "agent derivation operation",
            ),
            (
                self.workspace_delta_id,
                "coding-agent-workspace-delta",
                "agent derivation workspace delta",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        for values, name in (
            (self.ancestor_candidate_ids, "agent derivation ancestors"),
            (self.source_dependency_ids, "agent derivation source dependencies"),
        ):
            if values != tuple(sorted(set(values))):
                raise ExpertCandidateDerivationError(
                    f"{name} must be sorted and unique"
                )
            for value in values:
                require_content_id(value, name)
        if not self.source_dependency_ids:
            raise ExpertCandidateDerivationError(
                "agent derivation source dependencies must not be empty"
            )
        if not self.origin_principal_ids or self.origin_principal_ids != tuple(
            sorted(set(self.origin_principal_ids))
        ):
            raise ExpertCandidateDerivationError(
                "agent derivation origin principals must be canonical and non-empty"
            )
        for principal_id in self.origin_principal_ids:
            require_identifier(principal_id, "agent derivation origin principal")
        if not self.operation_artifact_checksums:
            raise ExpertCandidateDerivationError(
                "agent derivation artifact closure must not be empty"
            )
        for name, digest in self.operation_artifact_checksums.items():
            if (
                not name
                or not isinstance(digest, str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
            ):
                raise ExpertCandidateDerivationError(
                    "agent derivation artifact checksum is invalid"
                )


@dataclass(frozen=True)
class ExpertAgentProposalDerivation:
    """Runtime closure for one agent-authored candidate derivation."""

    record: ExpertAgentProposalDerivationRecord
    trigger_packet: ExpertTriggerEvidencePacket
    trigger_decision: ExpertEvolutionTriggerDecision
    operation: ExpertCandidateOperationRecord
    workspace_delta: CodingAgentWorkspaceDelta
    operation_artifacts: Mapping[str, bytes]
    ancestor_inputs: tuple[ExpertCandidateAncestorInput, ...]

    def __post_init__(self) -> None:
        if (
            type(self.record) is not ExpertAgentProposalDerivationRecord
            or type(self.trigger_packet) is not ExpertTriggerEvidencePacket
            or type(self.trigger_decision) is not ExpertEvolutionTriggerDecision
            or type(self.operation) is not ExpertCandidateOperationRecord
            or type(self.workspace_delta) is not CodingAgentWorkspaceDelta
            or type(self.ancestor_inputs) is not tuple
            or any(
                type(ancestor) is not ExpertCandidateAncestorInput
                for ancestor in self.ancestor_inputs
            )
            or not isinstance(self.operation_artifacts, Mapping)
        ):
            raise ExpertCandidateDerivationError(
                "agent proposal derivation requires exact typed authorities"
            )
        frozen_artifacts = MappingProxyType(dict(self.operation_artifacts))
        object.__setattr__(self, "operation_artifacts", frozen_artifacts)
        expected_artifact_checksums = {
            name: tree_or_blob_digest(payload)
            for name, payload in sorted(frozen_artifacts.items())
        }
        expected_ancestor_ids = tuple(
            ancestor.manifest.candidate_id for ancestor in self.ancestor_inputs
        )
        if (
            self.record.trigger_evidence_packet_id
            != self.trigger_packet.evidence_packet_id
            or self.record.trigger_decision_id
            != self.trigger_decision.trigger_decision_id
            or self.record.operation_record_id != self.operation.operation_record_id
            or self.record.workspace_delta_id != self.workspace_delta.workspace_delta_id
            or self.record.ancestor_candidate_ids != expected_ancestor_ids
            or self.record.origin_principal_ids
            != (self.operation.proposer_authority.principal_id,)
            or dict(self.record.operation_artifact_checksums)
            != expected_artifact_checksums
        ):
            raise ExpertCandidateDerivationError(
                "agent proposal derivation record differs from its exact closure"
            )
