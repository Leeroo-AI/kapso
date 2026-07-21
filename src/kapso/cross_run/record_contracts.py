"""Dependency-pure immutable contracts stored by capture and catalog modules."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, ClassVar, Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    normalize_utc_timestamp,
    parse_json_bytes,
    require_content_id,
    require_identifier,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    CodingAgentOperationReceipt,
    ContractValidationError,
    EpisodeEvaluationStatus,
    ExecutionStatus,
    MissingReferenceError,
    StrictContract,
)

EXECUTION_REVISION_EVENT_SCHEMA = "kapso.execution_revision_event.v1"
SANITATION_REPORT_SCHEMA = "kapso.sanitation_report.v1"
SANITATION_SCANNER_VERSION = "kapso.deterministic_text_scanner.v1"

_FINDING_CODES = frozenset(
    {
        "assigned_secret",
        "aws_access_key",
        "credential_url",
        "denied_path",
        "file_too_large",
        "forbidden_artifact_class",
        "github_token",
        "invalid_utf8",
        "nul_byte",
        "openai_key",
        "private_key",
        "unapproved_spdx_license",
        "unclassified_license",
    }
)
_EXCLUSION_REASONS = frozenset(
    {
        "artifact_class",
        "denied_path",
        "file_too_large",
        "non_regular_file",
        "unknown_size",
    }
)
_TAINT_SOURCES = frozenset(
    {"unclassified_license"} | {f"excluded_{reason}" for reason in _EXCLUSION_REASONS}
)


class CatalogAgentOperationError(ValueError):
    """A catalog agent workspace or operation artifact set is invalid."""


class BundleProjectionError(ValueError):
    """A sanitized bundle cannot be projected without changing its meaning."""


def _require_sorted_content_ids(values: tuple[str, ...], name: str) -> None:
    if not values or values != tuple(sorted(set(values))):
        raise ContractValidationError(f"{name} must be non-empty, sorted, and unique")
    for value in values:
        require_content_id(value, name)


def catalog_agent_operation_id(preimage: Mapping[str, Any]) -> str:
    digest = tree_or_blob_digest(canonical_json_bytes(preimage))[7:]
    return f"agent_call_{digest[:32]}"


@dataclass(frozen=True)
class CatalogAgentOperationRecord(StrictContract):
    """Exact model input/output binding behind framework-minted catalog facts."""

    operation_record_id: str
    operation_kind: str
    operation_receipt_id: str
    operation_preimage: Mapping[str, Any]
    final_output: str
    produced_object_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "catalog-agent-operation"
    IDENTITY_FIELD: ClassVar[str] = "operation_record_id"

    def _validate(self) -> None:
        if self.operation_kind not in {"claim_proposal", "catalog_review"}:
            raise ContractValidationError("catalog agent operation kind is invalid")
        require_content_id(self.operation_receipt_id, "operation_receipt_id")
        if not isinstance(self.operation_preimage, Mapping):
            raise ContractValidationError("operation preimage must be an object")
        if not isinstance(self.final_output, str) or not self.final_output.strip():
            raise ContractValidationError("catalog agent final output is empty")
        parse_json_bytes(self.final_output.encode("utf-8"))
        if self.produced_object_ids != tuple(sorted(set(self.produced_object_ids))):
            raise ContractValidationError(
                "produced object IDs must be sorted and unique"
            )
        for object_id in self.produced_object_ids:
            require_content_id(object_id, "produced_object_ids")

    @property
    def packet_payload(self) -> Mapping[str, Any]:
        packet = self.operation_preimage.get("packet")
        if not isinstance(packet, Mapping):
            raise CatalogAgentOperationError("operation preimage packet is absent")
        return packet

    def validate_receipt(self, receipt: CodingAgentOperationReceipt) -> None:
        if receipt.operation_receipt_id != self.operation_receipt_id:
            raise CatalogAgentOperationError("operation receipt identity differs")
        require_identifier(receipt.operation_id, "operation_id")
        if catalog_agent_operation_id(self.operation_preimage) != receipt.operation_id:
            raise CatalogAgentOperationError(
                "operation preimage does not match the receipt operation"
            )
        if tree_or_blob_digest(self.final_output.encode("utf-8")) != (
            receipt.artifact_checksums["final.json"]
        ):
            raise CatalogAgentOperationError(
                "operation final output does not match its receipt checksum"
            )


@dataclass(frozen=True)
class BundleProjectionManifest(StrictContract):
    """Exact stored closure produced by one deterministic bundle projection."""

    projection_manifest_id: str
    source_bundle_id: str
    sanitation_report_id: str
    episode_ids: tuple[str, ...]
    prior_idea_ids: tuple[str, ...]
    derivation_object_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "bundle-projection-manifest"
    IDENTITY_FIELD: ClassVar[str] = "projection_manifest_id"

    def _validate(self) -> None:
        require_content_id(self.source_bundle_id, "source_bundle_id")
        require_content_id(self.sanitation_report_id, "sanitation_report_id")
        for name in ("episode_ids", "prior_idea_ids", "derivation_object_ids"):
            values = getattr(self, name)
            if values != tuple(sorted(set(values))):
                raise BundleProjectionError(f"{name} must be sorted and unique")
            for value in values:
                require_content_id(value, name)
        if set(self.episode_ids) & set(self.prior_idea_ids):
            raise BundleProjectionError("episode and prior-idea identities overlap")


@dataclass(frozen=True)
class ClaimEvidenceClosure(StrictContract):
    """Exact episode universe classified by one claim-proposal operation."""

    claim_evidence_closure_id: str
    claim_revision_id: str
    evaluated_episode_ids: tuple[str, ...]
    supporting_episode_ids: tuple[str, ...]
    contradicting_episode_ids: tuple[str, ...]
    evidence_assessments: tuple[Mapping[str, str], ...]
    proposer_operation_receipt_id: str
    packet_digest: str

    CONTENT_NAMESPACE: ClassVar[str] = "claim-evidence-closure"
    IDENTITY_FIELD: ClassVar[str] = "claim_evidence_closure_id"

    def _validate(self) -> None:
        require_content_id(self.claim_revision_id, "closure claim_revision_id")
        require_content_id(
            self.proposer_operation_receipt_id,
            "closure proposer_operation_receipt_id",
        )
        _require_sorted_content_ids(
            self.evaluated_episode_ids,
            "closure evaluated_episode_ids",
        )
        for name in ("supporting_episode_ids", "contradicting_episode_ids"):
            values = getattr(self, name)
            if values:
                if values != tuple(sorted(set(values))):
                    raise ContractValidationError(
                        f"closure {name} must be sorted and unique"
                    )
                for value in values:
                    require_content_id(value, f"closure {name}")
        support = set(self.supporting_episode_ids)
        contradictions = set(self.contradicting_episode_ids)
        if support & contradictions:
            raise ContractValidationError(
                "closure support and contradiction sets must be disjoint"
            )
        if not support and not contradictions:
            raise ContractValidationError(
                "closure must classify support or contradiction evidence"
            )
        if not (support | contradictions).issubset(self.evaluated_episode_ids):
            raise MissingReferenceError(
                "closure evidence classifications leave the evaluated universe"
            )
        assessment_ids: list[str] = []
        assessment_relationships: dict[str, str] = {}
        for assessment in self.evidence_assessments:
            if set(assessment) != {"episode_id", "rationale", "relationship"}:
                raise ContractValidationError(
                    "closure evidence assessment fields are invalid"
                )
            if any(
                not isinstance(value, str) or not value.strip()
                for value in assessment.values()
            ):
                raise ContractValidationError(
                    "closure evidence assessment values must be non-empty text"
                )
            episode_id = assessment["episode_id"]
            require_content_id(episode_id, "assessment episode_id")
            relationship = assessment["relationship"]
            if relationship not in {"support", "contradiction", "not_applicable"}:
                raise ContractValidationError(
                    "closure evidence assessment relationship is invalid"
                )
            assessment_ids.append(episode_id)
            assessment_relationships[episode_id] = relationship
        if tuple(assessment_ids) != tuple(sorted(set(assessment_ids))):
            raise ContractValidationError(
                "closure evidence assessments must be sorted and unique"
            )
        if tuple(assessment_ids) != self.evaluated_episode_ids:
            raise MissingReferenceError(
                "closure must assess every evaluated episode exactly once"
            )
        if self.supporting_episode_ids != tuple(
            episode_id
            for episode_id in assessment_ids
            if assessment_relationships[episode_id] == "support"
        ) or self.contradicting_episode_ids != tuple(
            episode_id
            for episode_id in assessment_ids
            if assessment_relationships[episode_id] == "contradiction"
        ):
            raise ContractValidationError(
                "closure classifications differ from evidence assessments"
            )
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.packet_digest) is None:
            raise ContractValidationError("closure packet_digest must be sha256")

    @property
    def not_applicable_episode_ids(self) -> tuple[str, ...]:
        classified = set(self.supporting_episode_ids)
        classified.update(self.contradicting_episode_ids)
        return tuple(sorted(set(self.evaluated_episode_ids) - classified))


@dataclass(frozen=True)
class CatalogRevocation(StrictContract):
    """Immutable direct withdrawal of trust from one catalog proof object."""

    revocation_id: str
    subject_id: str
    reason_code: str
    rationale: str
    exact_evidence_refs: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "catalog-revocation"
    IDENTITY_FIELD: ClassVar[str] = "revocation_id"

    def _validate(self) -> None:
        require_content_id(self.subject_id, "revocation subject_id")
        require_identifier(self.reason_code, "revocation reason_code")
        if not self.rationale.strip():
            raise ContractValidationError("revocation rationale must not be empty")
        _require_sorted_content_ids(
            self.exact_evidence_refs,
            "revocation exact_evidence_refs",
        )


@dataclass(frozen=True)
class CatalogTaint(StrictContract):
    """Immutable finding that one proof object is contaminated by another."""

    taint_id: str
    subject_id: str
    source_subject_id: str
    reason_code: str
    rationale: str
    exact_evidence_refs: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "catalog-taint"
    IDENTITY_FIELD: ClassVar[str] = "taint_id"

    def _validate(self) -> None:
        require_content_id(self.subject_id, "taint subject_id")
        require_content_id(self.source_subject_id, "taint source_subject_id")
        if self.subject_id == self.source_subject_id:
            raise ContractValidationError(
                "taint subject and source must be distinct proof objects"
            )
        require_identifier(self.reason_code, "taint reason_code")
        if not self.rationale.strip():
            raise ContractValidationError("taint rationale must not be empty")
        _require_sorted_content_ids(
            self.exact_evidence_refs,
            "taint exact_evidence_refs",
        )


@dataclass(frozen=True)
class SanitationReport(StrictContract):
    report_id: str
    schema: str
    capture_manifest_id: str
    scope_id: str
    task_family_id: str
    policy_version: str
    policy_fingerprint: str
    scanner_version: str
    status: str
    findings: tuple[Mapping[str, Any], ...]
    excluded_paths: tuple[Mapping[str, Any], ...]
    taint_sources: tuple[str, ...]
    admitted_refs: Mapping[str, str]

    CONTENT_NAMESPACE: ClassVar[str] = "sanitation-report"
    IDENTITY_FIELD: ClassVar[str] = "report_id"

    def _validate(self) -> None:
        if self.schema != SANITATION_REPORT_SCHEMA:
            raise ValueError("sanitation report schema is incompatible")
        require_content_id(self.capture_manifest_id, "capture_manifest_id")
        require_identifier(self.scope_id, "scope_id")
        require_identifier(self.task_family_id, "task_family_id")
        if self.status not in {"admitted", "rejected"}:
            raise ValueError("sanitation report status is invalid")
        if (
            not self.policy_version
            or self.scanner_version != SANITATION_SCANNER_VERSION
        ):
            raise ValueError("sanitation policy/scanner identity is invalid")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.policy_fingerprint) is None:
            raise ValueError("sanitation policy fingerprint is invalid")
        if self.taint_sources != tuple(sorted(set(self.taint_sources))):
            raise ValueError("sanitation taint sources must be sorted and unique")
        for taint_source in self.taint_sources:
            if taint_source not in _TAINT_SOURCES:
                raise ValueError("sanitation taint source is invalid")
        finding_keys = {"code", "evidence_digest", "path", "severity"}
        for finding in self.findings:
            if set(finding) != finding_keys or any(
                not isinstance(value, str) for value in finding.values()
            ):
                raise ValueError("sanitation finding shape is invalid")
            if finding["code"] not in _FINDING_CODES:
                raise ValueError("sanitation finding code is invalid")
            if re.fullmatch(r"sha256:[0-9a-f]{64}", finding["evidence_digest"]) is None:
                raise ValueError("sanitation finding evidence digest is invalid")
            self._require_relative_path(finding["path"], "sanitation finding path")
            if finding["severity"] not in {"notice", "reject"}:
                raise ValueError("sanitation finding severity is invalid")
        expected_findings = tuple(
            sorted(
                self.findings,
                key=lambda item: (
                    item["path"],
                    item["code"],
                    item["evidence_digest"],
                    item["severity"],
                ),
            )
        )
        if self.findings != expected_findings or len(
            {tuple(sorted(item.items())) for item in self.findings}
        ) != len(self.findings):
            raise ValueError("sanitation findings must be sorted and unique")
        exclusion_keys = {"path", "reason"}
        for exclusion in self.excluded_paths:
            if set(exclusion) != exclusion_keys or any(
                not isinstance(value, str) for value in exclusion.values()
            ):
                raise ValueError("sanitation exclusion shape is invalid")
            self._require_relative_path(exclusion["path"], "sanitation exclusion path")
            if exclusion["reason"] not in _EXCLUSION_REASONS:
                raise ValueError("sanitation exclusion reason is invalid")
        expected_exclusions = tuple(
            sorted(
                self.excluded_paths,
                key=lambda item: (item["path"], item["reason"]),
            )
        )
        if self.excluded_paths != expected_exclusions or len(
            {tuple(sorted(item.items())) for item in self.excluded_paths}
        ) != len(self.excluded_paths):
            raise ValueError("sanitation exclusions must be sorted and unique")
        if self.status == "admitted" and any(
            finding["severity"] == "reject" for finding in self.findings
        ):
            raise ValueError("admitted sanitation report contains a rejection")
        if self.status == "rejected" and not any(
            finding["severity"] == "reject" for finding in self.findings
        ):
            raise ValueError("rejected sanitation report lacks a rejection finding")
        if self.status == "rejected" and self.admitted_refs:
            raise ValueError("rejected sanitation report contains admitted refs")
        for path, digest in self.admitted_refs.items():
            if not isinstance(path, str) or not isinstance(digest, str):
                raise ValueError("sanitation admitted ref is invalid")
            self._require_relative_path(path, "sanitation admitted ref")
            if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
                raise ValueError("sanitation admitted ref is invalid")

    @staticmethod
    def _require_relative_path(path: str, name: str) -> None:
        normalized = PurePosixPath(path)
        if (
            not path
            or normalized == PurePosixPath(".")
            or normalized.is_absolute()
            or ".." in normalized.parts
            or normalized.as_posix() != path
        ):
            raise ValueError(f"{name} is invalid")


@dataclass(frozen=True)
class ExecutionRevisionEvent(StrictContract):
    """One immutable projection of one executed node revision."""

    event_id: str
    schema: str
    run_id: str
    campaign_id: str
    node_id: int
    execution_revision: int
    idea_id: str | None
    selection_batch_id: str | None
    parent_node_id: int | None
    started_at: str
    recorded_at: str
    execution_status: ExecutionStatus
    evaluation_status: EpisodeEvaluationStatus
    evaluator_fingerprint_ids: tuple[str, ...]
    measurements: Mapping[str, float]
    feedback: str
    technical_difficulties: str
    artifact_refs: Mapping[str, str]
    projection: Mapping[str, Any]

    CONTENT_NAMESPACE: ClassVar[str] = "execution-revision-event"
    IDENTITY_FIELD: ClassVar[str] = "event_id"

    def _validate(self) -> None:
        if self.schema != EXECUTION_REVISION_EVENT_SCHEMA:
            raise ValueError("execution revision event schema is incompatible")
        require_identifier(self.run_id, "run_id")
        require_identifier(self.campaign_id, "campaign_id")
        for value, name in (
            (self.node_id, "node_id"),
            (self.execution_revision, "execution_revision"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"execution revision {name} must be non-negative")
        if self.parent_node_id is not None and (
            type(self.parent_node_id) is not int or self.parent_node_id < 0
        ):
            raise ValueError("execution revision parent_node_id is invalid")
        if (self.idea_id is None) != (self.selection_batch_id is None):
            raise ValueError("execution revision idea and batch must appear together")
        if self.idea_id is not None:
            require_identifier(self.idea_id, "idea_id")
            require_identifier(self.selection_batch_id, "selection_batch_id")
        normalize_utc_timestamp(self.started_at, "started_at")
        normalize_utc_timestamp(self.recorded_at, "recorded_at")
        if not isinstance(self.execution_status, ExecutionStatus) or not isinstance(
            self.evaluation_status, EpisodeEvaluationStatus
        ):
            raise ValueError("execution revision statuses are invalid")
        if not isinstance(self.feedback, str) or not isinstance(
            self.technical_difficulties, str
        ):
            raise ValueError("execution revision observations must be strings")
        if self.evaluator_fingerprint_ids != tuple(
            sorted(set(self.evaluator_fingerprint_ids))
        ):
            raise ValueError("evaluator fingerprints must be sorted and unique")
        for evaluator_id in self.evaluator_fingerprint_ids:
            require_identifier(evaluator_id, "evaluator_fingerprint_id")
        if not isinstance(self.artifact_refs, Mapping) or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in self.artifact_refs.items()
        ):
            raise ValueError("execution revision artifact refs must be non-empty")
        if not isinstance(self.measurements, Mapping) or any(
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for key, value in self.measurements.items()
        ):
            raise ValueError("execution revision measurements are invalid")
        projection = self.projection
        if not isinstance(projection, Mapping):
            raise ValueError("execution revision projection must be an object")
        expected_links = {
            "node_id": self.node_id,
            "execution_revision": self.execution_revision,
            "idea_id": self.idea_id,
            "selection_batch_id": self.selection_batch_id,
            "parent_node_id": self.parent_node_id,
            "feedback": self.feedback,
            "technical_difficulties": self.technical_difficulties,
        }
        conflicts = tuple(
            sorted(
                name
                for name, expected in expected_links.items()
                if projection.get(name) != expected
            )
        )
        if conflicts:
            raise ValueError(
                f"execution revision projection conflicts with event: {conflicts}"
            )
        had_error = projection.get("had_error")
        evaluation_valid = projection.get("evaluation_valid")
        if type(had_error) is not bool or type(evaluation_valid) is not bool:
            raise ValueError("execution revision projection states are invalid")
        if had_error and self.execution_status is ExecutionStatus.COMPLETED:
            raise ValueError("failed projection cannot have completed execution")
        if not had_error and self.execution_status is not ExecutionStatus.COMPLETED:
            raise ValueError("successful projection must have completed execution")
        has_measurement = projection.get("raw_score") is not None and bool(
            projection.get("evaluation_attempts")
        )
        expected_evaluation_status = (
            EpisodeEvaluationStatus.NOT_RUN
            if had_error
            else (
                EpisodeEvaluationStatus.INVALID
                if not evaluation_valid
                else (
                    EpisodeEvaluationStatus.VALID
                    if has_measurement
                    else EpisodeEvaluationStatus.PARTIAL
                )
            )
        )
        if self.evaluation_status is not expected_evaluation_status:
            raise ValueError("execution revision evaluation status is inconsistent")
        if had_error and self.evaluation_status is not EpisodeEvaluationStatus.NOT_RUN:
            raise ValueError("failed execution cannot claim an evaluation")

    def semantic_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("event_id")
        payload.pop("recorded_at")
        return payload
