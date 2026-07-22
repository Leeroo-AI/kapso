"""Precommitted factual release-matrix evidence for expert promotion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    require_content_id,
    require_identifier,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    EvaluationFingerprint,
    StrictContract,
    TaskAdapterManifest,
    TaskAdapterPackagePin,
    TaskContextBinding,
    TaskEvaluatorMetricComparisonBinding,
)
from kapso.cross_run.expert.replay_protocol_contracts import require_finite_float
from kapso.cross_run.task_adapters import (
    TaskAdapterVerificationReceipt,
    task_adapter_binding_id,
)


class ExpertReleaseMatrixContractError(ValueError):
    """Release-matrix facts are structurally invalid or internally inconsistent."""


class ExpertReleaseMatrixMode(str, Enum):
    """Whether a release matrix compares a candidate with a released parent."""

    PARENT_COMPARISON = "parent_comparison"
    BOOTSTRAP = "bootstrap"


class ExpertReleaseMatrixProvenanceKind(str, Enum):
    """Whether a context comes from source replay or an adapter-owned case."""

    SOURCE_REPLAY = "source_replay"
    ADAPTER_CASE = "adapter_case"


def _require_namespaced_id(value: str, namespace: str, name: str) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise ExpertReleaseMatrixContractError(f"{name} uses the wrong namespace")


def _require_sorted_content_ids(
    values: tuple[str, ...],
    name: str,
    *,
    required: bool = True,
) -> None:
    if (required and not values) or values != tuple(sorted(set(values))):
        raise ExpertReleaseMatrixContractError(
            f"{name} must be non-empty, sorted, and unique"
        )
    for value in values:
        require_content_id(value, name)


def _require_digest(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ExpertReleaseMatrixContractError(f"{name} is invalid")


def _require_observation_event_id(value: str, name: str) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] not in {
        "source-replay-execution-journal-event",
        "task-evaluation-journal-event",
    }:
        raise ExpertReleaseMatrixContractError(f"{name} uses the wrong namespace")


def _verified_task_adapter_dependency_ids(
    manifest: TaskAdapterManifest,
    receipt: TaskAdapterVerificationReceipt,
) -> tuple[str, ...]:
    """Mirror the exact public dependency projection of VerifiedTaskAdapter."""

    return tuple(
        sorted(
            {
                receipt.verification_receipt_id,
                receipt.source_extraction_receipt_id,
                manifest.sanitation_report_id,
                *receipt.proof_object_ids,
            }
        )
    )


@dataclass(frozen=True)
class ExpertReleaseMatrixAdapterAuthority(StrictContract):
    """Durable exact adapter authority captured after live provider verification."""

    adapter_authority_id: str
    task_adapter_pin: TaskAdapterPackagePin
    task_adapter_manifest: TaskAdapterManifest
    verification_receipt: TaskAdapterVerificationReceipt
    task_adapter_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-release-matrix-adapter-authority"
    IDENTITY_FIELD: ClassVar[str] = "adapter_authority_id"

    def _validate(self) -> None:
        pin = self.task_adapter_pin
        manifest = self.task_adapter_manifest
        receipt = self.verification_receipt
        for value, namespace, name in (
            (
                pin.adapter_binding_id,
                "task-adapter-binding",
                "release matrix adapter binding",
            ),
            (
                pin.task_adapter_manifest_id,
                "task-adapter-manifest",
                "release matrix adapter manifest",
            ),
            (
                pin.verification_receipt_id,
                "task-adapter-verification-receipt",
                "release matrix adapter verification receipt",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        expected_proof_refs = {
            manifest.sanitation_report_id,
            *manifest.validation_refs,
        }
        if (
            pin.adapter_binding_id
            != task_adapter_binding_id(
                manifest.task_family_id,
                manifest.task_adapter_id,
            )
            or pin.task_adapter_manifest_id != manifest.task_adapter_manifest_id
            or pin.verification_receipt_id != receipt.verification_receipt_id
            or receipt.task_adapter_manifest_id != manifest.task_adapter_manifest_id
            or receipt.full_manifest_digest
            != tree_or_blob_digest(manifest.to_json_bytes())
            or receipt.publisher_attestation_digest
            != tree_or_blob_digest(canonical_json_bytes(manifest.publisher_attestation))
            or receipt.source_archive_ref != manifest.source_tree_ref
            or receipt.source_tree_hash != manifest.tree_hash
            or set(receipt.proof_object_digests) != expected_proof_refs
        ):
            raise ExpertReleaseMatrixContractError(
                "release matrix adapter authority differs from its exact package"
            )
        _require_sorted_content_ids(
            self.task_adapter_dependency_ids,
            "release matrix adapter dependencies",
        )
        if self.task_adapter_dependency_ids != _verified_task_adapter_dependency_ids(
            manifest,
            receipt,
        ):
            raise ExpertReleaseMatrixContractError(
                "release matrix adapter dependency projection is not exact"
            )

    @property
    def exact_dependency_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    self.task_adapter_pin.adapter_binding_id,
                    self.task_adapter_manifest.task_adapter_manifest_id,
                    *self.task_adapter_dependency_ids,
                }
            )
        )

    @property
    def canonical_key(self) -> tuple[str, str, str]:
        pin = self.task_adapter_pin
        return (
            pin.adapter_binding_id,
            pin.task_adapter_manifest_id,
            pin.verification_receipt_id,
        )


@dataclass(frozen=True)
class ExpertReleaseMatrixProvenanceBinding(StrictContract):
    """One full task context bound to one lineage and its source provenance."""

    provenance_binding_id: str
    provenance_kind: ExpertReleaseMatrixProvenanceKind
    task_context_binding: TaskContextBinding
    adapter_case_id: str | None
    source_replay_selection_id: str | None
    source_bundle_id: str | None
    bundle_lineage_ids: tuple[str, ...]
    source_episode_ids: tuple[str, ...]
    context_materialization_receipt_id: str | None
    starting_artifact_ids: tuple[str, ...]
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-release-matrix-provenance-binding"
    IDENTITY_FIELD: ClassVar[str] = "provenance_binding_id"

    def _validate(self) -> None:
        if self.provenance_kind is ExpertReleaseMatrixProvenanceKind.ADAPTER_CASE:
            if (
                self.adapter_case_id is None
                or self.source_replay_selection_id is not None
                or self.source_bundle_id is not None
                or self.bundle_lineage_ids
                or self.source_episode_ids
                or self.context_materialization_receipt_id is not None
                or self.starting_artifact_ids
            ):
                raise ExpertReleaseMatrixContractError(
                    "adapter-case provenance must contain only its precommitted task case"
                )
            _require_namespaced_id(
                self.adapter_case_id,
                "task-adapter-release-matrix-case",
                "release matrix adapter-owned task case",
            )
            expected_dependencies = {
                self.task_context_binding.task_context_binding_id,
                self.adapter_case_id,
            }
            self._validate_dependency_closure(expected_dependencies)
            return
        if (
            self.adapter_case_id is not None
            or self.source_replay_selection_id is None
            or self.source_bundle_id is None
            or self.context_materialization_receipt_id is None
        ):
            raise ExpertReleaseMatrixContractError(
                "source-reuse provenance requires its complete replay authority"
            )
        for value, namespace, name in (
            (
                self.source_replay_selection_id,
                "expert-source-replay-selection",
                "release matrix source replay selection",
            ),
            (
                self.source_bundle_id,
                "run-bundle",
                "release matrix source bundle",
            ),
            (
                self.context_materialization_receipt_id,
                "expert-source-replay-context-materialization",
                "release matrix context materialization",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        if (
            not self.bundle_lineage_ids
            or len(self.bundle_lineage_ids) != len(set(self.bundle_lineage_ids))
            or self.bundle_lineage_ids[-1] != self.source_bundle_id
        ):
            raise ExpertReleaseMatrixContractError(
                "release matrix bundle lineage must be unique and end at the source bundle"
            )
        for bundle_id in self.bundle_lineage_ids:
            _require_namespaced_id(
                bundle_id,
                "run-bundle",
                "release matrix bundle lineage",
            )
        for values, namespace, name in (
            (
                self.source_episode_ids,
                "transfer-episode",
                "release matrix source episodes",
            ),
            (
                self.starting_artifact_ids,
                "source-replay-starting-artifact",
                "release matrix starting artifacts",
            ),
        ):
            _require_sorted_content_ids(
                values,
                name,
                required=namespace != "source-replay-starting-artifact",
            )
            for value in values:
                _require_namespaced_id(value, namespace, name)
        expected_dependencies = {
            self.task_context_binding.task_context_binding_id,
            self.source_replay_selection_id,
            *self.bundle_lineage_ids,
            *self.source_episode_ids,
            self.context_materialization_receipt_id,
            *self.starting_artifact_ids,
        }
        self._validate_dependency_closure(expected_dependencies)

    def _validate_dependency_closure(self, expected_dependencies: set[str]) -> None:
        _require_sorted_content_ids(
            self.exact_dependency_ids,
            "release matrix provenance dependencies",
        )
        if set(self.exact_dependency_ids) != expected_dependencies:
            raise ExpertReleaseMatrixContractError(
                "release matrix provenance dependency closure is not exact"
            )

    @property
    def canonical_key(self) -> tuple[str, str]:
        return (
            self.task_context_binding.task_context_binding_id,
            self.independence_identity_id,
        )

    @property
    def independence_identity_id(self) -> str:
        if self.provenance_kind is ExpertReleaseMatrixProvenanceKind.ADAPTER_CASE:
            return self.adapter_case_id
        return self.bundle_lineage_ids[0]


@dataclass(frozen=True)
class ExpertReleaseMatrixEvaluationCell(StrictContract):
    """One precommitted full-fingerprint candidate or candidate-parent cell."""

    evaluation_cell_id: str
    mode: ExpertReleaseMatrixMode
    validation_attempt_id: str
    candidate_id: str
    candidate_tree_hash: str
    parent_release_id: str | None
    parent_tree_hash: str | None
    adapter_authority_id: str
    provenance_binding_id: str
    task_context_binding: TaskContextBinding
    independence_identity_id: str
    evaluation_fingerprint: EvaluationFingerprint
    metric_comparison_binding: TaskEvaluatorMetricComparisonBinding
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-release-matrix-evaluation-cell"
    IDENTITY_FIELD: ClassVar[str] = "evaluation_cell_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.validation_attempt_id,
                "expert-validation-attempt",
                "release matrix cell validation attempt",
            ),
            (
                self.candidate_id,
                "expert-candidate",
                "release matrix cell candidate",
            ),
            (
                self.adapter_authority_id,
                "expert-release-matrix-adapter-authority",
                "release matrix cell adapter authority",
            ),
            (
                self.provenance_binding_id,
                "expert-release-matrix-provenance-binding",
                "release matrix cell provenance binding",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        require_content_id(
            self.independence_identity_id,
            "release matrix cell independence identity",
        )
        if self.independence_identity_id.split(":sha256:", 1)[0] not in {
            "run-bundle",
            "task-adapter-release-matrix-case",
        }:
            raise ExpertReleaseMatrixContractError(
                "release matrix cell independence identity uses the wrong namespace"
            )
        _require_digest(self.candidate_tree_hash, "release matrix candidate tree")
        if self.mode is ExpertReleaseMatrixMode.BOOTSTRAP:
            if self.parent_release_id is not None or self.parent_tree_hash is not None:
                raise ExpertReleaseMatrixContractError(
                    "bootstrap release matrix cell cannot name a parent"
                )
        else:
            if self.parent_release_id is None or self.parent_tree_hash is None:
                raise ExpertReleaseMatrixContractError(
                    "parent-comparison release matrix cell requires a parent tree"
                )
            _require_namespaced_id(
                self.parent_release_id,
                "expert-base-release",
                "release matrix cell parent release",
            )
            _require_digest(self.parent_tree_hash, "release matrix parent tree")
        fingerprint = self.evaluation_fingerprint
        binding = self.metric_comparison_binding
        if (
            binding.evaluator_fingerprint != fingerprint.evaluator_fingerprint
            or binding.metric_name != fingerprint.metric_name
            or binding.objective_direction is not fingerprint.objective_direction
        ):
            raise ExpertReleaseMatrixContractError(
                "release matrix cell metric authority differs from its fingerprint"
            )
        _require_sorted_content_ids(
            self.exact_dependency_ids,
            "release matrix cell dependencies",
        )
        expected_dependencies = {
            self.validation_attempt_id,
            self.candidate_id,
            self.adapter_authority_id,
            self.provenance_binding_id,
            self.task_context_binding.task_context_binding_id,
            self.independence_identity_id,
            fingerprint.evaluation_fingerprint_id,
        }
        if self.parent_release_id is not None:
            expected_dependencies.add(self.parent_release_id)
        if len(expected_dependencies) != (8 if self.parent_release_id else 7):
            raise ExpertReleaseMatrixContractError(
                "release matrix cell dependency roles must be distinct"
            )
        if set(self.exact_dependency_ids) != expected_dependencies:
            raise ExpertReleaseMatrixContractError(
                "release matrix cell dependency closure is not exact"
            )

    @property
    def metric_key(self) -> tuple[str, str]:
        return (
            self.evaluation_fingerprint.evaluator_fingerprint,
            self.evaluation_fingerprint.metric_name,
        )

    @property
    def canonical_key(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.metric_comparison_binding.comparison_dimension_id,
            self.adapter_authority_id,
            self.evaluation_fingerprint.evaluation_fingerprint_id,
            self.task_context_binding.task_context_binding_id,
            self.independence_identity_id,
            self.provenance_binding_id,
        )


@dataclass(frozen=True)
class ExpertReleaseMatrixEvaluationPlan(StrictContract):
    """Canonical evaluation coverage fixed before observations are accepted."""

    evaluation_plan_id: str
    mode: ExpertReleaseMatrixMode
    validation_attempt_id: str
    candidate_id: str
    candidate_commit_record_id: str
    candidate_tree_hash: str
    scope_contract_id: str
    parent_release_id: str | None
    parent_tree_hash: str | None
    validation_policy_id: str
    configuration_fingerprint: str
    adapter_authorities: tuple[ExpertReleaseMatrixAdapterAuthority, ...]
    provenance_bindings: tuple[ExpertReleaseMatrixProvenanceBinding, ...]
    evaluation_cells: tuple[ExpertReleaseMatrixEvaluationCell, ...]
    external_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-release-matrix-evaluation-plan"
    IDENTITY_FIELD: ClassVar[str] = "evaluation_plan_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.validation_attempt_id,
                "expert-validation-attempt",
                "release matrix plan validation attempt",
            ),
            (
                self.candidate_id,
                "expert-candidate",
                "release matrix plan candidate",
            ),
            (
                self.candidate_commit_record_id,
                "expert-candidate-commit",
                "release matrix plan candidate commit",
            ),
            (
                self.scope_contract_id,
                "expert-scope-contract",
                "release matrix plan scope contract",
            ),
            (
                self.validation_policy_id,
                "expert-validation-policy",
                "release matrix plan validation policy",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        _require_digest(self.candidate_tree_hash, "release matrix plan candidate tree")
        _require_digest(
            self.configuration_fingerprint,
            "release matrix plan configuration fingerprint",
        )
        if self.mode is ExpertReleaseMatrixMode.BOOTSTRAP:
            if self.parent_release_id is not None or self.parent_tree_hash is not None:
                raise ExpertReleaseMatrixContractError(
                    "bootstrap release matrix plan cannot name a parent"
                )
        else:
            if self.parent_release_id is None or self.parent_tree_hash is None:
                raise ExpertReleaseMatrixContractError(
                    "parent-comparison release matrix plan requires a parent tree"
                )
            _require_namespaced_id(
                self.parent_release_id,
                "expert-base-release",
                "release matrix plan parent release",
            )
            _require_digest(self.parent_tree_hash, "release matrix plan parent tree")
        authority_keys = tuple(
            authority.canonical_key for authority in self.adapter_authorities
        )
        authority_ids = tuple(
            authority.adapter_authority_id for authority in self.adapter_authorities
        )
        authority_bindings = tuple(
            authority.task_adapter_pin.adapter_binding_id
            for authority in self.adapter_authorities
        )
        if (
            not authority_keys
            or authority_keys != tuple(sorted(set(authority_keys)))
            or len(authority_ids) != len(set(authority_ids))
            or len(authority_bindings) != len(set(authority_bindings))
        ):
            raise ExpertReleaseMatrixContractError(
                "release matrix adapter authorities must be canonical and unique"
            )
        provenance_keys = tuple(
            provenance.canonical_key for provenance in self.provenance_bindings
        )
        provenance_ids = tuple(
            provenance.provenance_binding_id for provenance in self.provenance_bindings
        )
        if (
            not provenance_keys
            or provenance_keys != tuple(sorted(set(provenance_keys)))
            or len(provenance_ids) != len(set(provenance_ids))
        ):
            raise ExpertReleaseMatrixContractError(
                "release matrix provenance bindings must be canonical and unique"
            )
        cell_keys = tuple(cell.canonical_key for cell in self.evaluation_cells)
        cell_ids = tuple(cell.evaluation_cell_id for cell in self.evaluation_cells)
        if (
            not cell_keys
            or cell_keys != tuple(sorted(set(cell_keys)))
            or len(cell_ids) != len(set(cell_ids))
        ):
            raise ExpertReleaseMatrixContractError(
                "release matrix evaluation cells must be canonical and unique"
            )
        plan_subjects = (
            self.mode,
            self.validation_attempt_id,
            self.candidate_id,
            self.candidate_tree_hash,
            self.parent_release_id,
            self.parent_tree_hash,
        )
        if any(
            (
                cell.mode,
                cell.validation_attempt_id,
                cell.candidate_id,
                cell.candidate_tree_hash,
                cell.parent_release_id,
                cell.parent_tree_hash,
            )
            != plan_subjects
            for cell in self.evaluation_cells
        ):
            raise ExpertReleaseMatrixContractError(
                "release matrix cells must share the plan subjects and mode"
            )
        if any(
            authority.task_adapter_manifest.scope_contract_id != self.scope_contract_id
            for authority in self.adapter_authorities
        ):
            raise ExpertReleaseMatrixContractError(
                "release matrix adapter authority differs from the plan scope"
            )
        authorities = {
            authority.adapter_authority_id: authority
            for authority in self.adapter_authorities
        }
        provenance_by_id = {
            provenance.provenance_binding_id: provenance
            for provenance in self.provenance_bindings
        }
        if {cell.adapter_authority_id for cell in self.evaluation_cells} != set(
            authorities
        ):
            raise ExpertReleaseMatrixContractError(
                "release matrix plan adapter coverage is not exact"
            )
        if {cell.provenance_binding_id for cell in self.evaluation_cells} != set(
            provenance_by_id
        ):
            raise ExpertReleaseMatrixContractError(
                "release matrix plan provenance coverage is not exact"
            )
        if self.mode is ExpertReleaseMatrixMode.BOOTSTRAP and any(
            provenance.provenance_kind
            is not ExpertReleaseMatrixProvenanceKind.ADAPTER_CASE
            for provenance in self.provenance_bindings
        ):
            raise ExpertReleaseMatrixContractError(
                "bootstrap release matrix requires adapter-owned task cases"
            )
        context_owners: dict[str, tuple[str, str]] = {}
        fingerprint_groups: set[tuple[str, str, str]] = set()
        metric_groups: dict[tuple[str, str], set[tuple[str, str]]] = {}
        for cell in self.evaluation_cells:
            authority = authorities[cell.adapter_authority_id]
            provenance = provenance_by_id[cell.provenance_binding_id]
            manifest = authority.task_adapter_manifest
            context = cell.task_context_binding
            if (
                provenance.task_context_binding != context
                or provenance.independence_identity_id != cell.independence_identity_id
                or context.scope_contract_id != manifest.scope_contract_id
                or context.task_family_id != manifest.task_family_id
                or context.task_adapter_id != manifest.task_adapter_id
                or cell.evaluation_fingerprint.evaluator_fingerprint
                not in manifest.task_evaluator.supported_evaluator_fingerprints
                or cell.metric_comparison_binding
                not in manifest.task_evaluator.metric_comparison_bindings
            ):
                raise ExpertReleaseMatrixContractError(
                    "release matrix cell differs from adapter or provenance authority"
                )
            context_id = context.task_context_binding_id
            context_owner = (
                cell.provenance_binding_id,
                cell.independence_identity_id,
            )
            previous_context_owner = context_owners.get(context_id)
            if previous_context_owner is not None and (
                previous_context_owner != context_owner
            ):
                raise ExpertReleaseMatrixContractError(
                    "release matrix context has multiple lineage owners"
                )
            context_owners[context_id] = context_owner
            fingerprint_group = (
                cell.adapter_authority_id,
                cell.evaluation_fingerprint.evaluation_fingerprint_id,
                cell.provenance_binding_id,
            )
            if fingerprint_group in fingerprint_groups:
                raise ExpertReleaseMatrixContractError(
                    "release matrix plan repeats an adapter, fingerprint, and provenance cell"
                )
            fingerprint_groups.add(fingerprint_group)
            metric_groups.setdefault(
                (cell.adapter_authority_id, context_id), set()
            ).add(cell.metric_key)
        for group, observed_metric_keys in metric_groups.items():
            authority_id, _ = group
            declared_metric_keys = {
                (
                    binding.evaluator_fingerprint,
                    binding.metric_name,
                )
                for binding in authorities[
                    authority_id
                ].task_adapter_manifest.task_evaluator.metric_comparison_bindings
            }
            if observed_metric_keys != declared_metric_keys:
                raise ExpertReleaseMatrixContractError(
                    "release matrix plan metric coverage is not exact"
                )
        _require_sorted_content_ids(
            self.external_dependency_ids,
            "release matrix plan external dependencies",
        )
        internal_ids = {*authority_ids, *provenance_ids, *cell_ids}
        expected_external_dependencies = {
            dependency_id
            for authority in self.adapter_authorities
            for dependency_id in authority.exact_dependency_ids
        }
        expected_external_dependencies.update(
            dependency_id
            for provenance in self.provenance_bindings
            for dependency_id in provenance.exact_dependency_ids
        )
        expected_external_dependencies.update(
            dependency_id
            for cell in self.evaluation_cells
            for dependency_id in cell.exact_dependency_ids
            if dependency_id not in internal_ids
        )
        expected_external_dependencies.update(
            {
                self.validation_attempt_id,
                self.candidate_id,
                self.candidate_commit_record_id,
                self.scope_contract_id,
                self.validation_policy_id,
            }
        )
        if self.parent_release_id is not None:
            expected_external_dependencies.add(self.parent_release_id)
        if set(self.external_dependency_ids) != expected_external_dependencies:
            raise ExpertReleaseMatrixContractError(
                "release matrix plan external dependency closure is not exact"
            )

    @property
    def exact_dependency_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    *self.external_dependency_ids,
                    *(
                        authority.adapter_authority_id
                        for authority in self.adapter_authorities
                    ),
                    *(
                        provenance.provenance_binding_id
                        for provenance in self.provenance_bindings
                    ),
                    *(cell.evaluation_cell_id for cell in self.evaluation_cells),
                }
            )
        )


@dataclass(frozen=True)
class ExpertReleaseMatrixComparisonRow(StrictContract):
    """Observed values for one and only one precommitted evaluation cell."""

    comparison_row_id: str
    evaluation_cell_id: str
    candidate_observation_event_id: str
    parent_observation_event_id: str | None
    candidate_replicate_values: Mapping[str, float]
    parent_replicate_values: Mapping[str, float] | None

    CONTENT_NAMESPACE: ClassVar[str] = "expert-release-matrix-comparison-row"
    IDENTITY_FIELD: ClassVar[str] = "comparison_row_id"

    def _validate(self) -> None:
        _require_namespaced_id(
            self.evaluation_cell_id,
            "expert-release-matrix-evaluation-cell",
            "release matrix row evaluation cell",
        )
        _require_observation_event_id(
            self.candidate_observation_event_id,
            "release matrix candidate observation event",
        )
        if self.parent_observation_event_id is not None:
            _require_observation_event_id(
                self.parent_observation_event_id,
                "release matrix parent observation event",
            )
            if self.parent_observation_event_id == self.candidate_observation_event_id:
                raise ExpertReleaseMatrixContractError(
                    "release matrix candidate and control events must be distinct"
                )
        if not self.candidate_replicate_values:
            raise ExpertReleaseMatrixContractError(
                "release matrix candidate replicate values must not be empty"
            )
        for replicate_id, value in self.candidate_replicate_values.items():
            require_identifier(replicate_id, "release matrix candidate replicate")
            require_finite_float(value, "release matrix candidate replicate value")
            if value == 0.0 and math.copysign(1.0, value) < 0.0:
                raise ExpertReleaseMatrixContractError(
                    "release matrix candidate replicate value must normalize signed zero"
                )
        if self.parent_replicate_values is not None:
            if not self.parent_replicate_values:
                raise ExpertReleaseMatrixContractError(
                    "release matrix parent replicate values must not be empty"
                )
            for replicate_id, value in self.parent_replicate_values.items():
                require_identifier(replicate_id, "release matrix parent replicate")
                require_finite_float(value, "release matrix parent replicate value")
                if value == 0.0 and math.copysign(1.0, value) < 0.0:
                    raise ExpertReleaseMatrixContractError(
                        "release matrix parent replicate value must normalize signed zero"
                    )

    @property
    def exact_dependency_ids(self) -> tuple[str, ...]:
        dependencies = {
            self.evaluation_cell_id,
            self.candidate_observation_event_id,
        }
        if self.parent_observation_event_id is not None:
            dependencies.add(self.parent_observation_event_id)
        return tuple(sorted(dependencies))


@dataclass(frozen=True)
class ExpertReleaseMatrixReport(StrictContract):
    """Observed matrix values bound exactly to one embedded precommitted plan."""

    release_matrix_report_id: str
    mode: ExpertReleaseMatrixMode
    validation_attempt_id: str
    candidate_id: str
    candidate_commit_record_id: str
    candidate_tree_hash: str
    scope_contract_id: str
    parent_release_id: str | None
    parent_tree_hash: str | None
    validation_policy_id: str
    configuration_fingerprint: str
    evaluation_plan: ExpertReleaseMatrixEvaluationPlan
    evidence_rows: tuple[ExpertReleaseMatrixComparisonRow, ...]
    exact_evidence_input_ids: tuple[str, ...]
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-release-matrix-report"
    IDENTITY_FIELD: ClassVar[str] = "release_matrix_report_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.validation_attempt_id,
                "expert-validation-attempt",
                "release matrix validation attempt",
            ),
            (
                self.candidate_id,
                "expert-candidate",
                "release matrix candidate",
            ),
            (
                self.candidate_commit_record_id,
                "expert-candidate-commit",
                "release matrix candidate commit",
            ),
            (
                self.scope_contract_id,
                "expert-scope-contract",
                "release matrix scope contract",
            ),
            (
                self.validation_policy_id,
                "expert-validation-policy",
                "release matrix validation policy",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        _require_digest(self.candidate_tree_hash, "release matrix candidate tree")
        _require_digest(
            self.configuration_fingerprint,
            "release matrix configuration fingerprint",
        )
        if self.mode is ExpertReleaseMatrixMode.BOOTSTRAP:
            if self.parent_release_id is not None or self.parent_tree_hash is not None:
                raise ExpertReleaseMatrixContractError(
                    "bootstrap release matrix cannot name a parent"
                )
        else:
            if self.parent_release_id is None or self.parent_tree_hash is None:
                raise ExpertReleaseMatrixContractError(
                    "parent-comparison release matrix requires a parent tree"
                )
            _require_namespaced_id(
                self.parent_release_id,
                "expert-base-release",
                "release matrix parent release",
            )
            _require_digest(self.parent_tree_hash, "release matrix parent tree")
        plan = self.evaluation_plan
        if (
            plan.mode,
            plan.validation_attempt_id,
            plan.candidate_id,
            plan.candidate_commit_record_id,
            plan.candidate_tree_hash,
            plan.scope_contract_id,
            plan.parent_release_id,
            plan.parent_tree_hash,
            plan.validation_policy_id,
            plan.configuration_fingerprint,
        ) != (
            self.mode,
            self.validation_attempt_id,
            self.candidate_id,
            self.candidate_commit_record_id,
            self.candidate_tree_hash,
            self.scope_contract_id,
            self.parent_release_id,
            self.parent_tree_hash,
            self.validation_policy_id,
            self.configuration_fingerprint,
        ):
            raise ExpertReleaseMatrixContractError(
                "release matrix plan differs from report authority"
            )
        cell_ids = tuple(cell.evaluation_cell_id for cell in plan.evaluation_cells)
        row_cell_ids = tuple(row.evaluation_cell_id for row in self.evidence_rows)
        if row_cell_ids != cell_ids:
            raise ExpertReleaseMatrixContractError(
                "release matrix rows must cover plan cells exactly once and in order"
            )
        provenance_by_id = {
            provenance.provenance_binding_id: provenance
            for provenance in plan.provenance_bindings
        }
        for cell, row in zip(plan.evaluation_cells, self.evidence_rows, strict=True):
            self._validate_observation(
                cell,
                row,
                provenance_by_id[cell.provenance_binding_id],
            )
        _require_sorted_content_ids(
            self.exact_evidence_input_ids,
            "release matrix exact evidence inputs",
        )
        expected_evidence_inputs = {
            plan.evaluation_plan_id,
            *plan.external_dependency_ids,
        }
        if set(self.exact_evidence_input_ids) != expected_evidence_inputs:
            raise ExpertReleaseMatrixContractError(
                "release matrix evidence input closure is not exact"
            )
        _require_sorted_content_ids(
            self.exact_dependency_ids,
            "release matrix report dependencies",
        )
        expected_dependencies = {
            self.validation_attempt_id,
            self.candidate_id,
            self.candidate_commit_record_id,
            self.scope_contract_id,
            self.validation_policy_id,
            plan.evaluation_plan_id,
            *plan.exact_dependency_ids,
            *(row.comparison_row_id for row in self.evidence_rows),
            *(
                dependency_id
                for row in self.evidence_rows
                for dependency_id in row.exact_dependency_ids
            ),
        }
        if self.parent_release_id is not None:
            expected_dependencies.add(self.parent_release_id)
        if set(self.exact_dependency_ids) != expected_dependencies:
            raise ExpertReleaseMatrixContractError(
                "release matrix report dependency closure is not exact"
            )

    @staticmethod
    def _validate_observation(
        cell: ExpertReleaseMatrixEvaluationCell,
        row: ExpertReleaseMatrixComparisonRow,
        provenance: ExpertReleaseMatrixProvenanceBinding,
    ) -> None:
        expected_event_namespace = (
            "source-replay-execution-journal-event"
            if provenance.provenance_kind
            is ExpertReleaseMatrixProvenanceKind.SOURCE_REPLAY
            else "task-evaluation-journal-event"
        )
        if row.candidate_observation_event_id.split(":sha256:", 1)[0] != (
            expected_event_namespace
        ) or (
            row.parent_observation_event_id is not None
            and row.parent_observation_event_id.split(":sha256:", 1)[0]
            != expected_event_namespace
        ):
            raise ExpertReleaseMatrixContractError(
                "release matrix observation channel differs from its provenance"
            )
        expected_replicates = set(cell.evaluation_fingerprint.seed_or_replicate_ids)
        if set(row.candidate_replicate_values) != expected_replicates:
            raise ExpertReleaseMatrixContractError(
                "release matrix candidate replicate coverage differs from its plan cell"
            )
        if cell.mode is ExpertReleaseMatrixMode.BOOTSTRAP:
            if (
                row.parent_replicate_values is not None
                or row.parent_observation_event_id is not None
            ):
                raise ExpertReleaseMatrixContractError(
                    "bootstrap row must contain only the candidate result"
                )
            return
        if (
            row.parent_replicate_values is None
            or row.parent_observation_event_id is None
            or set(row.parent_replicate_values) != expected_replicates
        ):
            raise ExpertReleaseMatrixContractError(
                "parent-comparison row requires exact paired replicate results"
            )
