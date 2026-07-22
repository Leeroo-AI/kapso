"""Complete packet-only coding-agent review for expert candidates."""

from __future__ import annotations

import base64
import os
import stat
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from kapso.cross_run.agent_artifacts import CodingAgentWorkspaceAccess
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    parse_json_bytes,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ExpertCandidateOperationRecord,
    ExpertCandidateValidationState,
    ExpertEvaluatorOutcome,
    ExpertEvaluatorResultRecord,
    ExpertPromotionState,
    ExpertReviewDisposition,
    ExpertSealedCanaryAggregate,
    ExpertValidationAttempt,
    ExpertValidationStage,
)
from kapso.cross_run.expert.proposal_contract import (
    ExpertCandidateAncestorInput,
    mint_expert_candidate_ancestor_input,
)
from kapso.cross_run.expert.replay_publication_contracts import (
    ExpertSourceReplayStageResultRecord,
)
from kapso.cross_run.expert.review_contracts import (
    EXPERT_AUTOMATED_REVIEW_CONTRACT_VERSION,
    ExpertAutomatedReviewAdjudication,
    ExpertAutomatedReviewAssertion,
    ExpertAutomatedReviewError,
    ExpertAutomatedReviewOperationRecord,
    ExpertAutomatedReviewOutcome,
    ExpertAutomatedReviewPacket,
    ExpertAutomatedReviewStageResultRecord,
)
from kapso.cross_run.expert.store import StoredExpertCandidate
from kapso.cross_run.settings import (
    ExpertPromotionPolicySettings,
    ExpertReviewerSettings,
    ExpertSettings,
    ExpertValidationPolicy,
)
from kapso.execution.coding_agents.operation_receipt import (
    seal_coding_agent_operation,
)
from kapso.execution.coding_agents.structured_call import (
    CodingAgentCallRequest,
    CodingAgentRunnerSettings,
    CodingAgentWorkspacePolicy,
    SubprocessCodingAgentCallRunner,
    coding_agent_invocation_bytes,
    coding_agent_mcp_configuration_fingerprint,
    coding_agent_response_schema_bytes,
)

_PROMPT_TEMPLATE_PATH = Path(__file__).parents[1] / "prompts" / "expert_reviewer.md"
_PROMPT_PACKET_MARKER = "AUTOMATED_REVIEW_PACKET_JSON"
_PROVISIONAL_OPERATION_ID = "agent_call_" + "0" * 32
_AUTOMATED_REVIEW_EXECUTION_SEAL = object()
_TRUSTED_CODING_AGENT_RUN_METHOD = SubprocessCodingAgentCallRunner.run

ExpertAcceptedReviewInput = (
    ExpertEvaluatorResultRecord | ExpertSourceReplayStageResultRecord
)


@dataclass(frozen=True)
class PreparedExpertAutomatedReviewPacket:
    """Verified packet plus the complete records rendered for reviewers."""

    packet: ExpertAutomatedReviewPacket
    candidate_input: ExpertCandidateAncestorInput
    candidate_operation: ExpertCandidateOperationRecord
    validation_attempt: ExpertValidationAttempt
    authorization_state: ExpertCandidateValidationState
    validation_policy: ExpertValidationPolicy
    accepted_stage_results: tuple[ExpertAcceptedReviewInput, ...]

    def __post_init__(self) -> None:
        packet = self.packet
        attempt = self.validation_attempt
        state = self.authorization_state
        candidate_manifest = self.candidate_input.manifest
        if (
            packet.candidate_input_id != self.candidate_input.ancestor_input_id
            or packet.proposer_operation_record_id
            != self.candidate_operation.operation_record_id
            or packet.validation_attempt_id != attempt.validation_attempt_id
            or packet.authorization_state_id != state.validation_state_id
            or packet.validation_policy_id
            != self.validation_policy.validation_policy_id
            or packet.accepted_stage_results != state.accepted_stage_results
            or len(self.accepted_stage_results) != len(packet.accepted_stage_results)
            or state.promotion_state is not ExpertPromotionState.VALIDATING
            or state.next_stage is not ExpertValidationStage.AUTOMATED_REVIEW
            or state.review_assertion_ids
            or state.terminal_evidence_ids
            or packet.candidate_id != attempt.candidate_id
            or packet.candidate_tree_hash != attempt.candidate_tree_hash
            or packet.candidate_commit_record_id != attempt.candidate_commit_record_id
            or packet.scope_contract_id != attempt.scope_contract_id
            or packet.parent_release_id != attempt.parent_release_id
            or packet.configuration_fingerprint != attempt.configuration_fingerprint
            or candidate_manifest.candidate_id != packet.candidate_id
            or candidate_manifest.candidate_tree_hash != packet.candidate_tree_hash
            or candidate_manifest.scope_contract_id != packet.scope_contract_id
            or candidate_manifest.parent_release_id != packet.parent_release_id
            or candidate_manifest.proposer_operation_record_id
            != self.candidate_operation.operation_record_id
            or candidate_manifest.trigger_evidence_packet_id
            != packet.trigger_evidence_packet_id
            or candidate_manifest.trigger_decision_id != packet.trigger_decision_id
            or self.candidate_operation.trigger_evidence_packet_id
            != packet.trigger_evidence_packet_id
            or self.candidate_operation.trigger_decision_id
            != packet.trigger_decision_id
        ):
            raise ExpertAutomatedReviewError(
                "prepared automated review packet closure is inconsistent"
            )
        _validate_prepared_review_stage_results(self)


class ExpertAutomatedReviewExecution:
    """Process-local coordinator authority over authenticated review facts."""

    __slots__ = (
        "_consumed",
        "_coordinator",
        "_owner_process_id",
        "adjudication",
        "assertions",
        "operation_records",
        "prepared_packet",
        "stage_result",
    )

    def __init__(
        self,
        seal: object,
        coordinator: "ExpertAutomatedReviewCoordinator",
        *,
        prepared_packet: PreparedExpertAutomatedReviewPacket,
        assertions: tuple[ExpertAutomatedReviewAssertion, ...],
        operation_records: tuple[ExpertAutomatedReviewOperationRecord, ...],
        adjudication: ExpertAutomatedReviewAdjudication,
        stage_result: ExpertAutomatedReviewStageResultRecord,
    ) -> None:
        if seal is not _AUTOMATED_REVIEW_EXECUTION_SEAL:
            raise ExpertAutomatedReviewError(
                "automated review execution is not coordinator sealed"
            )
        object.__setattr__(self, "_coordinator", coordinator)
        object.__setattr__(self, "_owner_process_id", os.getpid())
        object.__setattr__(self, "_consumed", False)
        object.__setattr__(self, "prepared_packet", prepared_packet)
        object.__setattr__(self, "assertions", assertions)
        object.__setattr__(self, "operation_records", operation_records)
        object.__setattr__(self, "adjudication", adjudication)
        object.__setattr__(self, "stage_result", stage_result)

    def __setattr__(self, name, value) -> None:
        raise ExpertAutomatedReviewError("automated review execution is immutable")

    def _require_bound(self, coordinator: object) -> None:
        if (
            self._coordinator is not coordinator
            or self._owner_process_id != os.getpid()
            or self._consumed
        ):
            raise ExpertAutomatedReviewError(
                "automated review execution is consumed or foreign"
            )

    def _consume(self, coordinator: object) -> None:
        self._require_bound(coordinator)
        object.__setattr__(self, "_consumed", True)


class ExpertAutomatedReviewCoordinator:
    """Prepare one immutable review round and invoke every configured slot."""

    def __init__(self, settings: ExpertSettings, workspace_root: Path):
        self._validate_workspace_root(workspace_root)
        self.settings = settings
        self.workspace_root = workspace_root
        self._owner_process_id = os.getpid()
        self.runner = SubprocessCodingAgentCallRunner(self._runner_settings())

    def _runner_settings(self) -> CodingAgentRunnerSettings:
        return CodingAgentRunnerSettings(
            artifact_root=str(
                self.workspace_root / self.settings.agent_artifact_path
            ),
            termination_grace_seconds=self.settings.termination_grace_seconds,
            sensitive_file_glob_scan_max_depth=(
                self.settings.sensitive_file_glob_scan_max_depth
            ),
        )

    def _require_runner_authority(self) -> None:
        if (
            self._owner_process_id != os.getpid()
            or type(self.runner) is not SubprocessCodingAgentCallRunner
            or self.runner.settings != self._runner_settings()
            or "run" in vars(self.runner)
            or type(self.runner).run is not _TRUSTED_CODING_AGENT_RUN_METHOD
        ):
            raise ExpertAutomatedReviewError(
                "automated review runner lacks configured CLI authority"
            )

    def prepare(
        self,
        *,
        stored_candidate: StoredExpertCandidate,
        validation_attempt: ExpertValidationAttempt,
        authorization_transition_id: str,
        authorization_state: ExpertCandidateValidationState,
        accepted_stage_results: tuple[ExpertAcceptedReviewInput, ...],
    ) -> PreparedExpertAutomatedReviewPacket:
        policy = self.settings.validation.policy.validation_policy()
        closure = stored_candidate.closure
        candidate_input = mint_expert_candidate_ancestor_input(
            manifest=closure.manifest,
            scope_contract=closure.trigger_packet.scope_contract,
            patch=closure.patch,
            candidate_tree=closure.candidate_tree,
            repository_map=closure.repository_map,
            module_contracts=closure.module_contracts,
            workspace_delta=closure.workspace_delta,
            sanitation_report=closure.sanitation_report,
            candidate_contents=closure.candidate_contents,
        )
        self._validate_preparation(
            stored_candidate=stored_candidate,
            candidate_input=candidate_input,
            attempt=validation_attempt,
            authorization_state=authorization_state,
            accepted_stage_results=accepted_stage_results,
            policy=policy,
        )
        dependencies = {
            validation_attempt.validation_attempt_id,
            authorization_transition_id,
            authorization_state.validation_state_id,
            validation_attempt.candidate_id,
            validation_attempt.candidate_commit_record_id,
            candidate_input.ancestor_input_id,
            closure.operation.operation_record_id,
            closure.trigger_packet.evidence_packet_id,
            closure.trigger_decision.trigger_decision_id,
            validation_attempt.scope_contract_id,
            validation_attempt.validation_policy_id,
            *(
                result.stage_result_record_id
                for result in authorization_state.accepted_stage_results
            ),
        }
        if validation_attempt.parent_release_id is not None:
            dependencies.add(validation_attempt.parent_release_id)
        packet = ExpertAutomatedReviewPacket.mint(
            validation_attempt_id=validation_attempt.validation_attempt_id,
            authorization_transition_id=authorization_transition_id,
            authorization_state_id=authorization_state.validation_state_id,
            candidate_id=validation_attempt.candidate_id,
            candidate_tree_hash=validation_attempt.candidate_tree_hash,
            candidate_commit_record_id=(validation_attempt.candidate_commit_record_id),
            candidate_input_id=candidate_input.ancestor_input_id,
            proposer_operation_record_id=closure.operation.operation_record_id,
            trigger_evidence_packet_id=closure.trigger_packet.evidence_packet_id,
            trigger_decision_id=closure.trigger_decision.trigger_decision_id,
            scope_contract_id=validation_attempt.scope_contract_id,
            parent_release_id=validation_attempt.parent_release_id,
            validation_policy_id=validation_attempt.validation_policy_id,
            configuration_fingerprint=(validation_attempt.configuration_fingerprint),
            agent_artifact_byte_limit=self.settings.agent_artifact_byte_limit,
            accepted_stage_results=authorization_state.accepted_stage_results,
            exact_dependency_ids=tuple(sorted(dependencies)),
        )
        return PreparedExpertAutomatedReviewPacket(
            packet=packet,
            candidate_input=candidate_input,
            candidate_operation=closure.operation,
            validation_attempt=validation_attempt,
            authorization_state=authorization_state,
            validation_policy=policy,
            accepted_stage_results=accepted_stage_results,
        )

    def execute(
        self,
        prepared: PreparedExpertAutomatedReviewPacket,
        *,
        workspace: Path,
    ) -> ExpertAutomatedReviewExecution:
        self._require_runner_authority()
        if (
            prepared.packet.agent_artifact_byte_limit
            != self.settings.agent_artifact_byte_limit
            or prepared.packet.configuration_fingerprint
            != self.settings.validation.configuration_fingerprint
            or prepared.validation_policy
            != self.settings.validation.policy.validation_policy()
        ):
            raise ExpertAutomatedReviewError(
                "automated review packet differs from configured execution authority"
            )
        self._validate_review_workspace(workspace)
        assertions = []
        operations = []
        for reviewer in self.settings.validation.policy.reviewers:
            assertion, operation = self._review(
                prepared,
                reviewer,
                workspace,
            )
            assertions.append(assertion)
            operations.append(operation)
        assertion_tuple = tuple(
            sorted(assertions, key=lambda assertion: assertion.reviewer_id)
        )
        operation_tuple = tuple(
            sorted(
                operations,
                key=lambda operation: operation.operation_receipt.principal_id,
            )
        )
        adjudication = adjudicate_expert_automated_review(
            packet=prepared.packet,
            validation_policy=prepared.validation_policy,
            assertions=assertion_tuple,
            operation_records=operation_tuple,
        )
        stage_result = build_expert_automated_review_stage_result(
            prepared=prepared,
            assertions=assertion_tuple,
            operation_records=operation_tuple,
            adjudication=adjudication,
        )
        validate_expert_automated_review_facts(
            prepared=prepared,
            assertions=assertion_tuple,
            operation_records=operation_tuple,
            adjudication=adjudication,
            stage_result=stage_result,
        )
        return ExpertAutomatedReviewExecution(
            _AUTOMATED_REVIEW_EXECUTION_SEAL,
            self,
            prepared_packet=prepared,
            assertions=assertion_tuple,
            operation_records=operation_tuple,
            adjudication=adjudication,
            stage_result=stage_result,
        )

    def _review(
        self,
        prepared: PreparedExpertAutomatedReviewPacket,
        reviewer: ExpertReviewerSettings,
        workspace: Path,
    ) -> tuple[
        ExpertAutomatedReviewAssertion,
        ExpertAutomatedReviewOperationRecord,
    ]:
        configured = {
            slot.reviewer_id: slot for slot in self.settings.validation.policy.reviewers
        }
        if configured.get(reviewer.reviewer_id) != reviewer:
            raise ExpertAutomatedReviewError("reviewer slot is not configured")
        if (
            reviewer.reviewer_id
            == prepared.candidate_operation.proposer_authority.principal_id
        ):
            raise ExpertAutomatedReviewError(
                "candidate proposer cannot review its own output"
            )
        template = self.operation_template()
        schema = self.response_schema(
            self.settings.validation.policy.promotion,
        )
        prompt_payload = self._prompt_payload(prepared, reviewer)
        prompt = template.replace(
            _PROMPT_PACKET_MARKER,
            canonical_json_bytes(prompt_payload).decode("utf-8"),
        )
        if len(prompt.encode("utf-8")) > prepared.packet.agent_artifact_byte_limit:
            raise ExpertAutomatedReviewError(
                "automated review prompt exceeds the configured artifact limit"
            )
        request, operation_preimage = self._request(
            prepared=prepared,
            reviewer=reviewer,
            workspace=workspace,
            prompt=prompt,
            schema=schema,
        )
        result = self.runner.run(request, schema)
        sealed = seal_coding_agent_operation(
            request=request,
            response_schema=schema,
            principal_id=reviewer.reviewer_id,
            agent=reviewer.agent,
            sensitive_file_glob_scan_max_depth=(
                self.settings.sensitive_file_glob_scan_max_depth
            ),
            result=result,
        )
        if sum(len(payload) for payload in sealed.artifact_bytes.values()) > (
            prepared.packet.agent_artifact_byte_limit
        ):
            raise ExpertAutomatedReviewError(
                "automated review artifacts exceed the configured limit"
            )
        assertion = self._parse_assertion(
            prepared.packet,
            reviewer,
            sealed.receipt.operation_receipt_id,
            sealed.final_output,
        )
        operation = ExpertAutomatedReviewOperationRecord.mint(
            review_packet_id=prepared.packet.review_packet_id,
            operation_preimage=operation_preimage,
            operation_receipt=sealed.receipt,
            final_output=sealed.final_output,
            artifact_payloads_base64={
                name: base64.b64encode(payload).decode("ascii")
                for name, payload in sorted(sealed.artifact_bytes.items())
            },
            produced_assertion_id=assertion.assertion_id,
        )
        validate_expert_automated_review_operation(
            packet=prepared.packet,
            reviewer=reviewer,
            assertion=assertion,
            operation=operation,
            promotion=self.settings.validation.policy.promotion,
        )
        return assertion, operation

    def _request(
        self,
        *,
        prepared: PreparedExpertAutomatedReviewPacket,
        reviewer: ExpertReviewerSettings,
        workspace: Path,
        prompt: str,
        schema: Mapping[str, Any],
    ) -> tuple[CodingAgentCallRequest, Mapping[str, Any]]:
        provisional = CodingAgentCallRequest(
            operation_id=_PROVISIONAL_OPERATION_ID,
            role=reviewer.reviewer_role,
            cli=reviewer.agent.cli,
            model=reviewer.agent.model,
            prompt=prompt,
            workspace=str(workspace),
            workspace_policy=CodingAgentWorkspacePolicy.read_only(),
            timeout_seconds=reviewer.agent.timeout_seconds,
            effort=reviewer.agent.effort,
            allowed_tools=(),
            prior_knowledge=None,
        )
        input_artifacts = {
            "invocation.json": coding_agent_invocation_bytes(
                provisional,
                sensitive_file_glob_scan_max_depth=(
                    self.settings.sensitive_file_glob_scan_max_depth
                ),
            ),
            "prior_knowledge.json": b"null\n",
            "prompt.txt": prompt.encode("utf-8"),
            "response_schema.json": coding_agent_response_schema_bytes(schema),
        }
        operation_preimage = {
            "input_artifact_checksums": {
                name: tree_or_blob_digest(payload)
                for name, payload in sorted(input_artifacts.items())
            },
            "mcp_configuration_fingerprint": (
                coding_agent_mcp_configuration_fingerprint(None)
            ),
            "review_contract_version": EXPERT_AUTOMATED_REVIEW_CONTRACT_VERSION,
            "review_packet_id": prepared.packet.review_packet_id,
            "reviewer": reviewer.to_dict(),
            "sensitive_file_glob_scan_max_depth": (
                self.settings.sensitive_file_glob_scan_max_depth
            ),
            "validation_configuration_fingerprint": (
                prepared.packet.configuration_fingerprint
            ),
        }
        operation_id = (
            "agent_call_"
            + tree_or_blob_digest(canonical_json_bytes(operation_preimage))[7:39]
        )
        request = replace(provisional, operation_id=operation_id)
        if input_artifacts["invocation.json"] != coding_agent_invocation_bytes(
            request,
            sensitive_file_glob_scan_max_depth=(
                self.settings.sensitive_file_glob_scan_max_depth
            ),
        ):
            raise ExpertAutomatedReviewError(
                "review operation identity changes its invocation preimage"
            )
        return request, operation_preimage

    def _prompt_payload(
        self,
        prepared: PreparedExpertAutomatedReviewPacket,
        reviewer: ExpertReviewerSettings,
    ) -> Mapping[str, Any]:
        return expert_automated_review_prompt_payload(prepared, reviewer)

    @staticmethod
    def _review_stage_evidence(
        result: ExpertAcceptedReviewInput,
    ) -> Mapping[str, Any]:
        if type(result) is ExpertSourceReplayStageResultRecord:
            return result.to_dict()
        if type(result) is not ExpertEvaluatorResultRecord:
            raise ExpertAutomatedReviewError(
                "review evidence uses an unsupported stage result type"
            )
        run = result.evaluator_run
        if run.stage is not ExpertValidationStage.SEALED_CANARY:
            return result.to_dict()
        aggregate = ExpertSealedCanaryAggregate.from_json_bytes(
            base64.b64decode(
                run.output_payloads_base64["aggregate.json"],
                validate=True,
            )
        )
        return {
            "attestation": result.attestation_envelope.attestation.to_dict(),
            "evaluator_result_record_id": result.evaluator_result_record_id,
            "evaluator_run_id": run.evaluator_run_id,
            "outcome": run.outcome.value,
            "sealed_canary_aggregate": aggregate.to_dict(),
            "stage": run.stage.value,
        }

    def _parse_assertion(
        self,
        packet: ExpertAutomatedReviewPacket,
        reviewer: ExpertReviewerSettings,
        operation_receipt_id: str,
        output: str,
    ) -> ExpertAutomatedReviewAssertion:
        parsed = parse_json_bytes(output)
        if not isinstance(parsed, MappingABC) or set(parsed) != {
            "disposition",
            "judgment",
            "rationale",
        }:
            raise ExpertAutomatedReviewError(
                "automated reviewer output fields are invalid"
            )
        judgment = parsed["judgment"]
        rationale = parsed["rationale"]
        disposition_value = parsed["disposition"]
        promotion = self.settings.validation.policy.promotion
        if judgment not in {
            promotion.approval_judgment,
            promotion.rejection_judgment,
        }:
            raise ExpertAutomatedReviewError(
                "automated reviewer judgment is not configured"
            )
        if not isinstance(disposition_value, str) or disposition_value not in {
            disposition.value for disposition in ExpertReviewDisposition
        }:
            raise ExpertAutomatedReviewError(
                "automated reviewer disposition is invalid"
            )
        if not isinstance(rationale, str) or not rationale.strip():
            raise ExpertAutomatedReviewError(
                "automated reviewer rationale must be non-empty"
            )
        disposition = ExpertReviewDisposition(disposition_value)
        approved = judgment == promotion.approval_judgment
        if approved != (disposition is ExpertReviewDisposition.CORE_ELIGIBLE):
            raise ExpertAutomatedReviewError(
                "automated reviewer judgment and disposition conflict"
            )
        return ExpertAutomatedReviewAssertion.mint(
            review_packet_id=packet.review_packet_id,
            validation_attempt_id=packet.validation_attempt_id,
            candidate_id=packet.candidate_id,
            candidate_tree_hash=packet.candidate_tree_hash,
            parent_release_id=packet.parent_release_id,
            reviewer_id=reviewer.reviewer_id,
            reviewer_role=reviewer.reviewer_role,
            rubric_version=reviewer.rubric_version,
            judgment=judgment,
            disposition=disposition,
            rationale=rationale,
            exact_evidence_ids=packet.evidence_ids,
            review_operation_receipt_id=operation_receipt_id,
        )

    @staticmethod
    def operation_template() -> str:
        template = _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
        if template.count(_PROMPT_PACKET_MARKER) != 1:
            raise ExpertAutomatedReviewError(
                "automated reviewer template marker is invalid"
            )
        return template

    @staticmethod
    def response_schema(
        promotion: ExpertPromotionPolicySettings,
    ) -> Mapping[str, Any]:
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": ["disposition", "judgment", "rationale"],
            "properties": {
                "disposition": {
                    "type": "string",
                    "enum": [
                        disposition.value for disposition in ExpertReviewDisposition
                    ],
                },
                "judgment": {
                    "type": "string",
                    "enum": [
                        promotion.approval_judgment,
                        promotion.rejection_judgment,
                    ],
                },
                "rationale": {"type": "string", "minLength": 1},
            },
        }

    def _validate_preparation(
        self,
        *,
        stored_candidate: StoredExpertCandidate,
        candidate_input: ExpertCandidateAncestorInput,
        attempt: ExpertValidationAttempt,
        authorization_state: ExpertCandidateValidationState,
        accepted_stage_results: tuple[ExpertAcceptedReviewInput, ...],
        policy: ExpertValidationPolicy,
    ) -> None:
        closure = stored_candidate.closure
        if (
            authorization_state.promotion_state is not ExpertPromotionState.VALIDATING
            or authorization_state.next_stage
            is not ExpertValidationStage.AUTOMATED_REVIEW
            or authorization_state.review_assertion_ids
            or authorization_state.terminal_evidence_ids
            or authorization_state.validation_attempt_id
            != attempt.validation_attempt_id
            or authorization_state.candidate_id != attempt.candidate_id
            or authorization_state.candidate_tree_hash != attempt.candidate_tree_hash
            or attempt.candidate_id != closure.manifest.candidate_id
            or attempt.candidate_tree_hash != closure.manifest.candidate_tree_hash
            or attempt.candidate_commit_record_id
            != stored_candidate.commit_record.commit_record_id
            or attempt.scope_contract_id != closure.manifest.scope_contract_id
            or attempt.parent_release_id != closure.manifest.parent_release_id
            or attempt.validation_policy_id != policy.validation_policy_id
            or attempt.configuration_fingerprint
            != self.settings.validation.configuration_fingerprint
            or candidate_input.manifest != closure.manifest
        ):
            raise ExpertAutomatedReviewError(
                "candidate review preparation differs from its active authority"
            )
        reviewer_ids = {
            reviewer.reviewer_id
            for reviewer in self.settings.validation.policy.reviewers
        }
        if closure.operation.proposer_authority.principal_id in reviewer_ids:
            raise ExpertAutomatedReviewError(
                "historical candidate proposer conflicts with a reviewer"
            )
        if len(accepted_stage_results) != len(
            authorization_state.accepted_stage_results
        ):
            raise ExpertAutomatedReviewError(
                "automated review accepted evidence is incomplete"
            )
        if len(accepted_stage_results) >= len(attempt.required_stages):
            raise ExpertAutomatedReviewError(
                "automated review accepted evidence exhausts the stage plan"
            )
        expected_prefix = attempt.required_stages[: len(accepted_stage_results)]
        if (
            not expected_prefix
            or expected_prefix[-1] is ExpertValidationStage.AUTOMATED_REVIEW
            or attempt.required_stages[len(accepted_stage_results)]
            is not ExpertValidationStage.AUTOMATED_REVIEW
        ):
            raise ExpertAutomatedReviewError(
                "automated review does not follow the exact accepted stage prefix"
            )
        evaluators = {
            evaluator.stage: evaluator for evaluator in policy.policy.evaluators
        }
        for position, result in enumerate(accepted_stage_results):
            expected_stage = expected_prefix[position]
            accepted_ref = authorization_state.accepted_stage_results[position]
            if expected_stage is ExpertValidationStage.SOURCE_RUN_REPLAY:
                if (
                    type(result) is not ExpertSourceReplayStageResultRecord
                    or result.stage_result_record_id
                    != accepted_ref.stage_result_record_id
                    or accepted_ref.stage is not expected_stage
                    or result.outcome is not ExpertEvaluatorOutcome.PASSED
                    or result.validation_attempt_id != attempt.validation_attempt_id
                    or result.candidate_id != attempt.candidate_id
                    or result.candidate_tree_hash != attempt.candidate_tree_hash
                    or result.validation_policy_id != attempt.validation_policy_id
                    or result.configuration_fingerprint
                    != attempt.configuration_fingerprint
                ):
                    raise ExpertAutomatedReviewError(
                        "source replay review evidence differs from the prefix"
                    )
                continue
            evaluator = evaluators.get(expected_stage)
            if type(result) is not ExpertEvaluatorResultRecord or evaluator is None:
                raise ExpertAutomatedReviewError(
                    "ordinary review evidence uses the wrong stage result type"
                )
            run = result.evaluator_run
            if (
                accepted_ref.stage is not expected_stage
                or accepted_ref.stage_result_record_id
                != result.evaluator_result_record_id
                or run.stage is not expected_stage
                or run.outcome is not ExpertEvaluatorOutcome.PASSED
                or run.validation_attempt_id != attempt.validation_attempt_id
                or run.candidate_id != attempt.candidate_id
                or run.candidate_tree_hash != attempt.candidate_tree_hash
                or run.evaluator_id != evaluator.evaluator_id
                or run.evaluator_role != evaluator.evaluator_role
                or run.evaluator_version != evaluator.evaluator_version
            ):
                raise ExpertAutomatedReviewError(
                    "ordinary review evidence differs from the accepted prefix"
                )

    @staticmethod
    def _validate_review_workspace(workspace: Path) -> None:
        if (
            not workspace.is_absolute()
            or workspace != Path(os.path.abspath(workspace))
            or workspace.is_symlink()
            or not workspace.is_dir()
            or any(workspace.iterdir())
        ):
            raise ExpertAutomatedReviewError(
                "automated review workspace must be an empty normalized directory"
            )
        status = os.stat(workspace, follow_symlinks=False)
        if not stat.S_ISDIR(status.st_mode) or status.st_mode & (
            stat.S_IRWXG | stat.S_IRWXO
        ):
            raise ExpertAutomatedReviewError(
                "automated review workspace must be private"
            )

    @staticmethod
    def _validate_workspace_root(workspace_root: Path) -> None:
        if (
            not isinstance(workspace_root, Path)
            or not workspace_root.is_absolute()
            or workspace_root != Path(os.path.abspath(workspace_root))
            or workspace_root.is_symlink()
            or not workspace_root.is_dir()
            or workspace_root.resolve() != workspace_root
        ):
            raise ExpertAutomatedReviewError(
                "automated review workspace root must be an authorized real directory"
            )
        metadata = workspace_root.stat(follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_mode & (
            stat.S_IRWXG | stat.S_IRWXO
        ):
            raise ExpertAutomatedReviewError(
                "automated review workspace root must be private"
            )


def expert_automated_review_prompt_payload(
    prepared: PreparedExpertAutomatedReviewPacket,
    reviewer: ExpertReviewerSettings,
) -> Mapping[str, Any]:
    return {
        "accepted_stage_evidence": tuple(
            ExpertAutomatedReviewCoordinator._review_stage_evidence(result)
            for result in prepared.accepted_stage_results
        ),
        "candidate_input": prepared.candidate_input.to_dict(),
        "candidate_proposer": {
            "operation_record_id": prepared.candidate_operation.operation_record_id,
            "operation_receipt_id": (
                prepared.candidate_operation.operation_receipt.operation_receipt_id
            ),
            "proposer_authority": (
                prepared.candidate_operation.proposer_authority.to_dict()
            ),
        },
        "promotion_policy": prepared.validation_policy.policy.promotion.to_dict(),
        "review_packet": prepared.packet.to_dict(),
        "reviewer_slot": reviewer.to_dict(),
    }


def validate_expert_automated_review_facts(
    *,
    prepared: PreparedExpertAutomatedReviewPacket,
    assertions: tuple[ExpertAutomatedReviewAssertion, ...],
    operation_records: tuple[ExpertAutomatedReviewOperationRecord, ...],
    adjudication: ExpertAutomatedReviewAdjudication,
    stage_result: ExpertAutomatedReviewStageResultRecord,
) -> None:
    policy = prepared.validation_policy.policy
    reviewers = {reviewer.reviewer_id: reviewer for reviewer in policy.reviewers}
    assertions_by_reviewer = {
        assertion.reviewer_id: assertion for assertion in assertions
    }
    operations_by_reviewer = {
        operation.operation_receipt.principal_id: operation
        for operation in operation_records
    }
    if (
        set(assertions_by_reviewer) != set(reviewers)
        or set(operations_by_reviewer) != set(reviewers)
        or len(assertions_by_reviewer) != len(assertions)
        or len(operations_by_reviewer) != len(operation_records)
    ):
        raise ExpertAutomatedReviewError(
            "automated review fact closure does not match configured reviewers"
        )
    template = ExpertAutomatedReviewCoordinator.operation_template()
    schema = ExpertAutomatedReviewCoordinator.response_schema(policy.promotion)
    for reviewer_id in sorted(reviewers):
        reviewer = reviewers[reviewer_id]
        assertion = assertions_by_reviewer[reviewer_id]
        operation = operations_by_reviewer[reviewer_id]
        validate_expert_automated_review_operation(
            packet=prepared.packet,
            reviewer=reviewer,
            assertion=assertion,
            operation=operation,
            promotion=policy.promotion,
        )
        prompt = template.replace(
            _PROMPT_PACKET_MARKER,
            canonical_json_bytes(
                expert_automated_review_prompt_payload(prepared, reviewer)
            ).decode("utf-8"),
        )
        payloads = {
            name: base64.b64decode(payload, validate=True)
            for name, payload in operation.artifact_payloads_base64.items()
        }
        if sum(len(payload) for payload in payloads.values()) > (
            prepared.packet.agent_artifact_byte_limit
        ):
            raise ExpertAutomatedReviewError(
                "automated review artifacts exceed the configured limit"
            )
        scan_depth = operation.operation_preimage[
            "sensitive_file_glob_scan_max_depth"
        ]
        expected_request = CodingAgentCallRequest(
            operation_id=operation.operation_receipt.operation_id,
            role=reviewer.reviewer_role,
            cli=reviewer.agent.cli,
            model=reviewer.agent.model,
            prompt=prompt,
            workspace="/",
            workspace_policy=CodingAgentWorkspacePolicy.read_only(),
            timeout_seconds=reviewer.agent.timeout_seconds,
            effort=reviewer.agent.effort,
            allowed_tools=(),
            prior_knowledge=None,
        )
        expected_inputs = {
            "invocation.json": coding_agent_invocation_bytes(
                expected_request,
                sensitive_file_glob_scan_max_depth=scan_depth,
            ),
            "prior_knowledge.json": b"null\n",
            "prompt.txt": prompt.encode("utf-8"),
            "response_schema.json": coding_agent_response_schema_bytes(schema),
        }
        expected_preimage = {
            "input_artifact_checksums": {
                name: tree_or_blob_digest(payload)
                for name, payload in sorted(expected_inputs.items())
            },
            "mcp_configuration_fingerprint": (
                coding_agent_mcp_configuration_fingerprint(None)
            ),
            "review_contract_version": EXPERT_AUTOMATED_REVIEW_CONTRACT_VERSION,
            "review_packet_id": prepared.packet.review_packet_id,
            "reviewer": reviewer.to_dict(),
            "sensitive_file_glob_scan_max_depth": scan_depth,
            "validation_configuration_fingerprint": (
                prepared.packet.configuration_fingerprint
            ),
        }
        if (
            canonical_json_bytes(operation.operation_preimage)
            != canonical_json_bytes(expected_preimage)
            or any(payloads[name] != payload for name, payload in expected_inputs.items())
        ):
            raise ExpertAutomatedReviewError(
                "automated review operation did not use the canonical packet prompt"
            )
    expected_adjudication = adjudicate_expert_automated_review(
        packet=prepared.packet,
        validation_policy=prepared.validation_policy,
        assertions=assertions,
        operation_records=operation_records,
    )
    expected_result = build_expert_automated_review_stage_result(
        prepared=prepared,
        assertions=assertions,
        operation_records=operation_records,
        adjudication=expected_adjudication,
    )
    if adjudication != expected_adjudication or stage_result != expected_result:
        raise ExpertAutomatedReviewError(
            "automated review aggregate differs from canonical facts"
        )


def validate_expert_automated_review_operation(
    *,
    packet: ExpertAutomatedReviewPacket,
    reviewer: ExpertReviewerSettings,
    assertion: ExpertAutomatedReviewAssertion,
    operation: ExpertAutomatedReviewOperationRecord,
    promotion: ExpertPromotionPolicySettings,
) -> None:
    receipt = operation.operation_receipt
    parsed = parse_json_bytes(operation.final_output)
    expected_output = {
        "disposition": assertion.disposition.value,
        "judgment": assertion.judgment,
        "rationale": assertion.rationale,
    }
    approved = assertion.judgment == promotion.approval_judgment
    if (
        operation.review_packet_id != packet.review_packet_id
        or operation.produced_assertion_id != assertion.assertion_id
        or canonical_json_bytes(operation.operation_preimage["reviewer"])
        != canonical_json_bytes(reviewer.to_dict())
        or assertion.review_packet_id != packet.review_packet_id
        or assertion.validation_attempt_id != packet.validation_attempt_id
        or assertion.candidate_id != packet.candidate_id
        or assertion.candidate_tree_hash != packet.candidate_tree_hash
        or assertion.parent_release_id != packet.parent_release_id
        or assertion.reviewer_id != reviewer.reviewer_id
        or assertion.reviewer_role != reviewer.reviewer_role
        or assertion.rubric_version != reviewer.rubric_version
        or assertion.exact_evidence_ids != packet.evidence_ids
        or assertion.review_operation_receipt_id != receipt.operation_receipt_id
        or parsed != expected_output
        or assertion.judgment
        not in {promotion.approval_judgment, promotion.rejection_judgment}
        or approved != (assertion.disposition is ExpertReviewDisposition.CORE_ELIGIBLE)
    ):
        raise ExpertAutomatedReviewError(
            "automated review operation, assertion, or packet binding differs"
        )


def _validate_prepared_review_stage_results(
    prepared: PreparedExpertAutomatedReviewPacket,
) -> None:
    packet = prepared.packet
    attempt = prepared.validation_attempt
    evaluators = {
        evaluator.stage: evaluator
        for evaluator in prepared.validation_policy.policy.evaluators
    }
    for position, result in enumerate(prepared.accepted_stage_results):
        reference = packet.accepted_stage_results[position]
        expected_stage = attempt.required_stages[position]
        if reference.stage is not expected_stage:
            raise ExpertAutomatedReviewError(
                "prepared review evidence stage differs from its ordered prefix"
            )
        if expected_stage is ExpertValidationStage.SOURCE_RUN_REPLAY:
            if (
                type(result) is not ExpertSourceReplayStageResultRecord
                or reference.stage_result_record_id != result.stage_result_record_id
                or result.outcome is not ExpertEvaluatorOutcome.PASSED
                or result.validation_attempt_id != attempt.validation_attempt_id
                or result.candidate_id != attempt.candidate_id
                or result.candidate_tree_hash != attempt.candidate_tree_hash
                or result.validation_policy_id != attempt.validation_policy_id
                or result.configuration_fingerprint != attempt.configuration_fingerprint
                or result.publication_authority_fence.scope_contract_id
                != attempt.scope_contract_id
                or result.publication_authority_fence.expected_parent_release_id
                != attempt.parent_release_id
            ):
                raise ExpertAutomatedReviewError(
                    "prepared source replay evidence differs from its reference"
                )
            continue
        evaluator = evaluators.get(expected_stage)
        if type(result) is not ExpertEvaluatorResultRecord or evaluator is None:
            raise ExpertAutomatedReviewError(
                "prepared ordinary evidence uses the wrong result type"
            )
        run = result.evaluator_run
        if (
            reference.stage_result_record_id != result.evaluator_result_record_id
            or run.stage is not expected_stage
            or run.outcome is not ExpertEvaluatorOutcome.PASSED
            or run.validation_attempt_id != attempt.validation_attempt_id
            or run.candidate_id != attempt.candidate_id
            or run.candidate_tree_hash != attempt.candidate_tree_hash
            or run.evaluator_id != evaluator.evaluator_id
            or run.evaluator_role != evaluator.evaluator_role
            or run.evaluator_version != evaluator.evaluator_version
        ):
            raise ExpertAutomatedReviewError(
                "prepared ordinary evidence differs from its exact reference"
            )


def adjudicate_expert_automated_review(
    *,
    packet: ExpertAutomatedReviewPacket,
    validation_policy: ExpertValidationPolicy,
    assertions: tuple[ExpertAutomatedReviewAssertion, ...],
    operation_records: tuple[ExpertAutomatedReviewOperationRecord, ...],
) -> ExpertAutomatedReviewAdjudication:
    policy = validation_policy.policy
    promotion = policy.promotion
    reviewers = {reviewer.reviewer_id: reviewer for reviewer in policy.reviewers}
    assertion_reviewers = tuple(assertion.reviewer_id for assertion in assertions)
    operation_reviewers = tuple(
        operation.operation_receipt.principal_id for operation in operation_records
    )
    if (
        assertion_reviewers != tuple(sorted(reviewers))
        or operation_reviewers != tuple(sorted(reviewers))
        or len(assertions) != len(operation_records)
    ):
        raise ExpertAutomatedReviewError(
            "automated review requires exactly one result from every reviewer"
        )
    operation_by_reviewer = {
        operation.operation_receipt.principal_id: operation
        for operation in operation_records
    }
    approvals = []
    rejections = []
    for assertion in assertions:
        reviewer = reviewers[assertion.reviewer_id]
        validate_expert_automated_review_operation(
            packet=packet,
            reviewer=reviewer,
            assertion=assertion,
            operation=operation_by_reviewer[assertion.reviewer_id],
            promotion=promotion,
        )
        if assertion.judgment == promotion.approval_judgment:
            approvals.append(assertion.reviewer_id)
        else:
            rejections.append(assertion.reviewer_id)
    if approvals and rejections:
        outcome = ExpertAutomatedReviewOutcome.DISPUTED
    elif len(rejections) >= promotion.required_rejections:
        outcome = ExpertAutomatedReviewOutcome.REJECTED
    elif len(approvals) >= promotion.required_approvals:
        outcome = ExpertAutomatedReviewOutcome.PASSED
    else:
        outcome = ExpertAutomatedReviewOutcome.DISPUTED
    if outcome is ExpertAutomatedReviewOutcome.DISPUTED and len(assertions) < 2:
        raise ExpertAutomatedReviewError(
            "a disputed review requires at least two reviewer assertions"
        )
    return ExpertAutomatedReviewAdjudication.mint(
        review_packet_id=packet.review_packet_id,
        validation_policy_id=validation_policy.validation_policy_id,
        promotion_policy_version=promotion.policy_version,
        assertion_ids=tuple(sorted(assertion.assertion_id for assertion in assertions)),
        approval_reviewer_ids=tuple(sorted(approvals)),
        rejection_reviewer_ids=tuple(sorted(rejections)),
        outcome=outcome,
    )


def build_expert_automated_review_stage_result(
    *,
    prepared: PreparedExpertAutomatedReviewPacket,
    assertions: tuple[ExpertAutomatedReviewAssertion, ...],
    operation_records: tuple[ExpertAutomatedReviewOperationRecord, ...],
    adjudication: ExpertAutomatedReviewAdjudication,
) -> ExpertAutomatedReviewStageResultRecord:
    packet = prepared.packet
    assertion_ids = tuple(sorted(assertion.assertion_id for assertion in assertions))
    operation_ids = tuple(
        sorted(operation.operation_record_id for operation in operation_records)
    )
    receipt_ids = tuple(
        sorted(
            operation.operation_receipt.operation_receipt_id
            for operation in operation_records
        )
    )
    recomputed_adjudication = adjudicate_expert_automated_review(
        packet=packet,
        validation_policy=prepared.validation_policy,
        assertions=assertions,
        operation_records=operation_records,
    )
    if (
        adjudication != recomputed_adjudication
        or adjudication.review_packet_id != packet.review_packet_id
        or adjudication.validation_policy_id != packet.validation_policy_id
        or adjudication.assertion_ids != assertion_ids
    ):
        raise ExpertAutomatedReviewError(
            "automated review adjudication differs from its exact facts"
        )
    dependencies = {
        packet.validation_attempt_id,
        packet.authorization_transition_id,
        packet.authorization_state_id,
        packet.candidate_id,
        packet.scope_contract_id,
        packet.validation_policy_id,
        packet.review_packet_id,
        adjudication.adjudication_id,
        *assertion_ids,
        *operation_ids,
        *receipt_ids,
    }
    if packet.parent_release_id is not None:
        dependencies.add(packet.parent_release_id)
    return ExpertAutomatedReviewStageResultRecord.mint(
        validation_attempt_id=packet.validation_attempt_id,
        authorization_transition_id=packet.authorization_transition_id,
        authorization_state_id=packet.authorization_state_id,
        candidate_id=packet.candidate_id,
        candidate_tree_hash=packet.candidate_tree_hash,
        scope_contract_id=packet.scope_contract_id,
        parent_release_id=packet.parent_release_id,
        validation_policy_id=packet.validation_policy_id,
        configuration_fingerprint=packet.configuration_fingerprint,
        review_packet_id=packet.review_packet_id,
        assertion_ids=assertion_ids,
        operation_record_ids=operation_ids,
        operation_receipt_ids=receipt_ids,
        adjudication_id=adjudication.adjudication_id,
        outcome=adjudication.outcome,
        exact_dependency_ids=tuple(sorted(dependencies)),
    )
