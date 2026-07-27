"""Pure interpretation and network-free fixtures for coding-agent run actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from kapso.cross_run.canonical import canonical_json_bytes, freeze_json
from kapso.cross_run.launch.run_action_coding_agent_contracts import (
    CODING_AGENT_RESULT_PROTOCOL_VERSION,
    CodingAgentInterpretationPolicy,
    CodingAgentPriorKnowledgeAccessEvent,
    CodingAgentRunActionRequest,
    CodingAgentRunActionResultEnvelope,
    RunActionCodingAgentContractError,
)
from kapso.cross_run.launch.run_action_coding_agent_schema import (
    validate_run_action_coding_agent_output,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunActionResultInterpreterIdentity,
    RunFrontierActionKind,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_recovery import (
    RunActionInterpretedResult,
)
from kapso.cross_run.launch.run_action_store import (
    RunActionResultDisposition,
)

CODING_AGENT_RESULT_INTERPRETER_IMPLEMENTATION_ID = (
    "kapso.coding_agent_result_interpreter"
)
CODING_AGENT_RESULT_INTERPRETER_IMPLEMENTATION_VERSION = (
    "kapso.coding_agent_result_interpreter.v1"
)
CODING_AGENT_RESULT_INTERPRETATION_PROTOCOL_VERSION = (
    "kapso.coding_agent_result_interpretation.v1"
)


def coding_agent_result_interpreter_identity(
    policy: CodingAgentInterpretationPolicy,
) -> RunActionResultInterpreterIdentity:
    """Mint the sole identity admitted by the concrete pure interpreter."""

    if type(policy) is not CodingAgentInterpretationPolicy:
        raise RunActionCodingAgentContractError(
            "coding-agent interpreter identity requires an exact policy"
        )
    return RunActionResultInterpreterIdentity.mint(
        kind=RunFrontierActionKind.CODING_AGENT,
        implementation_id=CODING_AGENT_RESULT_INTERPRETER_IMPLEMENTATION_ID,
        implementation_version=(CODING_AGENT_RESULT_INTERPRETER_IMPLEMENTATION_VERSION),
        interpretation_protocol_version=(
            CODING_AGENT_RESULT_INTERPRETATION_PROTOCOL_VERSION
        ),
        interpretation_policy_id=policy.interpretation_policy_id,
    )


@dataclass(frozen=True, slots=True)
class CodingAgentRunActionResultInterpreter:
    """Dependency-pure interpreter for one content-addressed coding-agent policy."""

    result_interpreter_identity: RunActionResultInterpreterIdentity
    interpretation_policy: CodingAgentInterpretationPolicy

    def __post_init__(self) -> None:
        if (
            type(self.result_interpreter_identity)
            is not RunActionResultInterpreterIdentity
            or type(self.interpretation_policy) is not CodingAgentInterpretationPolicy
            or self.result_interpreter_identity
            != coding_agent_result_interpreter_identity(self.interpretation_policy)
        ):
            raise RunActionCodingAgentContractError(
                "coding-agent interpreter differs from its exact policy identity"
            )

    def interpret(
        self,
        *,
        operation_id: str,
        request_payload: bytes,
        result_payload: bytes,
    ) -> RunActionInterpretedResult:
        """Validate complete canonical bytes without consulting another authority."""

        if (
            type(result_payload) is not bytes
            or not result_payload
            or len(result_payload) > self.interpretation_policy.maximum_raw_result_bytes
        ):
            raise RunActionCodingAgentContractError(
                "coding-agent raw result exceeds its exact byte limit"
            )
        request = _read_canonical_request(request_payload)
        request.require_policy(self.interpretation_policy)
        if request.operation_id != operation_id:
            raise RunActionCodingAgentContractError(
                "coding-agent request names another durable operation"
            )
        result = _read_canonical_result(result_payload)
        result.validate_against(
            policy=self.interpretation_policy,
            request=request,
        )
        validate_run_action_coding_agent_output(
            request.response_schema,
            result.structured_output,
        )
        editing = (
            self.interpretation_policy.workspace_access
            is RunFrontierWorkspaceAccess.EDIT_WORKSPACE
        )
        return RunActionInterpretedResult(
            disposition=RunActionResultDisposition.SUCCEEDED,
            operation_id=request.operation_id,
            accepted_result_payload=canonical_json_bytes(result.structured_output),
            expected_workspace_before_source_tree_digest=(
                request.edit_predecessor_source_tree_digest if editing else None
            ),
            expected_workspace_after_source_tree_digest=(
                result.edited_source_tree_digest if editing else None
            ),
        )


@dataclass(frozen=True, slots=True)
class FixedOfflineCodingAgentConsumer:
    """Network-free deterministic consumer used to exercise the production seam."""

    interpretation_policy: CodingAgentInterpretationPolicy
    structured_output: Mapping[str, Any]
    duration_nanoseconds: int
    input_tokens: int
    output_tokens: int
    cost_usd: str | None
    prior_knowledge_accesses: tuple[CodingAgentPriorKnowledgeAccessEvent, ...]
    edited_source_tree_digest: str | None

    def __post_init__(self) -> None:
        if type(self.interpretation_policy) is not CodingAgentInterpretationPolicy:
            raise RunActionCodingAgentContractError(
                "offline coding-agent consumer requires an exact policy"
            )
        object.__setattr__(
            self,
            "structured_output",
            freeze_json(self.structured_output, "offline coding-agent output"),
        )
        if type(self.prior_knowledge_accesses) is not tuple or any(
            type(event) is not CodingAgentPriorKnowledgeAccessEvent
            for event in self.prior_knowledge_accesses
        ):
            raise RunActionCodingAgentContractError(
                "offline coding-agent consumer access events are invalid"
            )

    def consume(self, request_payload: bytes) -> bytes:
        """Return one canonical result bound to the complete request bytes."""

        request = _read_canonical_request(request_payload)
        request.require_policy(self.interpretation_policy)
        validate_run_action_coding_agent_output(
            request.response_schema,
            self.structured_output,
        )
        result = CodingAgentRunActionResultEnvelope(
            protocol_version=CODING_AGENT_RESULT_PROTOCOL_VERSION,
            consumer_id=self.interpretation_policy.consumer_id,
            consumer_version=self.interpretation_policy.consumer_version,
            operation_id=request.operation_id,
            request_digest=request.request_digest,
            structured_output=self.structured_output,
            duration_nanoseconds=self.duration_nanoseconds,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost_usd=self.cost_usd,
            prior_knowledge_accesses=self.prior_knowledge_accesses,
            edited_source_tree_digest=self.edited_source_tree_digest,
        )
        result.validate_against(
            policy=self.interpretation_policy,
            request=request,
        )
        return result.to_json_bytes()


def _read_canonical_request(payload: bytes) -> CodingAgentRunActionRequest:
    if type(payload) is not bytes or not payload:
        raise RunActionCodingAgentContractError(
            "coding-agent request payload must be complete bytes"
        )
    request = CodingAgentRunActionRequest.from_json_bytes(payload)
    if request.to_json_bytes() != payload:
        raise RunActionCodingAgentContractError(
            "coding-agent request payload is not canonical"
        )
    return request


def _read_canonical_result(payload: bytes) -> CodingAgentRunActionResultEnvelope:
    if type(payload) is not bytes or not payload:
        raise RunActionCodingAgentContractError(
            "coding-agent result payload must be complete bytes"
        )
    result = CodingAgentRunActionResultEnvelope.from_json_bytes(payload)
    if result.to_json_bytes() != payload:
        raise RunActionCodingAgentContractError(
            "coding-agent result payload is not canonical"
        )
    return result


__all__ = [
    "CODING_AGENT_RESULT_INTERPRETATION_PROTOCOL_VERSION",
    "CODING_AGENT_RESULT_INTERPRETER_IMPLEMENTATION_ID",
    "CODING_AGENT_RESULT_INTERPRETER_IMPLEMENTATION_VERSION",
    "CodingAgentRunActionResultInterpreter",
    "FixedOfflineCodingAgentConsumer",
    "coding_agent_result_interpreter_identity",
]
