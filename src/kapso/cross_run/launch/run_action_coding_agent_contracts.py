"""Path-free canonical contracts for coding-agent run actions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    require_content_id,
    require_identifier,
    tree_or_blob_digest,
)
from kapso.cross_run.coding_agent_compatibility import (
    coding_agent_supported_efforts,
    coding_agent_supported_tools,
)
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.knowledge.access import (
    PriorKnowledgeAccess,
    PriorKnowledgeAccessMaterialization,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_coding_agent_schema import (
    validate_run_action_coding_agent_schema,
)

CODING_AGENT_REQUEST_PROTOCOL_VERSION = "kapso.run_action.coding_agent_request.v1"
CODING_AGENT_RESULT_PROTOCOL_VERSION = "kapso.run_action.coding_agent_result.v1"
CODING_AGENT_SCHEMA_PROTOCOL_VERSION = "json-schema.draft-2020-12"

_INTERPRETATION_POLICY_NAMESPACE = "run-action-coding-agent-interpretation-policy"
_OPERATION_ID_PATTERN = re.compile(r"^agent_call_[0-9a-f]{32}$")
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_NORMALIZED_DECIMAL_PATTERN = re.compile(r"^(?:0|[1-9][0-9]*)(?:[.][0-9]*[1-9])?$")
_CODING_AGENT_WORKSPACE_ACCESS = frozenset(
    {
        RunFrontierWorkspaceAccess.READ_ONLY,
        RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
    }
)


class RunActionCodingAgentContractError(ValueError):
    """A coding-agent run-action contract or semantic join is invalid."""


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunActionCodingAgentContractError(f"{name} must be non-empty text")
    return value


def _require_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise RunActionCodingAgentContractError(f"{name} must be a sha256 digest")
    return value


def _require_nonnegative_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise RunActionCodingAgentContractError(
            f"{name} must be a non-negative integer"
        )
    return value


def _require_positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise RunActionCodingAgentContractError(f"{name} must be a positive integer")
    return value


def _require_policy_id(value: object) -> str:
    require_content_id(value, "coding-agent interpretation policy ID")
    if value.split(":sha256:", 1)[0] != _INTERPRETATION_POLICY_NAMESPACE:
        raise RunActionCodingAgentContractError(
            "coding-agent interpretation policy ID uses the wrong namespace"
        )
    return value


class CodingAgentPriorKnowledgeAccessKind(str, Enum):
    """The two semantic reads exposed by the prior-knowledge gate."""

    LIST = "list"
    GET = "get"


@dataclass(frozen=True)
class CodingAgentInterpretationPolicy(StrictContract):
    """Immutable authority for one coding-agent request/result interpretation."""

    interpretation_policy_id: str
    request_protocol_version: str
    result_protocol_version: str
    schema_protocol_version: str
    consumer_id: str
    consumer_version: str
    principal_id: str
    role: str
    cli: str
    model: str
    effort: str
    allowed_tools: tuple[str, ...]
    timeout_nanoseconds: int
    workspace_access: RunFrontierWorkspaceAccess
    maximum_raw_result_bytes: int

    CONTENT_NAMESPACE: ClassVar[str] = _INTERPRETATION_POLICY_NAMESPACE
    IDENTITY_FIELD: ClassVar[str] = "interpretation_policy_id"

    def _validate(self) -> None:
        if (
            self.request_protocol_version != CODING_AGENT_REQUEST_PROTOCOL_VERSION
            or self.result_protocol_version != CODING_AGENT_RESULT_PROTOCOL_VERSION
            or self.schema_protocol_version != CODING_AGENT_SCHEMA_PROTOCOL_VERSION
        ):
            raise RunActionCodingAgentContractError(
                "coding-agent interpretation policy uses an unknown protocol"
            )
        for value, name in (
            (self.consumer_id, "coding-agent consumer ID"),
            (self.consumer_version, "coding-agent consumer version"),
            (self.principal_id, "coding-agent principal ID"),
            (self.role, "coding-agent role"),
        ):
            require_identifier(value, name)
        if self.cli not in {"codex", "claude_code"}:
            raise RunActionCodingAgentContractError(
                "coding-agent CLI must be codex or claude_code"
            )
        _require_text(self.model, "coding-agent model")
        if self.effort not in coding_agent_supported_efforts(self.cli):
            raise RunActionCodingAgentContractError(
                "coding-agent effort is incompatible with its CLI"
            )
        if self.allowed_tools != tuple(sorted(set(self.allowed_tools))):
            raise RunActionCodingAgentContractError(
                "coding-agent allowed tools must be sorted and unique"
            )
        if any(
            not isinstance(tool, str)
            or tool
            not in coding_agent_supported_tools(
                self.cli,
                edit_workspace=(
                    self.workspace_access is RunFrontierWorkspaceAccess.EDIT_WORKSPACE
                ),
            )
            for tool in self.allowed_tools
        ):
            raise RunActionCodingAgentContractError(
                "coding-agent policy contains an unsupported tool"
            )
        if self.workspace_access not in _CODING_AGENT_WORKSPACE_ACCESS:
            raise RunActionCodingAgentContractError(
                "coding-agent policy requires read-only or edit workspace access"
            )
        _require_positive_integer(
            self.timeout_nanoseconds,
            "coding-agent timeout nanoseconds",
        )
        _require_positive_integer(
            self.maximum_raw_result_bytes,
            "coding-agent maximum raw-result bytes",
        )


@dataclass(frozen=True)
class CodingAgentRunActionRequest(StrictContract):
    """Complete path-free input to one coding-agent run action."""

    protocol_version: str
    interpretation_policy_id: str
    operation_id: str
    prompt: str
    response_schema: Mapping[str, Any]
    prior_knowledge: PriorKnowledgeAccessMaterialization | None
    edit_predecessor_source_tree_digest: str | None

    def _validate(self) -> None:
        if self.protocol_version != CODING_AGENT_REQUEST_PROTOCOL_VERSION:
            raise RunActionCodingAgentContractError(
                "coding-agent request uses an unknown protocol"
            )
        _require_policy_id(self.interpretation_policy_id)
        if (
            not isinstance(self.operation_id, str)
            or _OPERATION_ID_PATTERN.fullmatch(self.operation_id) is None
        ):
            raise RunActionCodingAgentContractError(
                "coding-agent operation ID must be agent_call_<32 lowercase hex>"
            )
        _require_text(self.prompt, "coding-agent prompt")
        if not isinstance(self.response_schema, Mapping):
            raise RunActionCodingAgentContractError(
                "coding-agent response schema must be an object"
            )
        validate_run_action_coding_agent_schema(self.response_schema)
        if self.prior_knowledge is not None and type(self.prior_knowledge) is not (
            PriorKnowledgeAccessMaterialization
        ):
            raise RunActionCodingAgentContractError(
                "coding-agent prior knowledge is invalid"
            )
        if self.edit_predecessor_source_tree_digest is not None:
            _require_digest(
                self.edit_predecessor_source_tree_digest,
                "coding-agent edit predecessor source tree",
            )

    @property
    def request_digest(self) -> str:
        return tree_or_blob_digest(self.to_json_bytes())

    def require_policy(self, policy: CodingAgentInterpretationPolicy) -> None:
        if type(policy) is not CodingAgentInterpretationPolicy:
            raise RunActionCodingAgentContractError(
                "coding-agent request requires an exact interpretation policy"
            )
        if self.interpretation_policy_id != policy.interpretation_policy_id:
            raise RunActionCodingAgentContractError(
                "coding-agent request names another interpretation policy"
            )
        editing = policy.workspace_access is RunFrontierWorkspaceAccess.EDIT_WORKSPACE
        if editing != (self.edit_predecessor_source_tree_digest is not None):
            raise RunActionCodingAgentContractError(
                "coding-agent request edit predecessor differs from workspace access"
            )


@dataclass(frozen=True)
class CodingAgentPriorKnowledgeAccessEvent(StrictContract):
    """One ordered semantic access to an immutable prior-knowledge packet."""

    access_kind: CodingAgentPriorKnowledgeAccessKind
    record_id: str | None
    returned_record_ids: tuple[str, ...]
    response_digest: str

    def _validate(self) -> None:
        _require_digest(
            self.response_digest,
            "coding-agent prior-knowledge response",
        )
        if self.returned_record_ids != tuple(sorted(set(self.returned_record_ids))):
            raise RunActionCodingAgentContractError(
                "prior-knowledge returned record IDs must be sorted and unique"
            )
        for record_id in self.returned_record_ids:
            require_content_id(
                record_id,
                "coding-agent prior-knowledge returned record ID",
            )
        if self.access_kind is CodingAgentPriorKnowledgeAccessKind.LIST:
            if self.record_id is not None:
                raise RunActionCodingAgentContractError(
                    "prior-knowledge list access cannot name one record"
                )
            return
        if self.record_id is None:
            raise RunActionCodingAgentContractError(
                "prior-knowledge get access requires one record ID"
            )
        require_content_id(
            self.record_id,
            "coding-agent prior-knowledge record ID",
        )
        if self.returned_record_ids != (self.record_id,):
            raise RunActionCodingAgentContractError(
                "prior-knowledge get access must return exactly its named record"
            )


@dataclass(frozen=True)
class CodingAgentRunActionResultEnvelope(StrictContract):
    """Canonical, path-free result that remains complete after runtime cleanup."""

    protocol_version: str
    consumer_id: str
    consumer_version: str
    operation_id: str
    request_digest: str
    structured_output: Mapping[str, Any]
    duration_nanoseconds: int
    input_tokens: int
    output_tokens: int
    cost_usd: str | None
    prior_knowledge_accesses: tuple[CodingAgentPriorKnowledgeAccessEvent, ...]
    edited_source_tree_digest: str | None

    def _validate(self) -> None:
        if self.protocol_version != CODING_AGENT_RESULT_PROTOCOL_VERSION:
            raise RunActionCodingAgentContractError(
                "coding-agent result uses an unknown protocol"
            )
        for value, name in (
            (self.consumer_id, "coding-agent consumer ID"),
            (self.consumer_version, "coding-agent consumer version"),
        ):
            require_identifier(value, name)
        if (
            not isinstance(self.operation_id, str)
            or _OPERATION_ID_PATTERN.fullmatch(self.operation_id) is None
        ):
            raise RunActionCodingAgentContractError(
                "coding-agent result operation ID is invalid"
            )
        _require_digest(self.request_digest, "coding-agent result request")
        if not isinstance(self.structured_output, Mapping):
            raise RunActionCodingAgentContractError(
                "coding-agent structured output must be an object"
            )
        for value, name in (
            (self.duration_nanoseconds, "coding-agent duration nanoseconds"),
            (self.input_tokens, "coding-agent input tokens"),
            (self.output_tokens, "coding-agent output tokens"),
        ):
            _require_nonnegative_integer(value, name)
        if self.cost_usd is not None and (
            not isinstance(self.cost_usd, str)
            or _NORMALIZED_DECIMAL_PATTERN.fullmatch(self.cost_usd) is None
        ):
            raise RunActionCodingAgentContractError(
                "coding-agent cost must be normalized non-negative decimal text"
            )
        if self.edited_source_tree_digest is not None:
            _require_digest(
                self.edited_source_tree_digest,
                "coding-agent edited source tree",
            )

    def validate_against(
        self,
        *,
        policy: CodingAgentInterpretationPolicy,
        request: CodingAgentRunActionRequest,
    ) -> None:
        if (
            type(policy) is not CodingAgentInterpretationPolicy
            or type(request) is not CodingAgentRunActionRequest
        ):
            raise RunActionCodingAgentContractError(
                "coding-agent result join requires exact policy and request contracts"
            )
        request.require_policy(policy)
        if (
            self.consumer_id != policy.consumer_id
            or self.consumer_version != policy.consumer_version
        ):
            raise RunActionCodingAgentContractError(
                "coding-agent result names another consumer"
            )
        if self.operation_id != request.operation_id:
            raise RunActionCodingAgentContractError(
                "coding-agent result names another operation"
            )
        if self.request_digest != request.request_digest:
            raise RunActionCodingAgentContractError(
                "coding-agent result names another request"
            )
        if len(self.to_json_bytes()) > policy.maximum_raw_result_bytes:
            raise RunActionCodingAgentContractError(
                "coding-agent result exceeds its raw-result byte limit"
            )
        editing = policy.workspace_access is RunFrontierWorkspaceAccess.EDIT_WORKSPACE
        if editing != (self.edited_source_tree_digest is not None):
            raise RunActionCodingAgentContractError(
                "coding-agent result edited tree differs from workspace access"
            )
        if editing and self.edited_source_tree_digest == (
            request.edit_predecessor_source_tree_digest
        ):
            raise RunActionCodingAgentContractError(
                "coding-agent edit result did not change the source tree"
            )
        self._validate_prior_knowledge_accesses(request.prior_knowledge)

    def _validate_prior_knowledge_accesses(
        self,
        prior_knowledge: PriorKnowledgeAccessMaterialization | None,
    ) -> None:
        if prior_knowledge is None:
            if self.prior_knowledge_accesses:
                raise RunActionCodingAgentContractError(
                    "coding-agent result records undeclared prior-knowledge access"
                )
            return
        access = PriorKnowledgeAccess(prior_knowledge)
        listed_ids = tuple(record["record_id"] for record in access.list_records())
        list_digest = tree_or_blob_digest(
            canonical_json_bytes(access.list_response_payload())
        )
        for event in self.prior_knowledge_accesses:
            if event.access_kind is CodingAgentPriorKnowledgeAccessKind.LIST:
                if (
                    event.returned_record_ids != listed_ids
                    or event.response_digest != list_digest
                ):
                    raise RunActionCodingAgentContractError(
                        "coding-agent prior-knowledge list access is inconsistent"
                    )
                continue
            record_digest = tree_or_blob_digest(
                canonical_json_bytes(access.record_response_payload(event.record_id))
            )
            if event.response_digest != record_digest:
                raise RunActionCodingAgentContractError(
                    "coding-agent prior-knowledge get access is inconsistent"
                )


__all__ = [
    "CODING_AGENT_REQUEST_PROTOCOL_VERSION",
    "CODING_AGENT_RESULT_PROTOCOL_VERSION",
    "CODING_AGENT_SCHEMA_PROTOCOL_VERSION",
    "CodingAgentInterpretationPolicy",
    "CodingAgentPriorKnowledgeAccessEvent",
    "CodingAgentPriorKnowledgeAccessKind",
    "CodingAgentRunActionRequest",
    "CodingAgentRunActionResultEnvelope",
    "RunActionCodingAgentContractError",
]
