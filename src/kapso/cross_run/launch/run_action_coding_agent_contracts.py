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
    CODING_AGENT_LANDLOCK_POLICY_ABI_VERSION,
    coding_agent_supported_efforts,
)
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.git_refs import require_git_ref_name
from kapso.cross_run.knowledge.access import (
    PriorKnowledgeAccess,
    PriorKnowledgeAccessMaterialization,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_coding_agent_schema import (
    validate_run_action_coding_agent_provider_schema,
)

CODING_AGENT_REQUEST_PROTOCOL_VERSION = "kapso.run_action.coding_agent_request.v1"
CODING_AGENT_RESULT_PROTOCOL_VERSION = "kapso.run_action.coding_agent_result.v1"
CODING_AGENT_SCHEMA_PROTOCOL_VERSION = "json-schema.draft-2020-12"
CODING_AGENT_NATIVE_TOOL_POLICY_VERSION = "kapso.coding_agent_native_tools.v1"
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
    native_tool_policy_version: str
    web_search_enabled: bool
    timeout_nanoseconds: int
    termination_grace_nanoseconds: int
    supervisor_user_id: int
    supervisor_group_id: int
    provider_user_id: int
    provider_group_id: int
    landlock_abi_version: int
    workspace_access: RunFrontierWorkspaceAccess
    workspace_git_branch: str
    git_commit_author_name: str
    git_commit_author_email: str
    maximum_request_bytes: int
    maximum_response_schema_bytes: int
    maximum_cli_argument_bytes: int
    maximum_provider_output_bytes: int
    maximum_provider_diagnostic_bytes: int
    maximum_prior_knowledge_audit_bytes: int
    maximum_native_credential_bytes: int
    maximum_workspace_entries: int
    maximum_workspace_bytes: int
    maximum_workspace_git_entries: int
    maximum_workspace_git_bytes: int
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
        if self.native_tool_policy_version != CODING_AGENT_NATIVE_TOOL_POLICY_VERSION:
            raise RunActionCodingAgentContractError(
                "coding-agent native-tool policy version is unknown"
            )
        if type(self.web_search_enabled) is not bool:
            raise RunActionCodingAgentContractError(
                "coding-agent web-search authority must be boolean"
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
            self.termination_grace_nanoseconds,
            "coding-agent termination-grace nanoseconds",
        )
        if self.termination_grace_nanoseconds >= self.timeout_nanoseconds:
            raise RunActionCodingAgentContractError(
                "coding-agent termination grace must be below its timeout"
            )
        for value, name in (
            (self.supervisor_user_id, "coding-agent supervisor user ID"),
            (self.supervisor_group_id, "coding-agent supervisor group ID"),
            (self.provider_user_id, "coding-agent provider user ID"),
            (self.provider_group_id, "coding-agent provider group ID"),
        ):
            if type(value) is not int or not 0 < value <= 2_147_483_647:
                raise RunActionCodingAgentContractError(
                    f"{name} must be a positive Linux identity"
                )
        if (
            self.supervisor_user_id == self.provider_user_id
            or self.supervisor_group_id == self.provider_group_id
        ):
            raise RunActionCodingAgentContractError(
                "coding-agent supervisor and provider identities must differ"
            )
        if (
            type(self.landlock_abi_version) is not int
            or self.landlock_abi_version != CODING_AGENT_LANDLOCK_POLICY_ABI_VERSION
        ):
            raise RunActionCodingAgentContractError(
                "coding-agent Landlock ABI differs from the implemented policy"
            )
        if not isinstance(self.workspace_git_branch, str):
            raise RunActionCodingAgentContractError(
                "coding-agent workspace Git branch must be text"
            )
        require_git_ref_name(
            f"refs/heads/{self.workspace_git_branch}",
            "coding-agent workspace Git branch",
            qualified=True,
            error_type=RunActionCodingAgentContractError,
        )
        if not isinstance(self.git_commit_author_name, str) or (
            not self.git_commit_author_name.strip()
            or any(character in self.git_commit_author_name for character in "\r\n<>")
        ):
            raise RunActionCodingAgentContractError(
                "coding-agent Git commit author name is invalid"
            )
        if (
            not isinstance(self.git_commit_author_email, str)
            or re.fullmatch(
                r"[^<>\s@]+@[^<>\s@]+",
                self.git_commit_author_email,
            )
            is None
        ):
            raise RunActionCodingAgentContractError(
                "coding-agent Git commit author email is invalid"
            )
        for value, name in (
            (
                self.maximum_request_bytes,
                "coding-agent maximum request bytes",
            ),
            (
                self.maximum_response_schema_bytes,
                "coding-agent maximum response-schema bytes",
            ),
            (
                self.maximum_cli_argument_bytes,
                "coding-agent maximum CLI argument bytes",
            ),
            (
                self.maximum_provider_output_bytes,
                "coding-agent maximum provider-output bytes",
            ),
            (
                self.maximum_provider_diagnostic_bytes,
                "coding-agent maximum provider-diagnostic bytes",
            ),
            (
                self.maximum_prior_knowledge_audit_bytes,
                "coding-agent maximum prior-knowledge audit bytes",
            ),
            (
                self.maximum_native_credential_bytes,
                "coding-agent maximum native-credential bytes",
            ),
            (
                self.maximum_workspace_entries,
                "coding-agent maximum workspace entries",
            ),
            (
                self.maximum_workspace_bytes,
                "coding-agent maximum workspace bytes",
            ),
            (
                self.maximum_workspace_git_entries,
                "coding-agent maximum workspace Git entries",
            ),
            (
                self.maximum_workspace_git_bytes,
                "coding-agent maximum workspace Git bytes",
            ),
            (
                self.maximum_raw_result_bytes,
                "coding-agent maximum raw-result bytes",
            ),
        ):
            _require_positive_integer(value, name)
        if self.maximum_response_schema_bytes >= self.maximum_cli_argument_bytes:
            raise RunActionCodingAgentContractError(
                "coding-agent response-schema bound must fit inside its CLI "
                "argument bound"
            )


@dataclass(frozen=True)
class CodingAgentRunActionRequest(StrictContract):
    """Complete path-free input to one coding-agent run action."""

    protocol_version: str
    interpretation_policy: CodingAgentInterpretationPolicy
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
        if type(self.interpretation_policy) is not CodingAgentInterpretationPolicy:
            raise RunActionCodingAgentContractError(
                "coding-agent request lacks its complete interpretation policy"
            )
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
        validate_run_action_coding_agent_provider_schema(self.response_schema)
        if (
            len(canonical_json_bytes(self.response_schema))
            > self.interpretation_policy.maximum_response_schema_bytes
        ):
            raise RunActionCodingAgentContractError(
                "coding-agent response schema exceeds its exact byte limit"
            )
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
        if len(self.to_json_bytes()) > self.interpretation_policy.maximum_request_bytes:
            raise RunActionCodingAgentContractError(
                "coding-agent request exceeds its exact byte limit"
            )

    @property
    def request_digest(self) -> str:
        return tree_or_blob_digest(self.to_json_bytes())

    def require_policy(self, policy: CodingAgentInterpretationPolicy) -> None:
        if type(policy) is not CodingAgentInterpretationPolicy:
            raise RunActionCodingAgentContractError(
                "coding-agent request requires an exact interpretation policy"
            )
        if self.interpretation_policy != policy:
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
    cached_input_tokens: int | None
    output_tokens: int
    reasoning_output_tokens: int | None
    cost_usd: str | None
    provider_event_stream_digest: str
    provider_diagnostic_stream_digest: str
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
        for value, name in (
            (self.cached_input_tokens, "coding-agent cached input tokens"),
            (self.reasoning_output_tokens, "coding-agent reasoning output tokens"),
        ):
            if value is not None:
                _require_nonnegative_integer(value, name)
        if (
            self.cached_input_tokens is not None
            and self.cached_input_tokens > self.input_tokens
        ):
            raise RunActionCodingAgentContractError(
                "coding-agent cached input tokens exceed total input tokens"
            )
        if (
            self.reasoning_output_tokens is not None
            and self.reasoning_output_tokens > self.output_tokens
        ):
            raise RunActionCodingAgentContractError(
                "coding-agent reasoning tokens exceed total output tokens"
            )
        if self.cost_usd is not None and (
            not isinstance(self.cost_usd, str)
            or _NORMALIZED_DECIMAL_PATTERN.fullmatch(self.cost_usd) is None
        ):
            raise RunActionCodingAgentContractError(
                "coding-agent cost must be normalized non-negative decimal text"
            )
        _require_digest(
            self.provider_event_stream_digest,
            "coding-agent provider event stream",
        )
        _require_digest(
            self.provider_diagnostic_stream_digest,
            "coding-agent provider diagnostic stream",
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


def read_canonical_coding_agent_request(
    payload: bytes,
) -> CodingAgentRunActionRequest:
    """Parse one complete request only when its original bytes are canonical."""

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


def read_canonical_coding_agent_result(
    payload: bytes,
) -> CodingAgentRunActionResultEnvelope:
    """Parse one complete result only when its original bytes are canonical."""

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
    "CODING_AGENT_REQUEST_PROTOCOL_VERSION",
    "CODING_AGENT_RESULT_PROTOCOL_VERSION",
    "CODING_AGENT_SCHEMA_PROTOCOL_VERSION",
    "CODING_AGENT_NATIVE_TOOL_POLICY_VERSION",
    "CodingAgentInterpretationPolicy",
    "CodingAgentPriorKnowledgeAccessEvent",
    "CodingAgentPriorKnowledgeAccessKind",
    "CodingAgentRunActionRequest",
    "CodingAgentRunActionResultEnvelope",
    "RunActionCodingAgentContractError",
    "read_canonical_coding_agent_request",
    "read_canonical_coding_agent_result",
]
