from dataclasses import fields, replace

import pytest

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import PriorKnowledgeSnapshot
from kapso.cross_run.knowledge.access import (
    PriorKnowledgeAccess,
    PriorKnowledgeAccessMaterialization,
)
from kapso.cross_run.launch.run_action_coding_agent_contracts import (
    CODING_AGENT_REQUEST_PROTOCOL_VERSION,
    CODING_AGENT_RESULT_PROTOCOL_VERSION,
    CODING_AGENT_SCHEMA_PROTOCOL_VERSION,
    CODING_AGENT_NATIVE_TOOL_POLICY_VERSION,
    CodingAgentInterpretationPolicy,
    CodingAgentPriorKnowledgeAccessEvent,
    CodingAgentPriorKnowledgeAccessKind,
    CodingAgentRunActionRequest,
    CodingAgentRunActionResultEnvelope,
    RunActionCodingAgentContractError,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierWorkspaceAccess,
)

_OPERATION_ID = "agent_call_" + "1" * 32
_PREDECESSOR_DIGEST = tree_or_blob_digest(b"predecessor")
_EDITED_DIGEST = tree_or_blob_digest(b"edited")


def interpretation_policy(
    *,
    cli="codex",
    workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
    web_search_enabled=True,
    maximum_response_schema_bytes=65_536,
    maximum_raw_result_bytes=65_536,
):
    return CodingAgentInterpretationPolicy.mint(
        request_protocol_version=CODING_AGENT_REQUEST_PROTOCOL_VERSION,
        result_protocol_version=CODING_AGENT_RESULT_PROTOCOL_VERSION,
        schema_protocol_version=CODING_AGENT_SCHEMA_PROTOCOL_VERSION,
        consumer_id="kapso.coding_agent_consumer",
        consumer_version="v1",
        principal_id="kapso.ideation.generator",
        role="candidate_generator",
        cli=cli,
        model="gpt-5.6",
        effort="xhigh",
        native_tool_policy_version=CODING_AGENT_NATIVE_TOOL_POLICY_VERSION,
        web_search_enabled=web_search_enabled,
        timeout_nanoseconds=300_000_000_000,
        workspace_access=workspace_access,
        maximum_response_schema_bytes=maximum_response_schema_bytes,
        maximum_provider_output_bytes=1_048_576,
        maximum_provider_diagnostic_bytes=65_536,
        maximum_workspace_entries=10_000,
        maximum_workspace_bytes=1_073_741_824,
        maximum_raw_result_bytes=maximum_raw_result_bytes,
    )


def run_action_request(
    policy,
    *,
    response_schema=None,
    prior_knowledge=None,
    predecessor_digest=None,
):
    return CodingAgentRunActionRequest(
        protocol_version=CODING_AGENT_REQUEST_PROTOCOL_VERSION,
        interpretation_policy=policy,
        operation_id=_OPERATION_ID,
        prompt="Return the complete structured proposal.",
        response_schema=(
            {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            }
            if response_schema is None
            else response_schema
        ),
        prior_knowledge=prior_knowledge,
        edit_predecessor_source_tree_digest=predecessor_digest,
    )


def result_envelope(
    request,
    *,
    structured_output=None,
    cost_usd="0.00125",
    accesses=(),
    edited_digest=None,
):
    return CodingAgentRunActionResultEnvelope(
        protocol_version=CODING_AGENT_RESULT_PROTOCOL_VERSION,
        consumer_id="kapso.coding_agent_consumer",
        consumer_version="v1",
        operation_id=request.operation_id,
        request_digest=request.request_digest,
        structured_output=(
            {"answer": "Use the evidence."}
            if structured_output is None
            else structured_output
        ),
        duration_nanoseconds=12_345,
        input_tokens=101,
        cached_input_tokens=80,
        output_tokens=23,
        reasoning_output_tokens=7,
        cost_usd=cost_usd,
        provider_event_stream_digest=tree_or_blob_digest(b"provider events"),
        provider_diagnostic_stream_digest=tree_or_blob_digest(b"provider diagnostics"),
        prior_knowledge_accesses=accesses,
        edited_source_tree_digest=edited_digest,
    )


def empty_prior_knowledge():
    selected_records = ()
    packet = PriorKnowledgeSnapshot.mint(
        source_snapshot_id=content_id(
            "knowledge-snapshot",
            {"fixture": "coding-agent-contracts"},
        ),
        query={"problem": "Improve the system."},
        retrieval_policy_version="kapso.retrieval.v1",
        task_context_binding_id=content_id(
            "task-context-binding",
            {"fixture": "coding-agent-contracts"},
        ),
        selected_records=selected_records,
        selected_record_ids=(),
        proof_reference_ids=(),
        selection_metadata={},
        prompt_budget_policy={"maximum_records": 0},
        records_digest=tree_or_blob_digest(canonical_json_bytes(selected_records)),
    )
    return PriorKnowledgeAccessMaterialization.mint(
        prior_knowledge_snapshot=packet,
        proof_records=(),
    )


def test_policy_request_and_result_are_content_stable_canonical_contracts():
    policy = interpretation_policy()
    request = run_action_request(policy)
    result = result_envelope(request)

    request.require_policy(policy)
    result.validate_against(policy=policy, request=request)

    assert (
        CodingAgentInterpretationPolicy.from_json_bytes(policy.to_json_bytes())
        == policy
    )
    assert (
        CodingAgentRunActionRequest.from_json_bytes(request.to_json_bytes()) == request
    )
    assert (
        CodingAgentRunActionResultEnvelope.from_json_bytes(result.to_json_bytes())
        == result
    )
    assert request.to_json_bytes() == canonical_json_bytes(request.to_dict())
    assert result.to_json_bytes() == canonical_json_bytes(result.to_dict())
    assert request.to_dict()["interpretation_policy"] == policy.to_dict()
    assert "interpretation_policy_id" not in request.to_dict()

    with pytest.raises(ValueError, match="interpretation_policy_id mismatch"):
        replace(policy, role="candidate_reviewer")


def test_nested_request_and_result_inputs_are_recursively_frozen():
    schema = {
        "type": "object",
        "required": ["answer"],
        "properties": {"answer": {"type": "string", "enum": ["first"]}},
        "additionalProperties": False,
    }
    output = {"answer": "first", "evidence": ["record-a"]}
    policy = interpretation_policy()
    request = run_action_request(policy, response_schema=schema)
    result = result_envelope(request, structured_output=output)

    schema["required"].append("mutated")
    schema["properties"]["answer"]["enum"].append("mutated")
    output["answer"] = "mutated"
    output["evidence"].append("mutated")

    assert request.response_schema["required"] == ("answer",)
    assert request.response_schema["properties"]["answer"]["enum"] == ("first",)
    assert result.structured_output == {
        "answer": "first",
        "evidence": ("record-a",),
    }


@pytest.mark.parametrize(
    "cost",
    (
        "",
        "-1",
        "+1",
        "00",
        "01",
        ".1",
        "1.",
        "1.0",
        "1.230",
        "1e-3",
    ),
)
def test_result_rejects_non_normalized_cost(cost):
    policy = interpretation_policy()
    request = run_action_request(policy)

    with pytest.raises(
        RunActionCodingAgentContractError,
        match="normalized non-negative decimal",
    ):
        result_envelope(request, cost_usd=cost)


@pytest.mark.parametrize("cost", (1, 0.0))
def test_result_rejects_numeric_cost(cost):
    policy = interpretation_policy()
    request = run_action_request(policy)

    with pytest.raises(ValueError, match="cost_usd must be a string"):
        result_envelope(request, cost_usd=cost)


@pytest.mark.parametrize("cost", (None, "0", "1", "0.000001", "12.3405"))
def test_result_accepts_normalized_cost(cost):
    policy = interpretation_policy()
    request = run_action_request(policy)
    result = result_envelope(request, cost_usd=cost)

    result.validate_against(policy=policy, request=request)


@pytest.mark.parametrize(
    "field_name",
    ("cached_input_tokens", "reasoning_output_tokens"),
)
def test_result_rejects_invalid_optional_usage(field_name):
    policy = interpretation_policy()
    request = run_action_request(policy)

    with pytest.raises(ValueError, match=f"{field_name} must be an integer"):
        replace(result_envelope(request), **{field_name: 1.0})

    with pytest.raises(
        RunActionCodingAgentContractError,
        match="non-negative integer",
    ):
        replace(result_envelope(request), **{field_name: -1})


@pytest.mark.parametrize(
    "field_name",
    ("provider_event_stream_digest", "provider_diagnostic_stream_digest"),
)
def test_result_requires_provider_stream_digests(field_name):
    policy = interpretation_policy()
    request = run_action_request(policy)

    with pytest.raises(
        RunActionCodingAgentContractError,
        match="sha256 digest",
    ):
        replace(result_envelope(request), **{field_name: "unbound"})


def test_policy_rejects_unknown_native_tool_policy_and_non_boolean_web_search():
    with pytest.raises(
        RunActionCodingAgentContractError,
        match="native-tool policy",
    ):
        replace(
            interpretation_policy(),
            native_tool_policy_version="kapso.unknown",
        )

    with pytest.raises(
        ValueError,
        match="web_search_enabled must be a boolean",
    ):
        replace(interpretation_policy(), web_search_enabled=1)


def test_request_rejects_response_schema_above_the_policy_bound():
    policy = interpretation_policy(
        maximum_response_schema_bytes=1,
    )

    with pytest.raises(
        RunActionCodingAgentContractError,
        match="response schema exceeds its exact byte limit",
    ):
        run_action_request(policy)


@pytest.mark.parametrize(
    ("cli", "effort"),
    (
        ("codex", "max"),
        ("claude_code", "minimal"),
    ),
)
def test_policy_rejects_effort_incompatible_with_cli(cli, effort):
    policy = interpretation_policy()

    with pytest.raises(
        RunActionCodingAgentContractError,
        match="effort is incompatible",
    ):
        replace(policy, cli=cli, effort=effort)


def test_request_and_result_require_exact_digest_operation_and_consumer_joins():
    policy = interpretation_policy()
    request = run_action_request(policy)
    result = result_envelope(request)

    result.validate_against(policy=policy, request=request)

    another_policy = interpretation_policy(maximum_raw_result_bytes=65_537)
    with pytest.raises(
        RunActionCodingAgentContractError,
        match="another interpretation policy",
    ):
        replace(
            request,
            interpretation_policy=another_policy,
        ).require_policy(policy)

    with pytest.raises(
        RunActionCodingAgentContractError,
        match="another operation",
    ):
        replace(
            result,
            operation_id="agent_call_" + "2" * 32,
        ).validate_against(policy=policy, request=request)

    with pytest.raises(
        RunActionCodingAgentContractError,
        match="another request",
    ):
        replace(
            result,
            request_digest=tree_or_blob_digest(b"another request"),
        ).validate_against(policy=policy, request=request)

    with pytest.raises(
        RunActionCodingAgentContractError,
        match="another consumer",
    ):
        replace(
            result,
            consumer_version="v2",
        ).validate_against(policy=policy, request=request)


def test_edit_tree_digests_are_required_joined_and_distinct():
    policy = interpretation_policy(
        cli="claude_code",
        workspace_access=RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
    )
    request = run_action_request(
        policy,
        predecessor_digest=_PREDECESSOR_DIGEST,
    )
    result = result_envelope(request, edited_digest=_EDITED_DIGEST)

    request.require_policy(policy)
    result.validate_against(policy=policy, request=request)

    with pytest.raises(
        RunActionCodingAgentContractError,
        match="did not change",
    ):
        replace(
            result,
            edited_source_tree_digest=_PREDECESSOR_DIGEST,
        ).validate_against(policy=policy, request=request)

    with pytest.raises(
        RunActionCodingAgentContractError,
        match="edit predecessor",
    ):
        run_action_request(policy).require_policy(policy)


def test_prior_knowledge_access_events_are_semantically_joined():
    materialization = empty_prior_knowledge()
    access = PriorKnowledgeAccess(materialization)
    list_event = CodingAgentPriorKnowledgeAccessEvent(
        access_kind=CodingAgentPriorKnowledgeAccessKind.LIST,
        record_id=None,
        returned_record_ids=(),
        response_digest=tree_or_blob_digest(
            canonical_json_bytes(access.list_response_payload())
        ),
    )
    policy = interpretation_policy()
    request = run_action_request(
        policy,
        prior_knowledge=materialization,
    )
    result = result_envelope(request, accesses=(list_event, list_event))

    result.validate_against(policy=policy, request=request)
    assert result.prior_knowledge_accesses == (list_event, list_event)

    without_prior = run_action_request(policy)
    with pytest.raises(
        RunActionCodingAgentContractError,
        match="undeclared prior-knowledge",
    ):
        result_envelope(
            without_prior,
            accesses=(list_event,),
        ).validate_against(policy=policy, request=without_prior)


def test_contracts_reject_malformed_fields_numbers_and_result_size():
    policy = interpretation_policy()
    request = run_action_request(policy)
    result = result_envelope(request)
    malformed = request.to_dict()
    malformed["workspace_path"] = "/tmp/workspace"

    with pytest.raises(ValueError, match="fields mismatch"):
        CodingAgentRunActionRequest.from_dict(malformed)

    with pytest.raises(ValueError, match="duration_nanoseconds must be an integer"):
        replace(result, duration_nanoseconds=1.0)

    small_policy = interpretation_policy(
        maximum_raw_result_bytes=len(result.to_json_bytes()) - 1,
    )
    small_request = run_action_request(small_policy)
    small_result = result_envelope(small_request)
    with pytest.raises(
        RunActionCodingAgentContractError,
        match="raw-result byte limit",
    ):
        small_result.validate_against(
            policy=small_policy,
            request=small_request,
        )


def test_contract_surfaces_expose_no_path_or_legacy_artifact_fields():
    contract_types = (
        CodingAgentInterpretationPolicy,
        CodingAgentRunActionRequest,
        CodingAgentPriorKnowledgeAccessEvent,
        CodingAgentRunActionResultEnvelope,
    )
    forbidden_fragments = (
        "artifact",
        "base64",
        "path",
        "secret",
        "stderr",
        "stdout",
    )

    for contract_type in contract_types:
        field_names = tuple(field.name for field in fields(contract_type))
        assert all(
            fragment not in field_name
            for field_name in field_names
            for fragment in forbidden_fragments
        )
