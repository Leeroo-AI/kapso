import json
from dataclasses import replace

import pytest

from kapso.cross_run.canonical import canonical_json_bytes, tree_or_blob_digest
from kapso.cross_run.knowledge.access import PriorKnowledgeAccess
from kapso.cross_run.launch.run_action_coding_agent_contracts import (
    CodingAgentPriorKnowledgeAccessEvent,
    CodingAgentPriorKnowledgeAccessKind,
    RunActionCodingAgentContractError,
)
from kapso.cross_run.launch.run_action_coding_agent_interpreter import (
    CODING_AGENT_RESULT_INTERPRETATION_PROTOCOL_VERSION,
    CODING_AGENT_RESULT_INTERPRETER_IMPLEMENTATION_ID,
    CODING_AGENT_RESULT_INTERPRETER_IMPLEMENTATION_VERSION,
    CodingAgentRunActionResultInterpreter,
    FixedOfflineCodingAgentConsumer,
    coding_agent_result_interpreter_identity,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunActionResultInterpreterIdentity,
    RunFrontierActionKind,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_store import RunActionResultDisposition
from test_run_action_coding_agent_contracts import (
    _EDITED_DIGEST,
    _PREDECESSOR_DIGEST,
    empty_prior_knowledge,
    interpretation_policy,
    result_envelope,
    run_action_request,
)


def _interpreter(policy):
    return CodingAgentRunActionResultInterpreter(
        result_interpreter_identity=coding_agent_result_interpreter_identity(policy),
        interpretation_policy=policy,
    )


def _consumer(
    policy,
    *,
    structured_output=None,
    accesses=(),
    edited_digest=None,
):
    return FixedOfflineCodingAgentConsumer(
        interpretation_policy=policy,
        structured_output=(
            {"answer": "Use the exact evidence."}
            if structured_output is None
            else structured_output
        ),
        duration_nanoseconds=12_345,
        input_tokens=101,
        output_tokens=23,
        cost_usd="0.00125",
        prior_knowledge_accesses=accesses,
        edited_source_tree_digest=edited_digest,
    )


def test_identity_is_exactly_content_bound_to_the_interpretation_policy():
    policy = interpretation_policy()
    identity = coding_agent_result_interpreter_identity(policy)

    assert identity.kind is RunFrontierActionKind.CODING_AGENT
    assert (
        identity.implementation_id == CODING_AGENT_RESULT_INTERPRETER_IMPLEMENTATION_ID
    )
    assert (
        identity.implementation_version
        == CODING_AGENT_RESULT_INTERPRETER_IMPLEMENTATION_VERSION
    )
    assert (
        identity.interpretation_protocol_version
        == CODING_AGENT_RESULT_INTERPRETATION_PROTOCOL_VERSION
    )
    assert identity.interpretation_policy_id == policy.interpretation_policy_id

    substituted = RunActionResultInterpreterIdentity.mint(
        kind=RunFrontierActionKind.CODING_AGENT,
        implementation_id=identity.implementation_id,
        implementation_version=identity.implementation_version,
        interpretation_protocol_version=identity.interpretation_protocol_version,
        interpretation_policy_id=interpretation_policy(
            maximum_raw_result_bytes=65_535
        ).interpretation_policy_id,
    )
    with pytest.raises(
        RunActionCodingAgentContractError,
        match="exact policy identity",
    ):
        CodingAgentRunActionResultInterpreter(
            result_interpreter_identity=substituted,
            interpretation_policy=policy,
        )


def test_offline_consumer_and_interpreter_accept_only_the_structured_output():
    policy = interpretation_policy()
    request = run_action_request(policy)
    request_payload = request.to_json_bytes()
    result_payload = _consumer(policy).consume(request_payload)
    interpreter = _interpreter(policy)

    interpreted = interpreter.interpret(
        operation_id=request.operation_id,
        request_payload=request_payload,
        result_payload=result_payload,
    )
    repeated = interpreter.interpret(
        operation_id=request.operation_id,
        request_payload=request_payload,
        result_payload=result_payload,
    )

    assert interpreted == repeated
    assert interpreted.disposition is RunActionResultDisposition.SUCCEEDED
    assert interpreted.operation_id == request.operation_id
    assert interpreted.accepted_result_payload == canonical_json_bytes(
        {"answer": "Use the exact evidence."}
    )
    assert interpreted.expected_workspace_before_source_tree_digest is None
    assert interpreted.expected_workspace_after_source_tree_digest is None


@pytest.mark.parametrize("payload_kind", ("request", "result"))
def test_interpreter_rejects_noncanonical_payload_bytes(payload_kind):
    policy = interpretation_policy()
    request = run_action_request(policy)
    result = result_envelope(request)
    request_payload = request.to_json_bytes()
    result_payload = result.to_json_bytes()
    if payload_kind == "request":
        request_payload = json.dumps(
            request.to_dict(),
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
    else:
        result_payload = json.dumps(
            result.to_dict(),
            indent=2,
            sort_keys=True,
        ).encode("utf-8")

    with pytest.raises(
        RunActionCodingAgentContractError,
        match=f"{payload_kind} payload is not canonical",
    ):
        _interpreter(policy).interpret(
            operation_id=request.operation_id,
            request_payload=request_payload,
            result_payload=result_payload,
        )


@pytest.mark.parametrize("payload_kind", ("request", "result"))
def test_interpreter_rejects_duplicate_json_fields(payload_kind):
    policy = interpretation_policy()
    request = run_action_request(policy)
    request_payload = request.to_json_bytes()
    result_payload = _consumer(policy).consume(request_payload)
    duplicate = b'{"protocol_version":"first","protocol_version":"second"}'

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        _interpreter(policy).interpret(
            operation_id=request.operation_id,
            request_payload=duplicate if payload_kind == "request" else request_payload,
            result_payload=duplicate if payload_kind == "result" else result_payload,
        )


def test_interpreter_applies_policy_byte_limit_before_parsing_result():
    policy = interpretation_policy(maximum_raw_result_bytes=1)
    request = run_action_request(policy)

    with pytest.raises(
        RunActionCodingAgentContractError,
        match="raw result exceeds its exact byte limit",
    ):
        _interpreter(policy).interpret(
            operation_id=request.operation_id,
            request_payload=request.to_json_bytes(),
            result_payload=b"{}",
        )


def test_interpreter_validates_the_complete_output_against_the_closed_schema():
    policy = interpretation_policy()
    request = run_action_request(
        policy,
        response_schema={
            "type": "object",
            "properties": {"answer": {"type": "string", "minLength": 1}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )
    invalid_result = result_envelope(
        request,
        structured_output={"answer": ""},
    )

    with pytest.raises(ValueError, match="fewer than 1 characters"):
        _interpreter(policy).interpret(
            operation_id=request.operation_id,
            request_payload=request.to_json_bytes(),
            result_payload=invalid_result.to_json_bytes(),
        )


def test_request_rejects_unsupported_schema_before_execution():
    policy = interpretation_policy()

    with pytest.raises(ValueError, match="unsupported keywords"):
        run_action_request(
            policy,
            response_schema={
                "type": "object",
                "$ref": "https://example.com/remote.json",
            },
        )


def test_prior_knowledge_access_is_semantic_and_path_free():
    materialization = empty_prior_knowledge()
    access = PriorKnowledgeAccess(materialization)
    event = CodingAgentPriorKnowledgeAccessEvent(
        access_kind=CodingAgentPriorKnowledgeAccessKind.LIST,
        record_id=None,
        returned_record_ids=(),
        response_digest=tree_or_blob_digest(
            canonical_json_bytes(access.list_response_payload())
        ),
    )
    policy = interpretation_policy()
    request = run_action_request(policy, prior_knowledge=materialization)
    result_payload = _consumer(policy, accesses=(event, event)).consume(
        request.to_json_bytes()
    )

    interpreted = _interpreter(policy).interpret(
        operation_id=request.operation_id,
        request_payload=request.to_json_bytes(),
        result_payload=result_payload,
    )

    assert interpreted.accepted_result_payload == canonical_json_bytes(
        {"answer": "Use the exact evidence."}
    )
    serialized = json.loads(result_payload)
    assert set(serialized) == {
        "consumer_id",
        "consumer_version",
        "cost_usd",
        "duration_nanoseconds",
        "edited_source_tree_digest",
        "input_tokens",
        "operation_id",
        "output_tokens",
        "prior_knowledge_accesses",
        "protocol_version",
        "request_digest",
        "structured_output",
    }
    assert not any(
        fragment in key
        for key in serialized
        for fragment in ("artifact", "base64", "path", "stderr", "stdout")
    )


def test_edit_result_returns_the_exact_predecessor_and_successor_join():
    policy = interpretation_policy(
        cli="claude_code",
        workspace_access=RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
        allowed_tools=("Edit", "Read", "Write"),
    )
    request = run_action_request(
        policy,
        predecessor_digest=_PREDECESSOR_DIGEST,
    )
    result_payload = _consumer(
        policy,
        edited_digest=_EDITED_DIGEST,
    ).consume(request.to_json_bytes())

    interpreted = _interpreter(policy).interpret(
        operation_id=request.operation_id,
        request_payload=request.to_json_bytes(),
        result_payload=result_payload,
    )

    assert interpreted.expected_workspace_before_source_tree_digest == (
        _PREDECESSOR_DIGEST
    )
    assert interpreted.expected_workspace_after_source_tree_digest == _EDITED_DIGEST


def test_fixed_consumer_freezes_output_and_rejects_another_policy_request():
    output = {"answer": "original", "evidence": ["record"]}
    policy = interpretation_policy()
    consumer = _consumer(policy, structured_output=output)
    response_schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "evidence": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["answer", "evidence"],
        "additionalProperties": False,
    }
    output["answer"] = "mutated"
    output["evidence"].append("mutated")

    result_payload = consumer.consume(
        run_action_request(policy, response_schema=response_schema).to_json_bytes()
    )
    assert json.loads(result_payload)["structured_output"] == {
        "answer": "original",
        "evidence": ["record"],
    }

    other_policy = interpretation_policy(maximum_raw_result_bytes=65_535)
    with pytest.raises(
        RunActionCodingAgentContractError,
        match="another interpretation policy",
    ):
        consumer.consume(
            run_action_request(
                other_policy,
                response_schema=response_schema,
            ).to_json_bytes()
        )


def test_interpreter_rejects_request_result_and_policy_substitution():
    policy = interpretation_policy()
    request = run_action_request(policy)
    result = result_envelope(request)
    interpreter = _interpreter(policy)

    with pytest.raises(
        RunActionCodingAgentContractError,
        match="another request",
    ):
        interpreter.interpret(
            operation_id=request.operation_id,
            request_payload=request.to_json_bytes(),
            result_payload=replace(
                result,
                request_digest=tree_or_blob_digest(b"foreign request"),
            ).to_json_bytes(),
        )

    with pytest.raises(
        RunActionCodingAgentContractError,
        match="another durable operation",
    ):
        interpreter.interpret(
            operation_id="agent_call_" + "f" * 32,
            request_payload=request.to_json_bytes(),
            result_payload=result.to_json_bytes(),
        )

    with pytest.raises(
        RunActionCodingAgentContractError,
        match="another consumer",
    ):
        interpreter.interpret(
            operation_id=request.operation_id,
            request_payload=request.to_json_bytes(),
            result_payload=replace(
                result,
                consumer_id="kapso.foreign_consumer",
            ).to_json_bytes(),
        )
