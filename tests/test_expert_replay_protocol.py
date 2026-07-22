from __future__ import annotations

import math
from dataclasses import replace

import pytest

from kapso.cross_run.canonical import canonical_json_bytes, content_id
from kapso.cross_run.contracts import EvaluationFingerprint
from kapso.cross_run.expert.replay_protocol import build_task_evaluator_request
from kapso.cross_run.expert.replay_protocol_contracts import (
    TASK_EVALUATOR_ADAPTER_ROOT,
    TASK_EVALUATOR_EXPERT_ROOT,
    TASK_EVALUATOR_REQUEST_PATH,
    TASK_EVALUATOR_RESULT_PATH,
    TASK_EVALUATOR_TASK_ROOT,
    TASK_EVALUATOR_WRITABLE_ROOT,
    ExpertSourceReplayProtocolError,
    TaskEvaluatorFingerprintResult,
    TaskEvaluatorInvocationAllocation,
    TaskEvaluatorRequest,
    TaskEvaluatorResult,
    TaskEvaluatorStartingArtifactMount,
    parse_task_evaluator_result,
)
from test_expert_source_replay_request import _prepared, _request_fixture

SECOND_OPAQUE_INVOCATION_ID = "replay_invocation_fedcba9876543210fedcba9876543210"
INVOCATION_NONCE = "0123456789abcdef0123456789abcdef"


def _allocation(prepared, execution_leg_id, *, nonce=INVOCATION_NONCE):
    return TaskEvaluatorInvocationAllocation(
        reservation_id=content_id(
            "expert-source-replay-execution-reservation",
            {"fixture": "task-evaluator-protocol"},
        ),
        execution_case_id=prepared.cases[0].request_case.execution_case_id,
        execution_leg_id=execution_leg_id,
        invocation_nonce=nonce,
    )


def _request(tmp_path):
    fixture = _request_fixture(tmp_path)
    prepared = _prepared(fixture)
    request = build_task_evaluator_request(
        prepared.cases[0],
        _allocation(
            prepared,
            prepared.cases[0].request_case.control_leg.execution_leg_id,
        ),
    )
    return fixture, prepared, request


@pytest.fixture(scope="module")
def replay_protocol_fixture(tmp_path_factory):
    return _request(tmp_path_factory.mktemp("expert-replay-protocol"))


def _result_for(request, *, offset=0.0):
    fingerprint_results = []
    for fingerprint_position, fingerprint in enumerate(request.evaluation_fingerprints):
        replicate_values = {
            replicate_id: float(fingerprint_position + replicate_position + 1) + offset
            for replicate_position, replicate_id in enumerate(
                fingerprint.seed_or_replicate_ids
            )
        }
        fingerprint_results.append(
            TaskEvaluatorFingerprintResult(
                evaluation_fingerprint_id=fingerprint.evaluation_fingerprint_id,
                aggregate_value=math.fsum(replicate_values.values())
                / len(replicate_values),
                replicate_values=replicate_values,
            )
        )
    return TaskEvaluatorResult(
        protocol_version=request.protocol_version,
        opaque_invocation_id=request.opaque_invocation_id,
        fingerprint_results=tuple(fingerprint_results),
    )


def test_request_is_a_minimal_blinded_projection_of_the_verified_case(
    replay_protocol_fixture,
):
    _, prepared, request = replay_protocol_fixture
    replay_case = prepared.cases[0]
    context = replay_case.episode.task_context_binding
    consumed_dimension_ids = (
        replay_case.task_adapter.manifest.context_binding.consumed_dimension_ids
    )

    assert set(request.to_dict()) == {
        "protocol_version",
        "opaque_invocation_id",
        "input_contract_fingerprint",
        "target_contract_fingerprint",
        "evaluation_fingerprints",
        "context_dimensions",
        "starting_artifact_mounts",
    }
    assert request.input_contract_fingerprint == context.input_contract_fingerprint
    assert request.target_contract_fingerprint == context.target_contract_fingerprint
    assert request.context_dimensions == {
        dimension_id: context.transfer_dimensions[dimension_id]
        for dimension_id in consumed_dimension_ids
    }
    terminal_attempt = replay_case.episode.attempts[
        replay_case.episode.terminal_attempt_revision
    ]
    assert request.evaluation_fingerprints == terminal_attempt.evaluation_fingerprints
    assert tuple(
        (mount.starting_artifact_ref, mount.mount_path)
        for mount in request.starting_artifact_mounts
    ) == tuple(
        sorted(
            (
                (
                    item.artifact.starting_artifact_ref,
                    item.artifact.mount_path,
                )
                for item in replay_case.task_context.starting_artifacts
            )
        )
    )

    payload = request.to_json_bytes()
    forbidden_values = (
        prepared.request.candidate_id,
        prepared.request.parent_release_id,
        replay_case.request_case.control_leg.execution_leg_id,
        replay_case.request_case.candidate_leg.execution_leg_id,
        replay_case.episode.episode_id,
    )
    assert all(value.encode() not in payload for value in forbidden_values)
    assert all(
        forbidden_key not in request.to_dict()
        for forbidden_key in (
            "candidate_id",
            "parent_release_id",
            "execution_case_id",
            "execution_leg_id",
            "leg_kind",
            "bundle_lineage",
            "source_score",
            "score_of_record_fingerprint_id",
            "compute_binding",
        )
    )
    assert TaskEvaluatorRequest.from_json_bytes(payload) == request


def test_only_opaque_invocation_binding_differs_between_leg_requests(
    replay_protocol_fixture,
):
    _, prepared, control_request = replay_protocol_fixture
    candidate_request = build_task_evaluator_request(
        prepared.cases[0],
        _allocation(
            prepared,
            prepared.cases[0].request_case.candidate_leg.execution_leg_id,
        ),
    )
    control_payload = control_request.to_dict()
    candidate_payload = candidate_request.to_dict()
    control_payload.pop("opaque_invocation_id")
    candidate_payload.pop("opaque_invocation_id")

    assert control_payload == candidate_payload
    assert control_request.to_json_bytes() != candidate_request.to_json_bytes()


def test_protocol_paths_are_fixed_structural_authority():
    assert TASK_EVALUATOR_REQUEST_PATH == "/kapso/input/request.json"
    assert TASK_EVALUATOR_EXPERT_ROOT == "/kapso/input/expert"
    assert TASK_EVALUATOR_ADAPTER_ROOT == "/kapso/input/adapter"
    assert TASK_EVALUATOR_TASK_ROOT == "/kapso/input/task"
    assert TASK_EVALUATOR_WRITABLE_ROOT == "/kapso/writable"
    assert TASK_EVALUATOR_RESULT_PATH == "/kapso/writable/result.json"


def test_result_parser_accepts_only_the_exact_canonical_measurement_matrix(
    replay_protocol_fixture,
):
    fixture, _, request = replay_protocol_fixture
    result = _result_for(request)

    parsed = parse_task_evaluator_result(
        result.to_json_bytes(),
        request,
        fixture.settings.policy.source_replay_score_comparison_tolerance,
    )

    assert parsed == result


@pytest.mark.parametrize(
    "mutate",
    (
        lambda result, request: replace(
            result,
            opaque_invocation_id=SECOND_OPAQUE_INVOCATION_ID,
        ),
        lambda result, request: replace(result, fingerprint_results=()),
        lambda result, request: replace(
            result,
            fingerprint_results=(
                replace(
                    result.fingerprint_results[0],
                    evaluation_fingerprint_id=(
                        "evaluation-fingerprint:sha256:" + "f" * 64
                    ),
                ),
            ),
        ),
        lambda result, request: replace(
            result,
            fingerprint_results=(
                replace(
                    result.fingerprint_results[0],
                    replicate_values={"another-replicate": 1.0},
                ),
            ),
        ),
        lambda result, request: replace(
            result,
            fingerprint_results=(
                replace(
                    result.fingerprint_results[0],
                    aggregate_value=result.fingerprint_results[0].aggregate_value + 1.0,
                ),
            ),
        ),
    ),
)
def test_result_rejects_identity_coverage_and_aggregate_substitution(
    replay_protocol_fixture,
    mutate,
):
    fixture, _, request = replay_protocol_fixture

    with pytest.raises(ValueError):
        result = mutate(_result_for(request), request)
        parse_task_evaluator_result(
            result.to_json_bytes(),
            request,
            fixture.settings.policy.source_replay_score_comparison_tolerance,
        )


def test_result_aggregate_uses_the_pinned_tolerance(replay_protocol_fixture):
    _, _, request = replay_protocol_fixture
    result = _result_for(request)
    measurement = result.fingerprint_results[0]
    drifted = replace(
        result,
        fingerprint_results=(
            replace(
                measurement,
                aggregate_value=measurement.aggregate_value + 0.001,
            ),
        ),
    )

    assert (
        parse_task_evaluator_result(
            drifted.to_json_bytes(),
            request,
            0.001,
        )
        == drifted
    )
    with pytest.raises(ExpertSourceReplayProtocolError, match="aggregate"):
        parse_task_evaluator_result(
            drifted.to_json_bytes(),
            request,
            0.0009,
        )


@pytest.mark.parametrize(
    "invalid_value",
    (True, 1, float("nan"), float("inf"), "1.0", None, (), {}),
)
def test_result_rejects_non_float_or_non_finite_measurements(
    replay_protocol_fixture,
    invalid_value,
):
    _, _, request = replay_protocol_fixture
    fingerprint = request.evaluation_fingerprints[0]

    with pytest.raises(ValueError):
        TaskEvaluatorFingerprintResult(
            evaluation_fingerprint_id=fingerprint.evaluation_fingerprint_id,
            aggregate_value=invalid_value,
            replicate_values={
                fingerprint.seed_or_replicate_ids[0]: 1.0,
            },
        )
    with pytest.raises(ValueError):
        TaskEvaluatorFingerprintResult(
            evaluation_fingerprint_id=fingerprint.evaluation_fingerprint_id,
            aggregate_value=1.0,
            replicate_values={
                fingerprint.seed_or_replicate_ids[0]: invalid_value,
            },
        )


def test_result_parser_rejects_noncanonical_and_unknown_json(
    replay_protocol_fixture,
):
    fixture, _, request = replay_protocol_fixture
    result = _result_for(request)
    tolerance = fixture.settings.policy.source_replay_score_comparison_tolerance

    with pytest.raises(ExpertSourceReplayProtocolError, match="canonical"):
        parse_task_evaluator_result(
            result.to_json_bytes() + b"\n",
            request,
            tolerance,
        )
    payload_with_unknown_field = result.to_dict()
    payload_with_unknown_field["outcome"] = "passed"
    with pytest.raises(ValueError):
        parse_task_evaluator_result(
            canonical_json_bytes(payload_with_unknown_field),
            request,
            tolerance,
        )
    duplicate_key_payload = result.to_json_bytes().replace(
        b'{"fingerprint_results"',
        b'{"protocol_version":"kapso.task_evaluator.v1","fingerprint_results"',
        1,
    )
    with pytest.raises(ValueError, match="duplicate"):
        parse_task_evaluator_result(
            duplicate_key_payload,
            request,
            tolerance,
        )


def test_request_rejects_unknown_aggregation_before_execution(
    replay_protocol_fixture,
):
    _, _, request = replay_protocol_fixture
    fingerprint = request.evaluation_fingerprints[0]
    fingerprint_values = fingerprint.to_dict()
    fingerprint_values.pop("evaluation_fingerprint_id")
    fingerprint_values["aggregation_protocol"] = "task-specific-unknown"
    unsupported = EvaluationFingerprint.mint(**fingerprint_values)

    with pytest.raises(ExpertSourceReplayProtocolError, match="unsupported"):
        replace(request, evaluation_fingerprints=(unsupported,))


def test_result_rejects_a_malformed_fingerprint_content_id():
    with pytest.raises(ValueError):
        TaskEvaluatorFingerprintResult(
            evaluation_fingerprint_id="evaluation-fingerprint:sha256:",
            aggregate_value=1.0,
            replicate_values={"seed-1": 1.0},
        )


def test_result_recomputes_the_mean_without_finite_overflow(
    replay_protocol_fixture,
):
    _, _, request = replay_protocol_fixture
    fingerprint_values = request.evaluation_fingerprints[0].to_dict()
    fingerprint_values.pop("evaluation_fingerprint_id")
    fingerprint_values["seed_or_replicate_ids"] = ("seed-1", "seed-2")
    fingerprint = EvaluationFingerprint.mint(**fingerprint_values)
    two_replicate_request = replace(
        request,
        evaluation_fingerprints=(fingerprint,),
    )
    maximum_float = float.fromhex("0x1.fffffffffffffp+1023")
    result = TaskEvaluatorResult(
        protocol_version=two_replicate_request.protocol_version,
        opaque_invocation_id=two_replicate_request.opaque_invocation_id,
        fingerprint_results=(
            TaskEvaluatorFingerprintResult(
                evaluation_fingerprint_id=fingerprint.evaluation_fingerprint_id,
                aggregate_value=maximum_float,
                replicate_values={
                    "seed-1": maximum_float,
                    "seed-2": maximum_float,
                },
            ),
        ),
    )

    assert (
        parse_task_evaluator_result(
            result.to_json_bytes(),
            two_replicate_request,
            0.0,
        )
        == result
    )


@pytest.mark.parametrize("mount_path", ("a\x00b", "a\nb", "a\rb", "a\x7fb"))
def test_starting_artifact_mount_rejects_control_characters(mount_path):
    with pytest.raises(ExpertSourceReplayProtocolError):
        TaskEvaluatorStartingArtifactMount(
            starting_artifact_ref="artifact/base",
            mount_path=mount_path,
        )


@pytest.mark.parametrize(
    "invalid_nonce",
    (
        "control",
        "0123",
        "0123456789ABCDEF0123456789ABCDEF",
        "0123456789abcdef0123456789abcdeg",
    ),
)
def test_allocation_rejects_invalid_invocation_nonces(
    replay_protocol_fixture,
    invalid_nonce,
):
    _, prepared, _ = replay_protocol_fixture

    with pytest.raises(ExpertSourceReplayProtocolError, match="random"):
        _allocation(
            prepared,
            prepared.cases[0].request_case.control_leg.execution_leg_id,
            nonce=invalid_nonce,
        )
