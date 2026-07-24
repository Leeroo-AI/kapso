import json

import pytest

from kapso.cross_run.canonical import canonical_json_bytes
from kapso.cross_run.capture.revision_projection import (
    ExecutionRevisionProjection,
    RevisionProjectionConflictError,
    RevisionProjectionError,
)
from kapso.cross_run.contracts import (
    EpisodeEvaluationStatus,
    ExecutionStatus,
)
from kapso.cross_run.record_contracts import (
    EXECUTION_REVISION_EVENT_SCHEMA,
    ExecutionRevisionEvent,
)


def event(
    node_id: int,
    execution_revision: int,
    *,
    run_id: str = "run_test",
    campaign_id: str = "campaign_test",
    feedback: str = "observed",
    started_at: str = "2026-07-24T00:00:00Z",
    recorded_at: str = "2026-07-24T00:01:00Z",
) -> ExecutionRevisionEvent:
    projection = {
        "node_id": node_id,
        "execution_revision": execution_revision,
        "idea_id": f"idea_{node_id}",
        "selection_batch_id": f"batch_{node_id}",
        "parent_node_id": None if node_id == 0 else node_id - 1,
        "timestamp": started_at,
        "feedback": feedback,
        "technical_difficulties": "",
        "had_error": False,
        "evaluation_valid": True,
        "raw_score": 0.5,
        "evaluation_attempts": [{"seed": 1}],
    }
    return ExecutionRevisionEvent.mint(
        schema=EXECUTION_REVISION_EVENT_SCHEMA,
        run_id=run_id,
        campaign_id=campaign_id,
        node_id=node_id,
        execution_revision=execution_revision,
        idea_id=projection["idea_id"],
        selection_batch_id=projection["selection_batch_id"],
        parent_node_id=projection["parent_node_id"],
        started_at=projection["timestamp"],
        recorded_at=recorded_at,
        execution_status=ExecutionStatus.COMPLETED,
        evaluation_status=EpisodeEvaluationStatus.VALID,
        evaluator_fingerprint_ids=("evaluator_test",),
        measurements={"score": 0.5},
        feedback=feedback,
        technical_difficulties="",
        artifact_refs={"branch": f"candidate-{node_id}"},
        projection=projection,
    )


def append_semantics(
    journal: ExecutionRevisionProjection,
    candidate: ExecutionRevisionEvent,
):
    return journal.append_projection(
        node_id=candidate.node_id,
        execution_revision=candidate.execution_revision,
        idea_id=candidate.idea_id,
        selection_batch_id=candidate.selection_batch_id,
        parent_node_id=candidate.parent_node_id,
        started_at=candidate.started_at,
        recorded_at=candidate.recorded_at,
        execution_status=candidate.execution_status,
        evaluation_status=candidate.evaluation_status,
        evaluator_fingerprint_ids=candidate.evaluator_fingerprint_ids,
        measurements=candidate.measurements,
        feedback=candidate.feedback,
        technical_difficulties=candidate.technical_difficulties,
        artifact_refs=candidate.artifact_refs,
        projection=candidate.projection,
    )


def test_empty_and_nonempty_projection_have_exact_canonical_jsonl_bytes():
    empty = ExecutionRevisionProjection("run_test", "campaign_test", True)
    first = event(0, 0)
    projected, appended = empty.append_event(first)

    assert empty.jsonl_bytes == b""
    assert appended is first
    assert projected.jsonl_bytes == canonical_json_bytes(first.to_dict()) + b"\n"
    assert projected.watermark == 1
    assert (
        ExecutionRevisionProjection.from_jsonl_bytes(
            projected.jsonl_bytes,
            run_id="run_test",
            campaign_id="campaign_test",
            require_contiguous_node_ids=True,
        )
        == projected
    )


def test_projection_is_immutable_and_exposes_latest_event_per_node():
    base = ExecutionRevisionProjection("run_test", "campaign_test", True)
    after_first, _ = base.append_event(event(0, 0))
    after_second, _ = after_first.append_event(event(1, 0))
    final, _ = after_second.append_event(event(0, 1))

    assert base.events == ()
    assert after_first.watermark == 1
    assert after_second.watermark == 2
    assert [(item.node_id, item.execution_revision) for item in final.events] == [
        (0, 0),
        (1, 0),
        (0, 1),
    ]
    assert [
        (item.node_id, item.execution_revision) for item in final.terminal_events
    ] == [(0, 1), (1, 0)]


def test_semantic_builder_requires_explicit_time_and_matches_exact_event():
    base = ExecutionRevisionProjection("run_test", "campaign_test", True)
    candidate = event(0, 0)

    projected, appended = append_semantics(base, candidate)

    assert appended == candidate
    assert projected.events == (candidate,)


def test_same_key_and_semantics_are_idempotent_even_with_new_recording_time():
    first = event(0, 0)
    base, _ = ExecutionRevisionProjection(
        "run_test", "campaign_test", True
    ).append_event(first)
    retried = event(0, 0, recorded_at="2026-07-24T00:02:00Z")

    projected, appended = base.append_event(retried)

    assert projected is base
    assert appended is first
    assert appended.recorded_at == "2026-07-24T00:01:00Z"


def test_same_key_with_different_semantics_conflicts():
    first = event(0, 0)
    base, _ = ExecutionRevisionProjection(
        "run_test", "campaign_test", True
    ).append_event(first)

    with pytest.raises(RevisionProjectionConflictError, match="semantic"):
        base.append_event(event(0, 0, feedback="different"))


@pytest.mark.parametrize(
    "payload",
    [
        b"{",
        b"{}\n\n",
        b"\n",
        b"not-json\n",
        b'{"measurements":{"score":NaN}}\n',
        b'{"measurements":{"score":Infinity}}\n',
    ],
)
def test_parse_rejects_partial_blank_malformed_and_nonfinite_payloads(payload):
    with pytest.raises((ValueError, UnicodeError)):
        ExecutionRevisionProjection.from_jsonl_bytes(
            payload,
            run_id="run_test",
            campaign_id="campaign_test",
            require_contiguous_node_ids=True,
        )


def test_parse_rejects_noncanonical_json_even_when_event_is_valid():
    candidate = event(0, 0)
    noncanonical = (
        json.dumps(candidate.to_dict(), ensure_ascii=False, sort_keys=False).encode()
        + b"\n"
    )
    assert noncanonical != canonical_json_bytes(candidate.to_dict()) + b"\n"

    with pytest.raises(RevisionProjectionError, match="not canonical"):
        ExecutionRevisionProjection.from_jsonl_bytes(
            noncanonical,
            run_id="run_test",
            campaign_id="campaign_test",
            require_contiguous_node_ids=True,
        )


def test_parse_rejects_duplicate_object_keys():
    candidate = event(0, 0)
    canonical = canonical_json_bytes(candidate.to_dict())
    duplicated = b'{"event_id":"wrong","event_id":"also-wrong"}\n'
    assert canonical != duplicated

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        ExecutionRevisionProjection.from_jsonl_bytes(
            duplicated,
            run_id="run_test",
            campaign_id="campaign_test",
            require_contiguous_node_ids=True,
        )


def test_sequence_rejects_duplicate_and_gapped_node_revisions():
    first = event(0, 0)

    with pytest.raises(RevisionProjectionConflictError, match="duplicate"):
        ExecutionRevisionProjection(
            "run_test",
            "campaign_test",
            True,
            (first, first),
        )

    with pytest.raises(RevisionProjectionConflictError, match="gap-free"):
        ExecutionRevisionProjection(
            "run_test",
            "campaign_test",
            True,
            (first, event(0, 2)),
        )


def test_parse_rejects_duplicate_revision_lines():
    first_line = canonical_json_bytes(event(0, 0).to_dict()) + b"\n"

    with pytest.raises(RevisionProjectionConflictError, match="duplicate"):
        ExecutionRevisionProjection.from_jsonl_bytes(
            first_line + first_line,
            run_id="run_test",
            campaign_id="campaign_test",
            require_contiguous_node_ids=True,
        )


def test_sequence_requires_contiguous_first_seen_node_ids():
    with pytest.raises(RevisionProjectionConflictError, match="contiguous"):
        ExecutionRevisionProjection(
            "run_test",
            "campaign_test",
            True,
            (event(1, 0),),
        )

    noncontiguous = ExecutionRevisionProjection(
        "run_test",
        "campaign_test",
        False,
        (event(5, 0), event(2, 0)),
    )
    assert tuple(item.node_id for item in noncontiguous.terminal_events) == (5, 2)


def test_sequence_rejects_recording_before_execution_started():
    candidate = event(
        0,
        0,
        started_at="2026-07-24T00:02:00Z",
        recorded_at="2026-07-24T00:01:00Z",
    )

    with pytest.raises(RevisionProjectionConflictError, match="before it started"):
        ExecutionRevisionProjection(
            "run_test",
            "campaign_test",
            True,
            (candidate,),
        )


def test_sequence_rejects_regressing_recording_chronology():
    first = event(0, 0, recorded_at="2026-07-24T00:02:00Z")
    second = event(1, 0, recorded_at="2026-07-24T00:01:00Z")

    with pytest.raises(RevisionProjectionConflictError, match="moved backwards"):
        ExecutionRevisionProjection(
            "run_test",
            "campaign_test",
            True,
            (first, second),
        )


def test_projection_rejects_cross_run_or_campaign_events():
    base = ExecutionRevisionProjection("run_test", "campaign_test", True)

    with pytest.raises(RevisionProjectionConflictError, match="identity"):
        base.append_event(event(0, 0, run_id="run_other"))
    with pytest.raises(RevisionProjectionConflictError, match="identity"):
        base.append_event(event(0, 0, campaign_id="campaign_other"))
