"""Pure canonical experiment-history projection tests."""

import subprocess
import sys
from dataclasses import replace

import pytest

from kapso.core.embedding_contracts import EmbeddingRecord, complete_input_hash
from kapso.cross_run.canonical import canonical_json_bytes, content_id
from kapso.execution.memories.experiment_memory.projection import (
    ExperimentHistoryProjection,
    build_experiment_history_genesis,
    project_records,
)
from kapso.execution.memories.experiment_memory.record import ExperimentRecord
from kapso.execution.search_strategies.node import SearchNode

IDEA_ID = "idea_0123456789abcdef0123456789abcdef"
BATCH_ID = "batch_0123456789abcdef0123456789abcdef"
STARTED_AT = "2026-07-24T00:00:00Z"
EMBEDDING_SPACE_ID = content_id(
    "embedding-space",
    {
        "provider": "openai",
        "model": "test-embedding",
        "dimensions": 2,
        "canonicalizer_version": "kapso.embedding_input.v1",
    },
)
EMBEDDING_DIMENSIONS = 2


def embedding(
    solution: str,
    vector: tuple[float, ...],
    *,
    canonicalizer_version: str = "kapso.embedding_input.v1",
    input_text: str | None = None,
) -> EmbeddingRecord:
    return EmbeddingRecord(
        provider="openai",
        model="test-embedding",
        dimensions=len(vector),
        canonicalizer_version=canonicalizer_version,
        input_hash=complete_input_hash(solution if input_text is None else input_text),
        vector=vector,
    )


def node(
    node_id: int,
    *,
    execution_revision: int = 0,
    solution: str | None = None,
    feedback: str = "measured",
) -> SearchNode:
    return SearchNode(
        node_id=node_id,
        execution_revision=execution_revision,
        idea_id=IDEA_ID,
        selection_batch_id=BATCH_ID,
        parent_node_id=None if node_id == 0 else node_id - 1,
        solution=solution or f"candidate {node_id}",
        branch_name=f"candidate-{node_id}",
        feedback=feedback,
        score=None,
        evaluation_valid=True,
        started_at=STARTED_AT,
        build_fidelity="full",
        eval_fidelity="fast",
    )


def genesis() -> ExperimentHistoryProjection:
    return build_experiment_history_genesis(
        run_id="run_test",
        campaign_id="campaign_test",
        embedding_space_id=EMBEDDING_SPACE_ID,
        embedding_provider="openai",
        embedding_model="test-embedding",
        embedding_dimensions=EMBEDDING_DIMENSIONS,
        embedding_canonicalizer_version="kapso.embedding_input.v1",
        objective_direction="maximize",
        require_idea_links=True,
    )


def recoverable_node(node_id: int) -> SearchNode:
    return replace(
        node(node_id),
        had_error=True,
        recoverable_error=True,
        error_message="retryable implementation failure",
        evaluation_valid=False,
    )


def test_projection_import_has_no_agent_or_provider_runtime_dependency():
    script = "\n".join(
        (
            "import sys",
            "sys.modules['litellm'] = None",
            "sys.modules['kapso.core.llm'] = None",
            "sys.modules['kapso.execution.coding_agents.factory'] = None",
            (
                "from kapso.execution.memories.experiment_memory.projection "
                "import ExperimentHistoryProjection"
            ),
            "assert ExperimentHistoryProjection.__module__.endswith('.projection')",
        )
    )

    completed = subprocess.run(
        (sys.executable, "-c", script),
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout == ""
    assert completed.stderr == ""


def test_genesis_uses_canonical_bytes_without_pretty_print_or_newline():
    state = genesis()

    assert state.to_json_bytes() == (
        b'{"campaign_id":"campaign_test",'
        b'"embedding_canonicalizer_version":"kapso.embedding_input.v1",'
        b'"embedding_dimensions":2,"embedding_model":"test-embedding",'
        b'"embedding_provider":"openai",'
        b'"embedding_space_id":"' + EMBEDDING_SPACE_ID.encode("utf-8") + b'",'
        b'"objective_direction":"maximize",'
        b'"records":[],"require_idea_links":true,"revision":0,'
        b'"run_id":"run_test","schema":"kapso.run_experiment_history.v1"}'
    )
    assert not state.to_json_bytes().endswith(b"\n")
    assert ExperimentHistoryProjection.from_json_bytes(state.to_json_bytes()) == state


@pytest.mark.parametrize(
    "payload",
    [
        (
            b'{"campaign_id":"campaign_test","objective_direction":"maximize",'
            b'"records":[],"require_idea_links":true,"revision":0,'
            b'"run_id":"run_test","schema":"kapso.run_experiment_history.v1",'
            b'"schema":"kapso.run_experiment_history.v1"}'
        ),
        (
            b'{"campaign_id":"campaign_test","objective_direction":"maximize",'
            b'"records":[],"require_idea_links":true,"revision":NaN,'
            b'"run_id":"run_test","schema":"kapso.run_experiment_history.v1"}'
        ),
    ],
)
def test_parser_rejects_duplicate_keys_and_nonfinite_numbers(payload):
    with pytest.raises(ValueError):
        ExperimentHistoryProjection.from_json_bytes(payload)


def test_parser_rejects_unknown_document_and_record_fields():
    document = genesis().to_dict()
    document["unknown"] = "not allowed"
    with pytest.raises(ValueError, match="fields"):
        ExperimentHistoryProjection.from_json_bytes(canonical_json_bytes(document))

    projected = project_records(
        predecessor=genesis(),
        nodes=(node(0),),
        embeddings_by_node_revision={(0, 0): embedding("candidate 0", (0.1, 0.2))},
    ).to_dict()
    projected["records"][0]["unknown"] = "not allowed"
    with pytest.raises(ValueError, match="record fields"):
        ExperimentHistoryProjection.from_json_bytes(canonical_json_bytes(projected))


def test_parser_rejects_noncanonical_and_nonbyte_payloads():
    state = genesis()
    with pytest.raises(ValueError, match="bytes must be canonical"):
        ExperimentHistoryProjection.from_json_bytes(b" " + state.to_json_bytes())
    with pytest.raises(ValueError, match="payload must be bytes"):
        ExperimentHistoryProjection.from_json_bytes(
            state.to_json_bytes().decode("utf-8")
        )


def test_successor_appends_updates_and_reuses_embedding_exactly_once():
    first = project_records(
        predecessor=genesis(),
        nodes=(recoverable_node(0),),
        embeddings_by_node_revision={(0, 0): embedding("candidate 0", (0.1, 0.2))},
    )
    updated = project_records(
        predecessor=first,
        nodes=(node(0, execution_revision=1, feedback="better"), node(1)),
        embeddings_by_node_revision={(1, 0): embedding("candidate 1", (0.3, 0.4))},
    )

    assert updated.revision == 3
    assert tuple(record.node_id for record in updated.records) == (0, 1)
    assert updated.records[0].execution_revision == 1
    assert updated.records[0].solution_embedding == (0.1, 0.2)
    assert updated.records[1].solution_embedding == (0.3, 0.4)

    with pytest.raises(ValueError, match="match new node revisions exactly"):
        project_records(
            predecessor=first,
            nodes=(node(0, execution_revision=1, feedback="better"),),
            embeddings_by_node_revision={(0, 1): embedding("candidate 0", (9.0, 9.0))},
        )


def test_batch_can_apply_multiple_exact_revisions_for_one_new_node():
    projected = project_records(
        predecessor=genesis(),
        nodes=(
            recoverable_node(0),
            node(0, execution_revision=1, feedback="full measurement"),
        ),
        embeddings_by_node_revision={(0, 0): embedding("candidate 0", (0.1, 0.2))},
    )

    assert projected.revision == 2
    assert projected.records[0].execution_revision == 1
    assert projected.records[0].solution_embedding == (0.1, 0.2)


def test_new_node_requires_one_typed_input_and_space_bound_embedding():
    with pytest.raises(ValueError, match="requires one exact embedding"):
        project_records(
            predecessor=genesis(),
            nodes=(node(0),),
            embeddings_by_node_revision={},
        )
    with pytest.raises(ValueError, match="exact record"):
        project_records(
            predecessor=genesis(),
            nodes=(node(0),),
            embeddings_by_node_revision={(0, 0): (0.1, 0.2)},
        )
    with pytest.raises(ValueError, match="another space"):
        project_records(
            predecessor=genesis(),
            nodes=(node(0),),
            embeddings_by_node_revision={
                (0, 0): embedding(
                    "candidate 0",
                    (0.1, 0.2),
                    canonicalizer_version="kapso.knowledge_embedding.v1",
                )
            },
        )
    with pytest.raises(ValueError, match="another input"):
        project_records(
            predecessor=genesis(),
            nodes=(node(0),),
            embeddings_by_node_revision={
                (0, 0): embedding(
                    "candidate 0",
                    (0.1, 0.2),
                    input_text="another complete solution",
                )
            },
        )


def test_projection_rejects_rollback_gap_conflict_and_identity_change():
    first = project_records(
        predecessor=genesis(),
        nodes=(recoverable_node(0),),
        embeddings_by_node_revision={(0, 0): embedding("candidate 0", (0.1, 0.2))},
    )
    second = project_records(
        predecessor=first,
        nodes=(node(0, execution_revision=1, feedback="better"),),
        embeddings_by_node_revision={},
    )

    with pytest.raises(ValueError, match="moved backwards"):
        project_records(
            predecessor=second,
            nodes=(node(0),),
            embeddings_by_node_revision={},
        )
    with pytest.raises(ValueError, match="contains a gap"):
        project_records(
            predecessor=second,
            nodes=(node(0, execution_revision=3),),
            embeddings_by_node_revision={},
        )
    with pytest.raises(ValueError, match="conflicts"):
        project_records(
            predecessor=second,
            nodes=(node(0, execution_revision=1, feedback="conflict"),),
            embeddings_by_node_revision={},
        )
    with pytest.raises(ValueError, match="identity changed"):
        project_records(
            predecessor=second,
            nodes=(
                replace(
                    node(0, execution_revision=2),
                    idea_id="idea_ffffffffffffffffffffffffffffffff",
                ),
            ),
            embeddings_by_node_revision={},
        )


def test_projection_rejects_node_and_revision_gaps():
    with pytest.raises(ValueError, match="node ids must be contiguous"):
        project_records(
            predecessor=genesis(),
            nodes=(node(1),),
            embeddings_by_node_revision={(1, 0): embedding("candidate 1", (0.1, 0.2))},
        )
    with pytest.raises(ValueError, match="start at revision zero"):
        project_records(
            predecessor=genesis(),
            nodes=(node(0, execution_revision=1),),
            embeddings_by_node_revision={(0, 1): embedding("candidate 0", (0.1, 0.2))},
        )
    with pytest.raises(ValueError, match="timestamp"):
        project_records(
            predecessor=genesis(),
            nodes=(
                replace(
                    node(0),
                    started_at="2026-07-24T00:00:00+00:00",
                ),
            ),
            embeddings_by_node_revision={(0, 0): embedding("candidate 0", (0.1, 0.2))},
        )


def test_tree_projection_preserves_arbitrary_executed_node_order():
    tree_genesis = build_experiment_history_genesis(
        run_id="run_test",
        campaign_id="campaign_test",
        embedding_space_id=EMBEDDING_SPACE_ID,
        embedding_provider="openai",
        embedding_model="test-embedding",
        embedding_dimensions=EMBEDDING_DIMENSIONS,
        embedding_canonicalizer_version="kapso.embedding_input.v1",
        objective_direction="maximize",
        require_idea_links=False,
    )
    node_five = replace(
        node(5),
        idea_id=None,
        selection_batch_id=None,
        parent_node_id=None,
    )

    projected = project_records(
        predecessor=tree_genesis,
        nodes=(node_five,),
        embeddings_by_node_revision={(5, 0): embedding("candidate 5", (0.5, 0.6))},
    )

    assert tuple(record.node_id for record in projected.records) == (5,)


def test_tree_record_requires_a_caller_supplied_canonical_timestamp():
    tree_node = replace(
        node(5),
        idea_id=None,
        selection_batch_id=None,
        parent_node_id=None,
        started_at="",
    )

    with pytest.raises(ValueError, match="timestamp"):
        ExperimentRecord.from_node(
            tree_node,
            objective_direction="maximize",
            require_idea_links=False,
        )


def test_state_rejects_noncontiguous_records_and_cross_identity_changes():
    first = project_records(
        predecessor=genesis(),
        nodes=(node(0),),
        embeddings_by_node_revision={(0, 0): embedding("candidate 0", (0.1, 0.2))},
    )
    record = first.records[0]
    with pytest.raises(ValueError, match="contiguous"):
        replace(first, records=(replace(record, node_id=1),))
    with pytest.raises(ValueError, match="objective direction changed"):
        replace(first, objective_direction="minimize")
    with pytest.raises(ValueError, match="requires record idea links"):
        replace(
            first,
            records=(replace(record, idea_id=None, selection_batch_id=None),),
        )
    with pytest.raises(ValueError, match="revision differs from its records"):
        replace(first, revision=first.revision + 1)


def test_projection_is_pure_and_creates_no_files(tmp_path):
    state = project_records(
        predecessor=genesis(),
        nodes=(node(0),),
        embeddings_by_node_revision={(0, 0): embedding("candidate 0", (0.1, 0.2))},
    )

    assert state.to_json_bytes()
    assert tuple(tmp_path.iterdir()) == ()
