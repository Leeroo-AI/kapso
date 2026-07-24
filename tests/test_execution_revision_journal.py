from dataclasses import replace

import git
import pytest

from kapso.cross_run.capture.journal import JournalConflictError
from kapso.cross_run.contracts import EpisodeEvaluationStatus
from kapso.cross_run.github.command import GitHubCommandError
from kapso.execution.memories.experiment_memory.store import ExperimentHistoryStore
from test_experiment_idea_linkage import node


def store(tmp_path):
    return ExperimentHistoryStore(
        str(tmp_path / "history.json"),
        objective_direction="maximize",
        require_idea_links=True,
        run_id="run_test",
        campaign_id="campaign_test",
        journal_path=str(tmp_path / "events.jsonl"),
    )


def test_journal_preserves_failed_revision_before_recovery_projection(tmp_path):
    history = store(tmp_path)
    failed = node(0, 0.4)
    failed.score = None
    failed.evaluation_attempts = []
    failed.had_error = True
    failed.recoverable_error = True
    failed.evaluation_valid = False
    recovered = node(0, 0.7)
    recovered.idea_id = failed.idea_id
    recovered.selection_batch_id = failed.selection_batch_id
    recovered.solution = failed.solution
    recovered.execution_revision = 1

    history.add_experiment(failed)
    history.add_experiment(recovered)

    events = history.revision_journal.read_events()
    assert [(event.node_id, event.execution_revision) for event in events] == [
        (0, 0),
        (0, 1),
    ]
    assert history.revision == 2
    assert history.experiments[0].execution_revision == 1


def test_same_revision_is_idempotent_and_conflicting_content_is_rejected(tmp_path):
    history = store(tmp_path)
    candidate = node(0, 0.4)
    history.add_experiment(candidate)
    history.add_experiment(candidate)
    conflicting = replace(candidate, feedback="different")

    with pytest.raises(JournalConflictError, match="conflicts"):
        history.add_experiment(conflicting)

    assert history.revision_journal.watermark == 1


def test_rejected_identity_change_does_not_advance_either_authority(tmp_path):
    history = store(tmp_path)
    candidate = node(0, 0.4)
    history.add_experiment(candidate)
    invalid_revision = replace(
        candidate,
        execution_revision=1,
        solution="a different intervention",
    )

    with pytest.raises(ValueError, match="identity or revision changed"):
        history.add_experiment(invalid_revision)

    assert history.revision == 1
    assert history.revision_journal.watermark == 1
    assert history.experiments[0].solution == candidate.solution


def test_store_recovers_crash_between_journal_and_projection_replace(
    tmp_path, monkeypatch
):
    history = store(tmp_path)
    candidate = node(0, 0.4)
    original_save = history._save

    def fail_save(records, revision):
        raise OSError("simulated projection interruption")

    monkeypatch.setattr(history, "_save", fail_save)
    with pytest.raises(OSError, match="interruption"):
        history.add_experiment(candidate)
    monkeypatch.setattr(history, "_save", original_save)

    recovered = store(tmp_path)
    assert recovered.experiments[0].node_id == 0
    assert recovered.revision == recovered.revision_journal.watermark == 1


def test_stale_writers_serialize_reload_append_and_projection(tmp_path):
    initial = store(tmp_path)
    first = node(0, 0.4)
    initial.add_experiment(first)
    writer_a = store(tmp_path)
    writer_b = store(tmp_path)
    second = node(1, 0.5)
    recovered_first = node(0, 0.7)
    recovered_first.idea_id = first.idea_id
    recovered_first.selection_batch_id = first.selection_batch_id
    recovered_first.solution = first.solution
    recovered_first.execution_revision = 1

    writer_a.add_experiment(second)
    writer_b.add_experiment(recovered_first)

    reopened = store(tmp_path)
    assert reopened.revision == reopened.revision_journal.watermark == 3
    assert [record.node_id for record in reopened.experiments] == [0, 1]
    assert reopened.experiments[0].execution_revision == 1


def test_valid_but_unmeasured_revision_is_partial_not_valid(tmp_path):
    history = store(tmp_path)
    candidate = node(0, 0.4)
    candidate.score = None
    candidate.evaluation_attempts = []
    candidate.evaluation_valid = True

    history.add_experiment(candidate)

    assert (
        history.revision_journal.read_events()[0].evaluation_status
        is EpisodeEvaluationStatus.PARTIAL
    )


def test_experiment_store_enforces_configured_git_output_bound(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repo = git.Repo.init(workspace)
    with repo.config_writer() as config:
        config.set_value("user", "name", "Bound Test")
        config.set_value("user", "email", "bound@example.com")
    (workspace / "solution.py").write_text("VALUE = 1\n", encoding="utf-8")
    repo.git.add(["solution.py"])
    repo.git.commit("-m", "candidate")
    repo.create_head("candidate-0", repo.head.commit)
    candidate = node(0, 0.4)
    candidate.workspace_dir = str(workspace)
    candidate.evaluation_attempts = [
        replace(candidate.evaluation_attempts[0], commit_sha=repo.head.commit.hexsha)
    ]
    history = ExperimentHistoryStore(
        str(tmp_path / "history.json"),
        objective_direction="maximize",
        require_idea_links=True,
        run_id="run_test",
        campaign_id="campaign_test",
        journal_path=str(tmp_path / "events.jsonl"),
        git_command_timeout_seconds=5,
        git_command_output_bytes=8,
    )

    with pytest.raises(GitHubCommandError, match="stdout exceeds configured limit"):
        history.add_experiment(candidate)

    assert history.revision_journal.watermark == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("measurements", {"score": True}),
        ("measurements", {"score": float("inf")}),
        ("artifact_refs", {"branch": ["not", "text"]}),
        ("projection", []),
    ],
)
def test_execution_revision_event_rejects_malformed_nested_values(
    tmp_path, field, value
):
    history = store(tmp_path)
    history.add_experiment(node(0, 0.4))
    event = history.revision_journal.read_events()[0]

    with pytest.raises(ValueError):
        replace(event, **{field: value})
