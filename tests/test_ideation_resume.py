"""Strict checkpoint and cross-store resume behavior for GenericSearch."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import git
import pytest

from kapso.execution.memories.experiment_memory import ExperimentHistoryStore
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.ideation import (
    IdeaArchive,
    ResolvedParentSnapshot,
)
from kapso.execution.search_strategies.generic.strategy import GenericSearch
from test_ideation_domain import (
    BATCH_ID,
    IDEA_ID,
    NOW,
    eligible_analysis,
    generated_idea,
    planned_batch,
    selection,
)
from test_run_checkpoint import _strict_generic_strategy


CAMPAIGN_ID = "campaign_" + "f" * 32


def restored_shell(workspace: Path) -> GenericSearch:
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.workspace_dir = str(workspace)
    strategy.ideation_config = {"archive_path": "ideas.json"}
    strategy.idea_archive = None
    return strategy


def test_checkpoint_is_exact_v3_and_rejects_pre_v3_state(tmp_path):
    source = _strict_generic_strategy(tmp_path)
    state = source.dump_state()

    assert state["schema"] == "kapso.generic_search.v3"
    assert state["campaign_id"] == CAMPAIGN_ID
    assert state["archive_revision"] == 5
    assert state["active_batch_id"] == BATCH_ID

    legacy = dict(state)
    legacy.pop("schema")
    with pytest.raises(ValueError, match="fields are incompatible"):
        restored_shell(tmp_path).load_state(legacy)


def test_checkpoint_rejects_missing_archive_and_archive_behind(tmp_path):
    source = _strict_generic_strategy(tmp_path)
    valid_state = source.dump_state()
    state = dict(valid_state)
    state["archive_revision"] += 1
    with pytest.raises(ValueError, match="behind"):
        restored_shell(tmp_path).load_state(state)

    (tmp_path / "ideas.json").unlink()
    with pytest.raises(ValueError, match="archive is missing"):
        restored_shell(tmp_path).load_state(valid_state)


def test_checkpoint_requires_exact_search_node_fields(tmp_path):
    source = _strict_generic_strategy(tmp_path)
    state = source.dump_state()
    state["node_history"][0]["legacy_field"] = True

    with pytest.raises(ValueError, match="node fields are incompatible"):
        restored_shell(tmp_path).load_state(state)


def initialized_repo(path: Path) -> git.Repo:
    path.mkdir()
    repo = git.Repo.init(path)
    with repo.config_writer() as config:
        config.set_value("user", "name", "Resume Test")
        config.set_value("user", "email", "resume@example.com")
    (path / "README.md").write_text("baseline\n", encoding="utf-8")
    repo.git.add(["README.md"])
    repo.git.commit("-m", "baseline")
    repo.git.branch("-M", "main")
    return repo


def linked_strategy(tmp_path: Path) -> tuple[GenericSearch, IdeaArchive, git.Repo]:
    workspace = tmp_path / "workspace"
    repo = initialized_repo(workspace)
    parent = ResolvedParentSnapshot(
        node_id=None,
        branch_name="main",
        git_ref=repo.head.commit.hexsha,
        materialized_ref=repo.head.commit.hexsha,
        diff_base_ref=repo.head.commit.hexsha,
        feedback_base_ref=repo.head.commit.hexsha,
    )
    batch = planned_batch()
    batch = replace(
        batch,
        campaign_id=CAMPAIGN_ID,
        evidence_snapshot=replace(
            batch.evidence_snapshot,
            campaign_id=CAMPAIGN_ID,
        ),
        resolved_parents=(parent,),
    )
    idea = replace(generated_idea(), resolved_parent=parent)
    archive = IdeaArchive(workspace / "ideas.json", CAMPAIGN_ID)
    archive.create_batch(batch, expected_revision=0)
    archive.add_ideas(BATCH_ID, (idea,), expected_revision=1)
    archive.record_analysis(BATCH_ID, eligible_analysis(), expected_revision=2)
    archive.record_selection(BATCH_ID, selection(), expected_revision=3)
    archive.link_experiment(IDEA_ID, 0, BATCH_ID, expected_revision=4)
    repo.create_head("generic_exp_0", repo.head.commit)

    strategy = GenericSearch.__new__(GenericSearch)
    strategy.workspace_dir = str(workspace)
    strategy.workspace = SimpleNamespace(repo=repo)
    strategy.ideation_config = {"archive_path": "ideas.json"}
    strategy.ideation_campaign_id = CAMPAIGN_ID
    strategy.idea_archive = archive
    strategy.active_batch_id = BATCH_ID
    strategy.node_history = []
    strategy.problem_handler = SimpleNamespace(maximize_scoring=True)
    return strategy, archive, repo


def finalized_node() -> SearchNode:
    return SearchNode(
        node_id=0,
        idea_id=IDEA_ID,
        selection_batch_id=BATCH_ID,
        solution=generated_idea().proposal,
        branch_name="generic_exp_0",
        parent_branch_name="main",
        score=0.5,
        evaluation_valid=True,
        started_at=NOW,
        duration_seconds=2.0,
        cost_usd=0.25,
    )


def experiment_store(workspace: Path) -> ExperimentHistoryStore:
    return ExperimentHistoryStore(
        str(workspace / "experiments.json"),
        objective_direction="maximize",
        require_idea_links=True,
    )


def test_finalized_checkpoint_node_recreates_record_and_outcome(tmp_path):
    strategy, archive, _ = linked_strategy(tmp_path)
    strategy.node_history = [finalized_node()]
    store = experiment_store(Path(strategy.workspace_dir))

    strategy.reconcile_experiment_memory(store)

    assert [record.node_id for record in store.experiments] == [0]
    assert archive.get_idea(IDEA_ID).outcome is not None
    assert strategy.active_batch_id is None


def test_experiment_record_recreates_node_and_outcome_without_rerun(tmp_path):
    strategy, archive, _ = linked_strategy(tmp_path)
    store = experiment_store(Path(strategy.workspace_dir))
    store.add_experiment(finalized_node())

    strategy.reconcile_experiment_memory(store)

    assert len(strategy.node_history) == 1
    assert strategy.node_history[0].idea_id == IDEA_ID
    assert archive.get_idea(IDEA_ID).outcome is not None
    assert strategy.active_batch_id is None


def test_bridged_idea_without_record_becomes_same_node_recovery(tmp_path):
    strategy, archive, _ = linked_strategy(tmp_path)
    store = experiment_store(Path(strategy.workspace_dir))

    strategy.reconcile_experiment_memory(store)

    assert len(strategy.node_history) == 1
    node = strategy.node_history[0]
    assert node.node_id == 0
    assert node.idea_id == IDEA_ID
    assert node.selection_batch_id == BATCH_ID
    assert node.had_error is True
    assert node.recoverable_error is True
    assert archive.get_idea(IDEA_ID).outcome is None
    assert store.experiments == []
