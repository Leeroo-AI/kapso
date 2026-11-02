"""The experience in every record (wp3-experience.md, change 6): the
solution, the checkpoint, the status file and the launch record carry the
loaded sources with their pinned head and scope; a resume with nothing
loaded takes the record's list and re-pins each bank at the recorded
head, refuses a changed list by name, a gone source by its own message
and a re-ingested index; the judge's attribution rides the node."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from kapso.execution.inbox import write_launch_record
from kapso.execution.observability import BuildStatus
from kapso.execution.orchestrator import OrchestratorAgent
from kapso.execution.run_checkpoint import (
    RunCheckpoint,
    RunCheckpointCorruptError,
    RunCheckpointIncompatibleError,
    RunCheckpointStore,
    config_fingerprint,
)
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.strategy import GenericSearch
from kapso.experience import ExperienceError, ExperienceReferenceError, ExperienceStore, IndexRecord
from kapso.handles import Brief, Budget, ExperienceRef, Solution
from kapso.kapso import Kapso
from tests.handles_support import facade, git_workspace, make_solution, solve_result
from tests.harness_support import patch_engine, stub_solve
from tests.test_bank_retriever import card_text
from tests.test_experience_reference import settings_for
from tests.test_inbox_cli import _paused_campaign
from tests.test_update_frame import seed_bank_home

MINE = ExperienceRef(
    reference="mine", kind="bank", resolved="/home/me/.kapso/bank.git", head="a" * 40,
    provenance="measured by you", scope="harness",
)
TEAM = ExperienceRef(
    reference="acme/kapso-bank", kind="bank", resolved="/home/me/.kapso/experience/banks/x", head="b" * 40,
    provenance="measured by others", scope="all",
)


# ------------------------------------------------------------- the records

def test_the_solution_record_carries_each_source_with_its_scope(tmp_path):
    solution = make_solution(tmp_path, experience=[MINE, TEAM])
    solution.save()
    data = json.loads(solution.record_path.read_text())
    assert data["schema_version"] == 5
    assert [(s["reference"], s["head"], s["scope"]) for s in data["experience"]] == [
        ("mine", "a" * 40, "harness"), ("acme/kapso-bank", "b" * 40, "all"),
    ]
    assert "knowledge_index" not in data and "experience_head" not in data
    reopened = Solution.open(str(tmp_path))
    assert reopened.experience == [MINE, TEAM]
    text = reopened.explain()
    assert f"Experience:      mine at {'a' * 12}, measured by you" in text
    assert f"Experience:      acme/kapso-bank at {'b' * 12}, measured by others (every scope)" in text


def test_the_checkpoint_carries_the_sources_and_requires_the_field(tmp_path):
    checkpoint = RunCheckpoint.create(
        strategy_type="generic", goal="g", config_fingerprint=config_fingerprint({"m": 1}),
        status="running", completed_iterations=1, cumulative_cost=0.0, current_feedback=None,
        strategy_state={"node_history": []}, experience=[vars(MINE)],
    )
    assert checkpoint.schema_version == 7
    store = RunCheckpointStore(str(tmp_path))
    store.save(checkpoint)
    assert store.load().experience == [vars(MINE)]
    data = json.loads(store.path.read_text())
    del data["experience"]
    store.path.write_text(json.dumps(data))
    with pytest.raises(RunCheckpointCorruptError, match="experience"):
        store.load()


def test_the_status_file_names_the_sources_from_the_first_second(tmp_path):
    status = BuildStatus(tmp_path / "status.json", phase="setup", harness=None, experience=[vars(MINE)])
    status.update(heartbeat_seconds=None)
    assert json.loads((tmp_path / "status.json").read_text())["experience"] == [vars(MINE)]


def test_the_judges_attribution_rides_the_node():
    node = SearchNode(node_id=1, branch_name="b", cards_load_bearing=["group-relative-standing"])
    assert SearchNode.from_dict(node.to_dict()).cards_load_bearing == ["group-relative-standing"]
    assert SearchNode(node_id=2, branch_name="c").cards_load_bearing == []


# --------------------------------------------------------------- the resume

def _git_workspace(path: Path) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@t",
         "commit", "-q", "--allow-empty", "-m", "seed"],
        check=True,
    )


def _push_card(home: Path, name: str) -> str:
    """A later commit on the bank; returns the new head."""
    work = home.parent / f"push-{name}"
    subprocess.run(["git", "clone", "-q", str(home), str(work)], check=True)
    (work / "insights" / f"{name}.md").write_text(card_text(name))
    subprocess.run(["git", "-C", str(work), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(work), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "-m", name], check=True)
    subprocess.run(["git", "-C", str(work), "push", "-q", "origin", "main"], check=True)
    return subprocess.run(["git", "-C", str(work), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()


def _head(home: Path) -> str:
    return subprocess.run(["git", "--git-dir", str(home), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()


def _campaign(tmp_path, monkeypatch, experience):
    """A resumable campaign: a workspace with a checkpoint and a launch
    record naming `experience`; the orchestrator a fake that captures its
    arguments."""
    workspace = tmp_path / "campaign"
    _git_workspace(workspace)
    RunCheckpointStore(str(workspace)).save(RunCheckpoint.create(
        strategy_type="generic", goal=Brief(objective="Improve support").render(),
        config_fingerprint=config_fingerprint({"m": 1}), status="running", completed_iterations=1,
        cumulative_cost=1.0, current_feedback=None, strategy_state={"node_history": []},
        experience=experience,
    ))
    write_launch_record(workspace, {
        "config_path": None, "host": None,
        "output_path": str(workspace), "experiments": 3, "minutes": None,
        "usd": None, "reserve_minutes": None, "eval_dir": None,
        "data_dir": None, "harness": None,
        "experience": experience, "resumable_from_inbox": True, "dotenv_path": "",
    })
    captured = {}

    class FakeOrchestrator:
        def __init__(self, handler, **kwargs):
            captured.update(kwargs)
            self.search_strategy = SimpleNamespace(
                workspace=SimpleNamespace(workspace_dir=str(workspace)),
                checkout_to_best_experiment_branch=lambda: "main",
                get_deliverable_experiment=lambda: None,
                get_deliverable_score=lambda: None,
                get_experiment_history=lambda best_last=True: [],
                iteration_evaluations=[],
                ideation_gates=["research", "bank"],
            )
            self.operation_status = SimpleNamespace(path=str(workspace / ".kapso" / "status.json"))

        def solve(self, **kwargs):
            result = SimpleNamespace(
                final_feedback=None, iterations_run=0, cumulative_iterations=1, total_cost=0.0,
                stopped_reason="max_iterations", stop_detail=None, best_experiment=None,
                platform_budget=None, requests=[],
            )
            # the delivery runs inside solve(), as the real orchestrator runs it
            kwargs["finish"](result)
            return result

    monkeypatch.setattr("kapso.kapso.OrchestratorAgent", FakeOrchestrator)
    return workspace, captured


def _kapso(tmp_path, home: Path) -> Kapso:
    kapso = facade()
    kapso._experience_settings = {**settings_for(tmp_path), "bank_home": str(home)}
    kapso._bank_home = home
    return kapso


def test_a_resume_with_nothing_loaded_re_pins_each_bank_at_the_recorded_head(tmp_path, monkeypatch):
    home = seed_bank_home(tmp_path, {"a-card": card_text("a-card")})
    launched = _head(home)
    recorded = {"reference": "mine", "kind": "bank", "resolved": str(home), "head": launched,
                "provenance": "measured by you", "scope": "harness"}
    workspace, captured = _campaign(tmp_path, monkeypatch, [recorded])
    moved = _push_card(home, "later-card")
    assert moved != launched

    kapso = _kapso(tmp_path, home)
    kapso.build(output=str(workspace), resume=True)
    # the object, the orchestrator and the staged clone all sit at the launch's head
    assert [(s.reference, s.head) for s in kapso.experience] == [("mine", launched)]
    assert captured["experience"] == [recorded]
    [entry] = json.loads(captured["strategy_params_overrides"]["bank_serving"]["KAPSO_BANKS"])
    assert entry["head"] == launched
    assert subprocess.run(["git", "-C", entry["dir"], "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip() == launched
    assert not (Path(entry["dir"]) / "insights" / "later-card.md").exists()
    record = yaml.safe_load((workspace / ".kapso" / "serving" / "serving-record.yaml").read_text())
    assert record["sources"][0]["head"] == launched
    assert Solution.open(str(workspace)).experience[0].head == launched


def test_a_fresh_build_takes_the_briefs_experience_line_and_a_loaded_list_wins_over_it(tmp_path, monkeypatch):
    home = seed_bank_home(tmp_path, {"a-card": card_text("a-card")})
    _workspace, captured = _campaign(tmp_path, monkeypatch, [])
    monkeypatch.chdir(tmp_path)

    # Nothing loaded: the brief's items are loaded, in order, and served.
    kapso = _kapso(tmp_path, home)
    kapso.build(Brief(objective="Improve support", experience=["mine:all"]), output=str(tmp_path / "fresh"))
    assert [(s.reference, s.scope) for s in kapso.experience] == [("mine", "all")]
    assert [(r["reference"], r["scope"]) for r in captured["experience"]] == [("mine", "all")]

    # A loaded list wins over the brief's line, which is not even resolved.
    kapso = _kapso(tmp_path, home)
    kapso.load_experience("mine")
    kapso.build(Brief(objective="Improve support", experience=["./nowhere/bank.git"]), output=str(tmp_path / "second"))
    assert [(s.reference, s.scope) for s in kapso.experience] == [("mine", "harness")]

    # A brief line that names nothing refuses the build by name.
    with pytest.raises(ExperienceReferenceError, match="a reference written as a path must exist"):
        _kapso(tmp_path, home).build(Brief(objective="Improve support", experience=["./nowhere/bank.git"]), output=str(tmp_path / "third"))


def test_a_resume_that_names_a_different_list_is_refused_by_name(tmp_path, monkeypatch):
    home = seed_bank_home(tmp_path, {"a-card": card_text("a-card")})
    recorded = {"reference": "mine", "kind": "bank", "resolved": str(home), "head": _head(home),
                "provenance": "measured by you", "scope": "harness"}
    workspace, _captured = _campaign(tmp_path, monkeypatch, [recorded])
    kapso = _kapso(tmp_path, home)
    kapso.load_experience("mine:all")
    with pytest.raises(RunCheckpointIncompatibleError, match="experience was"):
        kapso.build(output=str(workspace), resume=True)


def test_a_resume_from_another_directory_finds_a_bank_read_by_path(tmp_path, monkeypatch):
    # Regression: the launch recorded `./team/bank.git` and a resume from
    # elsewhere resolved it against the new directory. The record's
    # absolute location is what a resume re-loads.
    home = seed_bank_home(tmp_path / "team", {"a-card": card_text("a-card")})
    recorded = {"reference": "./team/bank-home.git", "kind": "bank", "resolved": str(home), "head": _head(home),
                "provenance": "measured by others", "scope": "all"}
    workspace, captured = _campaign(tmp_path, monkeypatch, [recorded])
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    kapso = _kapso(tmp_path, home)
    kapso.build(output=str(workspace), resume=True)
    assert [(s.resolved, s.head) for s in kapso.experience] == [(str(home), _head(home))]
    assert captured["experience"][0]["resolved"] == str(home)


def test_a_reply_rebuilds_the_object_with_the_launchs_bank(tmp_path, monkeypatch):
    # Regression: a reply constructed a default Kapso, so a launch whose
    # bank lived elsewhere (`Kapso(bank=)`) resumed on the default home.
    # The record's `mine` entry names the home, and the reply uses it.
    home = seed_bank_home(tmp_path, {"a-card": card_text("a-card")})
    workspace = _paused_campaign(tmp_path)
    write_launch_record(workspace, {
        "config_path": None, "host": None,
        "output_path": str(workspace), "experiments": 3, "minutes": None,
        "usd": None, "reserve_minutes": None, "eval_dir": None,
        "data_dir": None, "harness": None,
        "experience": [{"reference": "mine", "kind": "bank", "resolved": str(home), "head": _head(home),
                        "provenance": "measured by you", "scope": "harness"}],
        "resumable_from_inbox": True, "dotenv_path": "",
    })
    built = {}
    monkeypatch.setattr(Kapso, "build", lambda self, **kwargs: built.update(bank=self._bank_home, **kwargs))
    Kapso.reply(str(workspace), 1, "done")
    assert built["bank"] == home and built["resume"] is True


def test_a_resume_refuses_a_gone_source_and_a_re_ingested_index(tmp_path, monkeypatch):
    home = seed_bank_home(tmp_path, {"a-card": card_text("a-card")})
    gone = {"reference": "./team/bank.git", "kind": "bank", "resolved": str(tmp_path / "team" / "bank.git"),
            "head": "c" * 40, "provenance": "measured by others", "scope": "all"}
    workspace, _captured = _campaign(tmp_path, monkeypatch, [gone])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ExperienceReferenceError, match="a reference written as a path must exist"):
        _kapso(tmp_path, home).build(output=str(workspace), resume=True)

    settings = settings_for(tmp_path)
    store = ExperienceStore(settings["banks"], settings["indexes"])
    store.write_index_record(IndexRecord(
        name="org-x", id="new-id", created_at="2026-10-25T12:00:00+00:00", source="https://github.com/org/x",
        search_backend="kg_graph_search", backend_refs={"collection": "org-x"}, page_count=3, path=store.index_dir("org-x"),
    ))
    stale = {"reference": "org-x", "kind": "index", "resolved": str(store.index_dir("org-x")), "head": "old-id",
             "provenance": "reported by others", "scope": "all"}
    workspace, _captured = _campaign(tmp_path / "second", monkeypatch, [stale])
    with pytest.raises(ExperienceError, match="org-x was re-ingested since this campaign launched"):
        _kapso(tmp_path, home).build(output=str(workspace), resume=True)


def test_the_real_orchestrator_hands_the_factory_the_sources_it_accepts(tmp_path, monkeypatch):
    """The facade, the orchestrator, the factory and the real strategy's
    constructor, with no fake between them: the loaded sources arrive as
    recorded (the status file names them, the strategy mounts the bank
    gate for them) and every keyword the orchestrator passes is one the
    factory takes. The facade tests above fake the orchestrator and the
    harness tests fake the strategy, so nothing else walks the seam; the
    proof run's first build died on it."""
    strategies = patch_engine(monkeypatch, real_strategy=True)
    stub_solve(monkeypatch, solve_result(stopped_reason="max_iterations"))
    home = seed_bank_home(tmp_path, {"a-card": card_text("a-card")})
    workspace = tmp_path / "campaign"
    git_workspace(workspace)
    kapso = _kapso(tmp_path, home)
    kapso.load_experience("mine")
    solution = kapso.build("Improve support", output=str(workspace), budget=Budget(experiments=1))

    [strategy] = strategies
    recorded = [vars(source.ref()) for source in kapso.experience]
    assert isinstance(strategy, GenericSearch) and strategy.harness is None
    assert strategy.bank_serving is not None and "bank" in strategy.ideation_gates
    status = json.loads((workspace / ".kapso" / "status.json").read_text())
    assert status["experience"] == recorded
    assert solution.experience == [source.ref() for source in kapso.experience] and solution.branch is None
