"""Support for the tests around a build and its handles: a ready-made
`Solution` for tests that only consume one, a git workspace a build can
read commits from, the facade stub, and a fake orchestrator that returns a
chosen `SolveResult` and records what the facade handed it."""

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence

from kapso.core.config import deep_merge, load_config
from kapso.experience.sources import experience_settings
import kapso.kapso as kapso_module
from kapso.execution.orchestrator import SolveResult
from kapso.handles import Brief, Solution
from kapso.kapso import Kapso


def make_solution(campaign: Any, objective: str = "score things", **overrides: Any) -> Solution:
    """A `Solution` with every field set to the plainest value."""
    fields = dict(
        brief=Brief(objective=objective),
        campaign=Path(campaign),
        branch="generic_exp_1",
        commit="0" * 40,
        base_commit="1" * 40,
        score=None,
        goal_achieved=False,
        stopped_reason="max_iterations",
        stop_detail=None,
        experiments=1,
        experiments_total=1,
        cost_usd=0.0,
        requests=[],
        feedback=None,
        evaluator=None,
        harness=None,
        experience=[],
        verification=None,
        rejected_candidates=0,
        callback_metrics={},
        callback_primary_metric=None,
        session_failures=[],
    )
    fields.update(overrides)
    return Solution(**fields)


def git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def git_workspace(path: Path) -> str:
    """A git repository on `main` with one commit; returns that commit."""
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q", "-b", "main")
    (path / "README.md").write_text("# seed\n")
    git(path, "add", "README.md")
    git(path, "commit", "-q", "-m", "seed")
    return git(path, "rev-parse", "HEAD")


def candidate_branch(path: Path, name: str, filename: str, content: str) -> str:
    """A candidate branch off `main` with one committed change; returns its
    commit and leaves `main` checked out, as a campaign does mid-run."""
    git(path, "checkout", "-q", "-b", name)
    (path / filename).write_text(content)
    git(path, "add", filename)
    git(path, "commit", "-q", "-m", f"candidate {name}")
    commit = git(path, "rev-parse", "HEAD")
    git(path, "checkout", "-q", "main")
    return commit


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "trace"


def recorded_campaign(tmp_path, fixture, branch="generic_exp_0", plant_commit=True):
    """The fixture's records in a campaign with a seed commit on `main`
    and the attempt's commit on its branch, with the records pointing at
    those commits; left checked out at `main`, as a campaign that delivered
    nothing is. Returns the campaign and the attempt's commit."""
    campaign = tmp_path / fixture
    shutil.copytree(FIXTURES / fixture, campaign)
    (campaign / "trace.txt").unlink()
    base = git_workspace(campaign)
    commit = candidate_branch(campaign, branch, "model.py", "improved = True\n")
    solution = campaign / ".kapso" / "solution.json"
    record = json.loads(solution.read_text(encoding="utf-8"))
    record["base_commit"] = base
    solution.write_text(json.dumps(record), encoding="utf-8")
    if plant_commit:
        state = campaign / ".kapso" / "run_state.json"
        checkpoint = json.loads(state.read_text(encoding="utf-8"))
        for node in checkpoint["strategy_state"]["node_history"]:
            if node["branch_name"] == branch:
                node["commit"] = commit
        state.write_text(json.dumps(checkpoint), encoding="utf-8")
    return campaign, commit


def deployable(tmp_path, fixture="churn-unjudged"):
    """A recorded campaign that also has what a deployment copies: a file
    tracked on `main` that the attempt deletes, a file only the attempt
    has, untracked data, and a directory the workspace never carries."""
    campaign, _ = recorded_campaign(tmp_path, fixture, plant_commit=False)
    (campaign / "old.py").write_text("legacy = True\n")
    git(campaign, "add", "old.py")
    git(campaign, "commit", "-q", "-m", "main moves on")
    git(campaign, "checkout", "-q", "generic_exp_0")
    (campaign / ".claude" / "skills" / "from_the_attempt").mkdir(parents=True)
    (campaign / ".claude" / "skills" / "from_the_attempt" / "SKILL.md").write_text("x")
    git(campaign, "add", ".claude")
    git(campaign, "commit", "-q", "-m", "the attempt's session left host files on the branch")
    commit = git(campaign, "rev-parse", "HEAD")
    git(campaign, "checkout", "-q", "main")
    (campaign / "kapso_datasets").mkdir()
    (campaign / "kapso_datasets" / "train.csv").write_text("id,y\n1,0\n")
    state = campaign / ".kapso" / "run_state.json"
    checkpoint = json.loads(state.read_text(encoding="utf-8"))
    checkpoint["strategy_state"]["node_history"][0]["commit"] = commit
    state.write_text(json.dumps(checkpoint), encoding="utf-8")
    return campaign, commit


def facade() -> Kapso:
    """A Kapso without `__init__`: the attributes `build()` reads. A new
    attribute set in `__init__` must be mirrored here (the stub gotcha).
    The config is read through the facade's module, as `__init__` reads
    it, so a test's patch of `kapso_module.load_config` (the fake
    orchestrator's inbox registry, the engine patch's) is what the facade
    runs with: patch first, build the facade after."""
    kapso = Kapso.__new__(Kapso)
    kapso.config_path = None
    kapso.knowledge_search = SimpleNamespace(is_enabled=lambda: False)
    kapso._config = kapso_module.load_config()
    kapso._bank_home = None
    kapso._harness = None
    kapso._harness_settings = dict(kapso._config["harness"])
    kapso._experience = []
    kapso._experience_settings = experience_settings(kapso._config)
    return kapso


def solve_result(**overrides: Any) -> SolveResult:
    values: Dict[str, Any] = dict(
        best_experiment=None,
        final_feedback=None,
        stopped_reason="max_iterations",
        iterations_run=1,
        total_cost=0.0,
        cumulative_iterations=1,
    )
    values.update(overrides)
    return SolveResult(**values)


def fake_orchestrator(
    monkeypatch,
    workspace: Path,
    result: SolveResult,
    *,
    best_branch: Optional[str] = None,
    score: Optional[float] = None,
    history: Sequence[Any] = (),
    inbox_registry: Optional[Path] = None,
) -> Dict[str, Any]:
    """Replace the orchestrator with one that records the handler and every
    keyword the facade passes, returns `result` from `solve()`, and checks
    `best_branch` out at delivery as the real strategy does. The inbox is
    off unless a registry path is given, so no test writes outside its
    temporary directory."""
    captured: Dict[str, Any] = {}

    class Strategy:
        def __init__(self) -> None:
            self.workspace = SimpleNamespace(workspace_dir=str(workspace))

        def get_experiment_history(self):
            return list(history)

        def get_deliverable_experiment(self):
            # The delivered node, as the real strategy returns it: the
            # history entry on `best_branch`; with no history, a node whose
            # judgment is the final one (the two coincide when the last
            # node is the delivered one); None when nothing was delivered.
            if best_branch is None:
                return None
            for node in history:
                if node.branch_name == best_branch:
                    return node
            return SimpleNamespace(
                branch_name=best_branch,
                feedback=result.final_feedback.feedback if result.final_feedback else "",
                verification=None,
            )

        def get_deliverable_score(self):
            return score

        def checkout_to_best_experiment_branch(self):
            if best_branch is not None:
                git(workspace, "checkout", "-q", best_branch)
            return best_branch

    class Orchestrator:
        def __init__(self, handler, **kwargs):
            captured["handler"] = handler
            captured.update(kwargs)
            self.search_strategy = Strategy()
            self.operation_status = SimpleNamespace(
                path=str(Path(workspace) / ".kapso" / "status.json")
            )

        def solve(self, experiment_max_iter, time_budget_minutes=None, cost_budget=None,
                  finalization_reserve_minutes=None, on_status=None, finish=None):
            captured["solve"] = {
                "experiment_max_iter": experiment_max_iter,
                "time_budget_minutes": time_budget_minutes,
                "cost_budget": cost_budget,
                "finalization_reserve_minutes": finalization_reserve_minutes,
            }
            # the delivery runs inside solve(), as the real orchestrator runs it
            if finish is not None:
                finish(result)
            return result

    inbox = (
        {"enabled": True, "stop_grace_seconds": 120, "registry": str(inbox_registry)}
        # every launch registers the campaign, inbox on or off (wp6-observability.md)
        if inbox_registry is not None
        else {"enabled": False, "registry": str(Path(workspace).with_name("registry.jsonl"))}
    )
    monkeypatch.setattr(kapso_module, "load_config", lambda config_path=None: deep_merge(load_config(), {"inbox": inbox}))
    monkeypatch.setattr(kapso_module, "OrchestratorAgent", Orchestrator)
    return captured
