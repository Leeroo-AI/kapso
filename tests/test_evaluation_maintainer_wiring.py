"""Hermetic tests for the maintainer's integration into the loop (M5b).

The real EvaluationMaintainer runs against the orchestrator harness with a
scripted agent and a fake evaluation subprocess; only the LLM boundary and
the calibration command are faked.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import kapso.execution.evaluation_maintainer.maintainer as maintainer_module
import kapso.execution.orchestrator as orchestrator_module
from kapso.execution.evaluation_integrity import AGENT_GENERATED
from kapso.execution.evaluation_maintainer import EvaluationMaintainerError
from kapso.execution.run_checkpoint import RunCheckpointStore
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.strategy import GenericSearch

from tests.test_run_checkpoint import (
    _init_git_workspace,
    _orchestrator,
    _patch_orchestrator,
)


MAINTAINER_BLOCK = {
    "type": "claude_code",
    "max_change_requests": 1,
    "calibration_timeout_seconds": 5,
}


class ScriptedMaintainerAgent:
    """One shared scripted agent for setup and change-request calls."""

    calls = []

    def __init__(self, workspace_action, output="done"):
        self.workspace_action = workspace_action
        self.output = output

    def initialize(self, workspace):
        self.workspace = Path(workspace)

    def generate_code(self, prompt):
        type(self).calls.append(prompt)
        self.workspace_action(self.workspace)
        return SimpleNamespace(success=True, output=self.output, error=None)

    def get_cumulative_cost(self):
        return 0.25

    def cleanup(self):
        pass


def write_entrypoint(root: Path) -> None:
    evaluation = root / "kapso_evaluation"
    evaluation.mkdir(exist_ok=True)
    (evaluation / "kapso_eval.py").write_text("ENTRYPOINT = True\n")


def manifest_stdout():
    payload = {
        "fidelity": "fast",
        "fraction": 0.03,
        "seed": 1337,
        "items": 5,
        "total_items": 100,
        "score": 0.4,
    }
    return (
        f"{maintainer_module.MANIFEST_MARKER} {json.dumps(payload)}\n"
    )


def patch_maintainer_environment(monkeypatch, agent):
    ScriptedMaintainerAgent.calls = []
    monkeypatch.setattr(
        orchestrator_module,
        "load_mode_config",
        lambda config_path, mode: {
            "search_strategy": {"type": "generic", "params": {}},
            "evaluation_maintainer": MAINTAINER_BLOCK,
        },
    )
    monkeypatch.setattr(
        maintainer_module.CodingAgentFactory,
        "create",
        classmethod(lambda cls, config: agent),
    )
    monkeypatch.setattr(
        maintainer_module,
        "subprocess",
        SimpleNamespace(
            run=lambda command, cwd, capture_output, text, timeout: (
                SimpleNamespace(
                    returncode=0, stdout=manifest_stdout(), stderr=""
                )
            )
        ),
    )


def test_setup_runs_once_and_is_budgeted(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)
    patch_maintainer_environment(
        monkeypatch, ScriptedMaintainerAgent(write_entrypoint)
    )

    orchestrator = _orchestrator(workspace)
    orchestrator.solve(experiment_max_iter=1)

    # Exactly one agent call: the setup transaction.
    assert len(ScriptedMaintainerAgent.calls) == 1
    components = orchestrator.budget_ledger.cost_by_component()
    assert components["evaluation_maintenance"] == 0.25
    registered = orchestrator.search_strategy.registered_evaluation
    assert "kapso_eval.py" in " ".join(registered["manifest"])
    assert "--fidelity full" in registered["command"]
    checkpoint = RunCheckpointStore(str(workspace)).load()
    assert checkpoint.cost_by_component["evaluation_maintenance"] == 0.25

    # Resume: the registry exists, so setup is skipped and consistency
    # validated; the strategy re-adopts the registered head.
    resumed = _orchestrator(workspace, resume=True)
    resumed.solve(experiment_max_iter=1)
    assert len(ScriptedMaintainerAgent.calls) == 1
    assert resumed.search_strategy.registered_evaluation["manifest"]


def test_registry_mismatch_fails_resume_loudly(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)
    patch_maintainer_environment(
        monkeypatch, ScriptedMaintainerAgent(write_entrypoint)
    )
    _orchestrator(workspace).solve(experiment_max_iter=1)

    (workspace / "kapso_evaluation" / "rogue.py").write_text("HACK = 1\n")

    resumed = _orchestrator(workspace, resume=True)
    with pytest.raises(
        EvaluationMaintainerError, match="does not match the registered"
    ):
        resumed.solve(experiment_max_iter=1)


def test_change_requests_route_to_the_maintainer_and_cap(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)
    patch_maintainer_environment(
        monkeypatch,
        ScriptedMaintainerAgent(
            write_entrypoint,
            output=(
                "<change_verdict>reject</change_verdict>"
                "<reason>evidence shows candidate error, not eval defect"
                "</reason>"
            ),
        ),
    )

    orchestrator = _orchestrator(workspace)
    orchestrator.search_strategy.next_agent_output = (
        "results...\n<evaluation_change_request>the harness crashes on the "
        "code suite\nTraceback: TimeoutError</evaluation_change_request>"
    )
    orchestrator.solve(experiment_max_iter=2)

    # setup + exactly one routed request (cap = 1; the second tag ignored).
    assert len(ScriptedMaintainerAgent.calls) == 2
    routed_prompt = ScriptedMaintainerAgent.calls[1]
    assert "the harness crashes on the code suite" in routed_prompt
    assert "Traceback: TimeoutError" in routed_prompt
    assert orchestrator._change_requests_filed == 1


def test_registered_integrity_enforced_in_agent_generated_mode(tmp_path):
    from contextlib import contextmanager

    candidate = tmp_path / "candidate"
    (candidate / "kapso_evaluation").mkdir(parents=True)
    (candidate / "kapso_evaluation" / "kapso_eval.py").write_text(
        "TAMPERED = True\n"
    )

    class FakeWorkspace:
        @contextmanager
        def materialize_ref(self, ref):
            yield str(candidate)

    strategy = GenericSearch.__new__(GenericSearch)
    strategy.workspace = FakeWorkspace()
    strategy.evaluation_provenance = AGENT_GENERATED
    strategy.registered_evaluation_manifest = {
        "kapso_eval.py": "0" * 64  # differs from the candidate's tree
    }
    strategy.registered_data_manifest = {}

    node = SearchNode(node_id=0, branch_name="generic_exp_0", score=0.9)
    assert strategy.enforce_evaluation_integrity(node) is False
    assert node.evaluation_valid is False
    assert node.score is None
    assert "integrity check failed" in node.evaluation_integrity_error


def test_evaluation_instructions_swap_with_registration():
    strategy = GenericSearch.__new__(GenericSearch)

    strategy.registered_evaluation_command = ""
    assert (
        "You MUST build and run evaluation"
        in strategy._evaluation_instructions()
    )

    strategy.registered_evaluation_command = (
        "python kapso_evaluation/kapso_eval.py --fidelity full "
        "--fraction 1.0 --seed 1337"
    )
    registered = strategy._evaluation_instructions()
    assert "read-and-execute only" in registered
    assert strategy.registered_evaluation_command in registered
    assert "<evaluation_change_request>" in registered


def test_registered_evaluation_syncs_into_sessions(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "kapso_evaluation").mkdir(parents=True)
    (workspace / "kapso_evaluation" / "kapso_eval.py").write_text("V2 = 2\n")
    session = tmp_path / "session"
    (session / "kapso_evaluation").mkdir(parents=True)
    (session / "kapso_evaluation" / "kapso_eval.py").write_text("V1 = 1\n")
    (session / "kapso_evaluation" / "stale.py").write_text("OLD = 1\n")

    strategy = GenericSearch.__new__(GenericSearch)
    strategy.workspace_dir = str(workspace)
    strategy._sync_registered_evaluation(str(session))

    synced = session / "kapso_evaluation"
    assert (synced / "kapso_eval.py").read_text() == "V2 = 2\n"
    assert not (synced / "stale.py").exists()


def test_fast_fraction_is_single_sourced_from_the_fidelity_block(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)
    patch_maintainer_environment(
        monkeypatch, ScriptedMaintainerAgent(write_entrypoint)
    )
    monkeypatch.setattr(
        orchestrator_module,
        "load_mode_config",
        lambda config_path, mode: {
            "search_strategy": {"type": "generic", "params": {}},
            "budget": {
                "fidelity": {"mode": "auto", "eval": {"fast_fraction": 0.2}}
            },
            "evaluation_maintainer": MAINTAINER_BLOCK,
        },
    )
    orchestrator = _orchestrator(workspace)
    assert orchestrator.evaluation_maintainer.fast_fraction == 0.2

    # The maintainer block has no fraction knob of its own.
    monkeypatch.setattr(
        orchestrator_module,
        "load_mode_config",
        lambda config_path, mode: {
            "search_strategy": {"type": "generic", "params": {}},
            "evaluation_maintainer": dict(
                MAINTAINER_BLOCK, fast_fraction=0.3
            ),
        },
    )
    with pytest.raises(ValueError, match="evaluation_maintainer config keys"):
        _orchestrator(str(tmp_path / "workspace_two"))


def test_registration_is_checkpointed_before_the_first_iteration(
    tmp_path, monkeypatch
):
    """A crash inside iteration 1 must not orphan the paid setup: the
    bootstrap checkpoint makes the campaign resumable from registration.
    """
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)
    patch_maintainer_environment(
        monkeypatch, ScriptedMaintainerAgent(write_entrypoint)
    )
    orchestrator = _orchestrator(workspace)
    monkeypatch.setattr(
        orchestrator.search_strategy,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("simulated crash inside iteration 1")
        ),
    )

    with pytest.raises(RuntimeError, match="simulated crash"):
        orchestrator.solve(experiment_max_iter=1)

    checkpoint = RunCheckpointStore(str(workspace)).load()
    assert checkpoint.status == "running"
    assert checkpoint.completed_iterations == 0
    # The paid registration itself is durable on disk alongside it.
    assert orchestrator.evaluation_maintainer.registry.exists()


def test_protected_data_is_registered_and_guarded_on_resume(
    tmp_path, monkeypatch
):
    """Registration captures the inputs half of evaluation identity, and a
    resume against tampered inputs fails loudly instead of silently
    scoring a different evaluation set.
    """
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    data_dir = workspace / "data"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text("PassengerId,y\n1,False\n")
    _patch_orchestrator(monkeypatch)
    patch_maintainer_environment(
        monkeypatch, ScriptedMaintainerAgent(write_entrypoint)
    )
    monkeypatch.setattr(
        orchestrator_module,
        "load_mode_config",
        lambda config_path, mode: {
            "search_strategy": {"type": "generic", "params": {}},
            "evaluation_maintainer": dict(
                MAINTAINER_BLOCK, protected_data_paths=["data"]
            ),
        },
    )

    orchestrator = _orchestrator(workspace)
    orchestrator.solve(experiment_max_iter=1)

    head = orchestrator.evaluation_maintainer.registry.head()
    assert set(head.data_manifest) == {"data/train.csv"}
    strategy = orchestrator.search_strategy
    assert strategy.registered_data_manifest == head.data_manifest

    # The live reward hack, replayed against resume: rig the inputs.
    (data_dir / "train.csv").write_text("PassengerId,y\n1,True\n")
    resumed = _orchestrator(workspace, resume=True)
    with pytest.raises(
        EvaluationMaintainerError, match="inputs do not match"
    ):
        resumed.solve(experiment_max_iter=1)
