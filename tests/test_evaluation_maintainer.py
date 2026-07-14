"""Hermetic tests for the EvaluationMaintainer frame (design doc, M5).

Every trust boundary under test is a mechanical post-condition: provided
bytes immutable, registration invariants, append-only registry, measured
(never hallucinated) timing.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import git
import pytest

import kapso.execution.evaluation_maintainer.maintainer as maintainer_module
from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.execution.evaluation_maintainer import (
    EvaluationChangeRequest,
    EvaluationMaintainer,
    EvaluationMaintainerError,
    EvaluationRegistry,
    EvaluationRegistryError,
    EvaluatorVersion,
    TimingModel,
)


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repo = git.Repo.init(workspace)
    with repo.config_writer() as config:
        config.set_value("user", "name", "Maintainer Test")
        config.set_value("user", "email", "maintainer@example.com")
    evaluation = workspace / "kapso_evaluation"
    evaluation.mkdir()
    (evaluation / "evaluate.py").write_text("SCORING = 'provided'\n")
    repo.git.add(["kapso_evaluation"])
    repo.git.commit("-m", "seed provided evaluation")
    repo.git.branch("-M", "main")
    return workspace


class ScriptedAgent:
    """Coding-agent stub that runs a scripted filesystem action."""

    def __init__(self, action, cost=0.25, output="done"):
        self.action = action
        self.cost = cost
        self.output = output

    def initialize(self, workspace):
        self.workspace = Path(workspace)

    def generate_code(self, prompt):
        self.action(self.workspace)
        return SimpleNamespace(success=True, output=self.output, error=None)

    def get_cumulative_cost(self):
        return self.cost

    def cleanup(self):
        pass


def manifest_stdout(items=10, total_items=200):
    payload = {
        "fidelity": "fast",
        "fraction": 0.05,
        "seed": 1337,
        "items": items,
        "total_items": total_items,
        "score": 0.5,
    }
    return f"eval output\n{maintainer_module.MANIFEST_MARKER} {json.dumps(payload)}\n"


def fake_runner(stdout=None, returncode=0):
    def run(command, cwd, capture_output, text, timeout):
        return SimpleNamespace(
            returncode=returncode,
            stdout=stdout if stdout is not None else manifest_stdout(),
            stderr="calibration failed" if returncode else "",
        )

    return run


def make_maintainer(workspace, runner=None):
    return EvaluationMaintainer(
        coding_agent_config=CodingAgentConfig(
            agent_type="stub",
            model="stub",
            debug_model="stub",
            agent_specific={},
        ),
        workspace_dir=str(workspace),
        fast_fraction=0.15,
        subsample_seed=1337,
        calibration_fraction=0.05,
        calibration_timeout_seconds=30.0,
        fast_variant_threshold_seconds=1.0,
        overhead_factor=1.25,
        command_runner=runner or fake_runner(),
    )


def patch_agent(monkeypatch, agent):
    monkeypatch.setattr(
        maintainer_module.CodingAgentFactory,
        "create",
        classmethod(lambda cls, config: agent),
    )


def write_wrapper(workspace: Path) -> None:
    (workspace / "kapso_evaluation" / "kapso_eval.py").write_text(
        "ENTRYPOINT = True\n"
    )


def test_setup_registers_v1_and_commits(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    patch_agent(monkeypatch, ScriptedAgent(write_wrapper))
    maintainer = make_maintainer(workspace)

    version = maintainer.setup(
        goal="improve accuracy", eval_dir=str(workspace), data_dir=None
    )

    assert version.version == 1
    assert version.provenance == "provided"
    assert version.fidelity_support["total_items"] == 200
    assert version.timing.per_item_seconds > 0
    head = EvaluationRegistry(str(workspace)).head()
    assert head is not None and head.evaluator_id == version.evaluator_id

    repo = git.Repo(workspace)
    assert "register evaluator v1" in repo.head.commit.message
    assert maintainer.last_transaction_telemetry.cost_usd == 0.25
    # The registered invocation contract uses CLI arguments, never env vars.
    command = maintainer.evaluation_command(fidelity="fast", fraction=0.15)
    assert "--fidelity fast" in command and "KAPSO_EVAL" not in command


def test_provided_bytes_are_immutable_as_a_post_condition(
    tmp_path, monkeypatch
):
    workspace = _workspace(tmp_path)

    def tamper(root: Path) -> None:
        (root / "kapso_evaluation" / "evaluate.py").write_text(
            "SCORING = 'weakened'\n"
        )

    patch_agent(monkeypatch, ScriptedAgent(tamper))
    maintainer = make_maintainer(workspace)

    with pytest.raises(EvaluationMaintainerError, match="modified"):
        maintainer.setup(
            goal="improve accuracy", eval_dir=str(workspace), data_dir=None
        )


def test_change_request_rejection_registers_nothing(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    patch_agent(monkeypatch, ScriptedAgent(write_wrapper))
    maintainer = make_maintainer(workspace)
    maintainer.setup(goal="g", eval_dir=str(workspace), data_dir=None)

    patch_agent(
        monkeypatch,
        ScriptedAgent(
            lambda root: None,
            output=(
                "<change_verdict>reject</change_verdict>"
                "<reason>lobbying from a losing candidate</reason>"
            ),
        ),
    )
    outcome = maintainer.handle_change_request(
        EvaluationChangeRequest(
            iteration=3,
            requested_by="implementation",
            summary="eval too strict",
            evidence="score was low",
        )
    )

    assert outcome.accepted is False
    assert outcome.requires_reanchor is False
    assert outcome.new_version is None
    assert len(EvaluationRegistry(str(workspace)).versions()) == 1


def test_accepted_change_registers_v2_with_reanchor(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    patch_agent(monkeypatch, ScriptedAgent(write_wrapper))
    maintainer = make_maintainer(workspace)
    v1 = maintainer.setup(goal="g", eval_dir=str(workspace), data_dir=None)

    def fix_wrapper(root: Path) -> None:
        (root / "kapso_evaluation" / "kapso_eval.py").write_text(
            "ENTRYPOINT = True\nTIMEOUT_FIXED = True\n"
        )

    patch_agent(
        monkeypatch,
        ScriptedAgent(
            fix_wrapper,
            output=(
                "<change_verdict>accept</change_verdict>"
                "<reason>wrapper timeout bug confirmed by traceback</reason>"
            ),
        ),
    )
    outcome = maintainer.handle_change_request(
        EvaluationChangeRequest(
            iteration=4,
            requested_by="implementation",
            summary="wrapper crashes on code suite",
            evidence="TimeoutError traceback",
        )
    )

    assert outcome.accepted is True
    assert outcome.requires_reanchor is True
    assert outcome.new_version.version == 2
    assert outcome.new_version.parent_evaluator == v1.evaluator_id
    versions = EvaluationRegistry(str(workspace)).versions()
    assert [v.version for v in versions] == [1, 2]
    assert versions[0].evaluator_id == v1.evaluator_id  # append-only


def test_accepted_change_with_identical_tree_fails_loud(
    tmp_path, monkeypatch
):
    workspace = _workspace(tmp_path)
    patch_agent(monkeypatch, ScriptedAgent(write_wrapper))
    maintainer = make_maintainer(workspace)
    maintainer.setup(goal="g", eval_dir=str(workspace), data_dir=None)

    patch_agent(
        monkeypatch,
        ScriptedAgent(
            lambda root: None,
            output=(
                "<change_verdict>accept</change_verdict>"
                "<reason>claims a fix but changed nothing</reason>"
            ),
        ),
    )
    with pytest.raises(EvaluationMaintainerError, match="byte-identical"):
        maintainer.handle_change_request(
            EvaluationChangeRequest(
                iteration=5,
                requested_by="feedback",
                summary="phantom fix",
                evidence="none",
            )
        )


def test_calibration_requires_the_manifest_line(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    patch_agent(monkeypatch, ScriptedAgent(write_wrapper))
    maintainer = make_maintainer(
        workspace, runner=fake_runner(stdout="no manifest here\n")
    )

    with pytest.raises(EvaluationMaintainerError, match="MANIFEST"):
        maintainer.setup(goal="g", eval_dir=str(workspace), data_dir=None)


def test_timing_estimates_and_record_run_tighten_upper(
    tmp_path, monkeypatch
):
    workspace = _workspace(tmp_path)
    patch_agent(monkeypatch, ScriptedAgent(write_wrapper))
    maintainer = make_maintainer(workspace)
    maintainer.setup(goal="g", eval_dir=str(workspace), data_dir=None)

    calibrated = maintainer.timing(1.0)
    assert calibrated.basis == "calibration"
    assert calibrated.upper_seconds == pytest.approx(
        calibrated.expected_seconds * 1.25
    )

    maintainer.record_run(fraction=1.0, duration_seconds=100.0)
    measured = maintainer.timing(1.0)
    assert measured.basis == "measured(n=1)"
    assert measured.expected_seconds == 100.0
    assert measured.upper_seconds == pytest.approx(125.0)


def test_registry_enforces_sequence_and_uniqueness(tmp_path):
    registry = EvaluationRegistry(str(tmp_path))
    timing = TimingModel(
        per_item_seconds=1.0, startup_seconds=0.0, total_items=10
    )

    def version(n, evaluator_id):
        return EvaluatorVersion(
            evaluator_id=evaluator_id,
            version=n,
            provenance="maintainer_built",
            parent_evaluator=None,
            fidelity_support={},
            timing=timing,
            created_at_iteration=0,
            reason="setup",
        )

    with pytest.raises(EvaluationRegistryError, match="must be 1"):
        registry.register(version(2, "id-a"))
    registry.register(version(1, "id-a"))
    with pytest.raises(EvaluationRegistryError, match="Expected version 2"):
        registry.register(version(3, "id-b"))
    with pytest.raises(EvaluationRegistryError, match="already registered"):
        registry.register(version(2, "id-a"))
    registry.register(version(2, "id-b"))
    assert [v.version for v in registry.versions()] == [1, 2]


def test_accepted_change_already_committed_by_the_agent(
    tmp_path, monkeypatch
):
    """The maintainer's own agent session may commit its edit itself; the
    frame's registration commit then has no staged delta and must be a
    no-op — not the crashed empty commit observed live after an accepted
    change request had already registered v2.
    """
    workspace = _workspace(tmp_path)
    patch_agent(monkeypatch, ScriptedAgent(write_wrapper))
    maintainer = make_maintainer(workspace)
    maintainer.setup(goal="g", eval_dir=str(workspace), data_dir=None)

    def edit_and_self_commit(root: Path) -> None:
        (root / "kapso_evaluation" / "kapso_eval.py").write_text(
            "ENTRYPOINT = True\nFIXED = True\n"
        )
        repo = git.Repo(root)
        repo.git.add(["kapso_evaluation"])
        repo.git.commit("-m", "agent's own commit of the fix")

    patch_agent(
        monkeypatch,
        ScriptedAgent(
            edit_and_self_commit,
            output=(
                "<change_verdict>accept</change_verdict>"
                "<reason>defect confirmed</reason>"
            ),
        ),
    )
    outcome = maintainer.handle_change_request(
        EvaluationChangeRequest(
            iteration=1,
            requested_by="implementation",
            summary="guard rejects every honest model",
            evidence="IncorrectSubmissionError trace",
        )
    )

    assert outcome.accepted is True
    assert len(EvaluationRegistry(str(workspace)).versions()) == 2
