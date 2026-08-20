# Codify-run driver tests (P7.3, CD§2): stage -> implement -> evaluate ->
# judge loop with the REAL LocalExecutor and real gates; fakes only at the
# session boundary (Rule 9: the regressions are a green verdict without a
# real reproduction, a judge veto ignored, staged-output leaks, and the
# feedback loop not feeding back).

from pathlib import Path

import pytest
import yaml

from kapso.learning.codify_run import CodifyRunDriver
from kapso.learning.trajectory_store import TrajectoryStore, save_trajectory
from tests.test_trajectory_store import TRAJECTORY_ID, build_work_dir

CARD = """---
type: procedure
title: Gated acceptance
representation: text
preconditions: "cpu-only; a metrics table with fold scores"
---

Accept a candidate only when its measured delta clears the gate.
"""

REQUEST = {
    "card": "gated-acceptance",
    "fixture": {
        "trajectory": TRAJECTORY_ID,
        "inputs": ["runs/run_0001/metrics.json"],
    },
    "materials": ["runs/run_0001/metrics.json"],
    "gates": {
        "decisions": {"gate_cleared": True},
        "metrics": {"delta": {"value": 0.002, "se": 0.0005}},
        "artifacts": {"decision": {"path": "outputs/decision.txt"}},
    },
}


def codify_config(tmp_path, max_iterations=2):
    return {
        "learning": {
            "trajectory_store": {"local": str(tmp_path / "store"), "remote": None},
            "codify": {
                "min_recurrence": 2, "max_iterations": max_iterations,
                "iteration_timeout_minutes": 1, "target": "local",
                "machine_type": "g2-standard-8", "replay_max_age_days": 60,
                "tolerance_z": 2,
                "implementor": {"cli": "codex", "model": "m", "effort": "xhigh"},
                "judge": {"cli": "claude_code", "model": "m",
                          "effort": "xhigh", "auth_mode": "oauth"},
            },
        }
    }


GOOD_EVAL = """import pathlib

import yaml

metrics = {"delta": 0.002}
decisions = {"gate_cleared": True}
assert abs(metrics["delta"] - 0.002) < 0.001  # asserts 0.002
assert decisions["gate_cleared"] is True
pathlib.Path("outputs").mkdir(exist_ok=True)
pathlib.Path("outputs/decision.txt").write_text("accept")
with open("outcome.yaml", "w") as handle:
    yaml.safe_dump({"decisions": decisions, "metrics": metrics}, handle)
"""


def implementor_writer(text=GOOD_EVAL):
    def write(workspace):
        workspace = Path(workspace)
        (workspace / "code").mkdir(exist_ok=True)
        (workspace / "code" / "gate.py").write_text("def gate():\n    return True\n")
        (workspace / "replay").mkdir(exist_ok=True)
        (workspace / "replay" / "eval.py").write_text(text)
    return write


class SessionFake:
    """Implementor calls run a writer; judge calls write a verdict."""

    def __init__(self, implementors, judges):
        self.implementors = list(implementors)
        self.judges = list(judges)
        self.prompts = []
        self.cwd = None

    def initialize(self, workspace):
        self.cwd = workspace

    def generate_code(self, prompt, debug_mode=False, timeout_seconds=None):
        self.prompts.append(prompt)

        class Result:
            success = True
            output = "done"

        if "You judge one codify run's CLAIMS" in prompt:
            verdict_path = next(
                token for token in prompt.replace(",", " ").split()
                if token.endswith(".yaml") and "judge-" in token
            )
            Path(verdict_path).write_text(self.judges.pop(0))
        else:
            self.implementors.pop(0)(self.cwd)
        return Result()


class SessionFactory:
    def __init__(self, agent):
        self.agent = agent

    def create(self, config):
        return self.agent


ENDORSE = "endorse: true\nfindings:\n  reproduction: green\nfeedback: ''\n"
REJECT = ("endorse: false\nfindings:\n  faithfulness: shortcut — the gate "
          "is hardcoded\nfeedback: >-\n  Implement the actual comparison.\n")


def make_driver(tmp_path, implementors, judges, max_iterations=2):
    config = codify_config(tmp_path, max_iterations)
    store = TrajectoryStore.from_config(config)
    work_dir, log = build_work_dir(tmp_path)
    save_trajectory(store, TRAJECTORY_ID, work_dir=str(work_dir),
                    campaign_log=str(log))
    agent = SessionFake(implementors, judges)
    return CodifyRunDriver(
        store, config, agent_factory=SessionFactory(agent)
    ), agent


def test_green_run_and_verdict(tmp_path):
    driver, agent = make_driver(tmp_path, [implementor_writer()], [ENDORSE])
    verdict = driver.run(REQUEST, CARD, str(tmp_path / "run"))
    assert verdict["status"] == "green" and verdict["iterations"] == 1
    assert verdict["mechanical_findings"] == []
    on_disk = yaml.safe_load((tmp_path / "run" / "verdict.yaml").read_text())
    assert on_disk["status"] == "green"
    # the evaluation really ran: the artifact gate's output exists
    assert (tmp_path / "run" / "workspace" / "outputs" / "decision.txt").is_file()


def test_judge_veto_feeds_back_then_exhausts(tmp_path):
    driver, agent = make_driver(
        tmp_path, [implementor_writer(), implementor_writer()],
        [REJECT, REJECT],
    )
    verdict = driver.run(REQUEST, CARD, str(tmp_path / "run"))
    assert verdict["status"] == "failed" and verdict["iterations"] == 2
    # the second implementor prompt carried the judge's feedback
    assert "Implement the actual comparison." in agent.prompts[2]


def test_mechanical_red_blocks_even_with_endorsement(tmp_path):
    # The evaluation reproduces the number (computed, in band) but never
    # ASSERTS the recorded value — green-but-weaker-than-expected_outcome.
    weak = """import pathlib

import yaml

metrics = {"delta": 21 / 10000}
decisions = {"gate_cleared": True}
assert metrics["delta"] > 0
assert decisions["gate_cleared"] is True
pathlib.Path("outputs").mkdir(exist_ok=True)
pathlib.Path("outputs/decision.txt").write_text("accept")
with open("outcome.yaml", "w") as handle:
    yaml.safe_dump({"decisions": decisions, "metrics": metrics}, handle)
"""
    driver, _ = make_driver(
        tmp_path, [implementor_writer(weak), implementor_writer(weak)],
        [ENDORSE, ENDORSE],
    )
    verdict = driver.run(REQUEST, CARD, str(tmp_path / "run"))
    assert verdict["status"] == "failed"
    assert any("never asserts the recorded value" in f
               for f in verdict["mechanical_findings"])


def test_staged_output_leak_refuses_to_run(tmp_path):
    leaky = dict(REQUEST)
    leaky["gates"] = dict(REQUEST["gates"])
    leaky["gates"]["artifacts"] = {
        "leak": {"path": "inputs/runs/run_0001/metrics.json"}
    }
    driver, _ = make_driver(tmp_path, [implementor_writer()], [ENDORSE])
    with pytest.raises(ValueError, match="leaked fixture outputs"):
        driver.run(leaky, CARD, str(tmp_path / "run"))


def test_request_notes_land_in_workspace(tmp_path):
    # Gate names underdetermine replay definitions; the request's notes
    # prose must reach the implementor (YAML comments die in the dump).
    request = dict(REQUEST)
    request["notes"] = "SE = clustered bootstrap, ddof 1."
    driver, _ = make_driver(tmp_path, [implementor_writer()], [ENDORSE])
    driver.run(request, CARD, str(tmp_path / "run"))
    notes = tmp_path / "run" / "workspace" / "replay-notes.md"
    assert notes.read_text() == "SE = clustered bootstrap, ddof 1."


def test_gate_tampering_is_restored_and_named(tmp_path):
    # An implementor that edits the contract files gets them restored and
    # a named mechanical finding — a forged schema must never reach the
    # judge or convert to green (seen live: dk-27 attempt 3).
    good_writer = implementor_writer()

    def tampering_implementor(workspace):
        good_writer(workspace)
        (Path(workspace) / "gates.yaml").write_text("decisions: {renamed: true}")

    driver, _ = make_driver(
        tmp_path, [tampering_implementor], [ENDORSE], max_iterations=1
    )
    verdict = driver.run(REQUEST, CARD, str(tmp_path / "run"))
    assert verdict["status"] == "failed"
    assert any("contract violation" in f for f in verdict["mechanical_findings"])
    gates_on_disk = (tmp_path / "run" / "workspace" / "gates.yaml").read_text()
    assert "renamed" not in gates_on_disk
    assert "gate_cleared" in gates_on_disk


def test_staging_preserves_run_relative_paths(tmp_path):
    # Two runs staging the same basename must land side by side — flat
    # staging silently overwrote one with the other.
    request = dict(REQUEST)
    request["fixture"] = dict(REQUEST["fixture"])
    request["fixture"]["inputs"] = [
        "runs/run_0001/metrics.json",
        "runs/run_0002/metrics.json",
    ]
    driver, _ = make_driver(tmp_path, [implementor_writer()], [ENDORSE])
    second = driver.store.resolve(TRAJECTORY_ID) / "runs" / "run_0002"
    second.mkdir(parents=True)
    (second / "metrics.json").write_text('{"delta": 0.9}')
    verdict = driver.run(request, CARD, str(tmp_path / "run"))
    assert verdict["status"] == "green"
    workspace = tmp_path / "run" / "workspace"
    assert (workspace / "inputs/runs/run_0001/metrics.json").is_file()
    assert (workspace / "inputs/runs/run_0002/metrics.json").is_file()
    inventory = (tmp_path / "run" / "staged-inventory.txt").read_text()
    assert "inputs/runs/run_0002/metrics.json" in inventory
