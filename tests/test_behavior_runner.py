# Behavior-runner plumbing tests (P4): the runner stages real machinery and
# gates on the reviewer's {verdict, rationale} (Rule 9 — the regression is a
# scenario silently passing without review, or a naked verdict admitted).

from pathlib import Path

import pytest
import yaml

from kapso.learning.behavior import BehaviorRunner
from kapso.learning.trajectory_store import TrajectoryStore
from tests.test_bank_retriever import build_bank, card_text


class FakeReviewer:
    def __init__(self, verdict_text):
        self.verdict_text = verdict_text
        self.prompts = []
        self.cwd = None

    def initialize(self, workspace):
        self.cwd = workspace

    def generate_code(self, prompt, debug_mode=False, timeout_seconds=None):
        self.prompts.append(prompt)
        target = next(
            token for token in prompt.replace(",", " ").split()
            if token.endswith("verdict.yaml")
        )
        Path(target).write_text(self.verdict_text)

        class Result:
            success = True
            output = "done"

        return Result()


class ReviewerFactory:
    def __init__(self, reviewer):
        self.reviewer = reviewer

    def create(self, config):
        return self.reviewer


def behavior_config(tmp_path, scenarios_dir):
    return {
        "learning": {
            "trajectory_store": {"local": str(tmp_path / "store"), "remote": None},
            "retriever": {"k_insights": 2, "k_procedures": 1, "k_pitfalls": 1,
                          "unvisited_discount": 0.5},
            "behavior": {
                "scenarios_dir": str(scenarios_dir),
                "reviewer": {"cli": "claude_code", "model": "m",
                             "effort": "xhigh", "auth_mode": "oauth"},
                "timeout_minutes": 1,
                "run_root": str(tmp_path / "behave-runs"),
            },
        }
    }


def make_serve_scenario(tmp_path):
    """A B8-flavored fixture: a bank with one relevant and one irrelevant
    card; truth says the relevant card serves and the noise stays out."""
    scenario_dir = tmp_path / "scenarios" / "b8-serving"
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "scenario.yaml").write_text(yaml.safe_dump({
        "id": "b8-serving",
        "machinery": "serve",
        "description": "serving relevance on a known-truth bank",
        "task": {"family": "entity_binary_classification", "dataset": "rel-hm"},
    }))
    (scenario_dir / "truth.md").write_text(
        "relevant-card must serve; avito-card must not (scope excludes rel-hm).\n"
    )
    (scenario_dir / "rubric.md").write_text(
        "1. Is every served card semantically relevant?\n"
        "2. Did any eligible relevant card stay out?\n"
    )
    build_bank(scenario_dir / "fixture", {
        "relevant-card": card_text("relevant-card"),
        "avito-card": card_text("avito-card", scope=["dataset:rel-avito"]),
    })
    return scenario_dir


def test_serve_scenario_runs_real_machinery_and_gates(tmp_path):
    # Regression: the runner invokes the REAL push core, hands the reviewer
    # truth + rubric + artifacts, and returns the parsed verdict.
    scenario_dir = make_serve_scenario(tmp_path)
    config = behavior_config(tmp_path, scenario_dir.parent)
    reviewer = FakeReviewer("verdict: PASS\nrationale: served set matches truth.\n")
    runner = BehaviorRunner(
        TrajectoryStore.from_config(config), config,
        agent_factory=ReviewerFactory(reviewer),
    )
    result = runner.run_scenario(str(scenario_dir), config["learning"]["behavior"]["run_root"])
    assert result["verdict"] == "PASS"
    run_dir = Path(result["run_dir"])
    record = yaml.safe_load((run_dir / "artifacts" / "serving-record.yaml").read_text())
    served = [row["card"] for row in record["served"]]
    assert served == ["relevant-card"]  # the real eligibility law ran
    assert "truth.md" in reviewer.prompts[0] and "rubric.md" in reviewer.prompts[0]


def test_rollup_gates_on_any_fail(tmp_path):
    # Regression: gates dominate — one FAIL fails the suite.
    scenario_dir = make_serve_scenario(tmp_path)
    config = behavior_config(tmp_path, scenario_dir.parent)
    reviewer = FakeReviewer(
        "verdict: FAIL\nrationale: noise served — avito-card in the brief.\n"
    )
    runner = BehaviorRunner(
        TrajectoryStore.from_config(config), config,
        agent_factory=ReviewerFactory(reviewer),
    )
    rollup = runner.run_all(config["learning"]["behavior"]["run_root"])
    assert rollup["verdict"] == "FAIL"


def test_naked_verdict_rejected(tmp_path):
    # Regression: a verdict without a rationale is refused (no naked tags).
    scenario_dir = make_serve_scenario(tmp_path)
    config = behavior_config(tmp_path, scenario_dir.parent)
    reviewer = FakeReviewer("verdict: PASS\nrationale: ''\n")
    runner = BehaviorRunner(
        TrajectoryStore.from_config(config), config,
        agent_factory=ReviewerFactory(reviewer),
    )
    with pytest.raises(ValueError, match="no rationale"):
        runner.run_scenario(str(scenario_dir), config["learning"]["behavior"]["run_root"])
