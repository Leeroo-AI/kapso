"""A round's ideation is one session, however many lanes it feeds.

The harness runs the real orchestrator over the real GenericSearch with only
the paid sessions faked. Each fake session charges one bill, so the ledger is
compared with the money that actually left, not with a figure worked out by
hand.
"""

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import git
import pytest

import kapso.execution.orchestrator as orchestrator_module
from kapso.execution.observability import knowledge_summary
from kapso.execution.orchestrator import OrchestratorAgent
from kapso.execution.run_checkpoint import RunCheckpointStore
from kapso.execution.cost_estimator import normalize_cost_config
from kapso.execution.search_strategies.generic.strategy import GenericSearch

from harness_support import fake_config

LANES = 4
IDEATION_USD = 0.1
IMPLEMENTATION_USD = 0.5
FEEDBACK_USD = 0.2
ROUND_USD = IDEATION_USD + LANES * (IMPLEMENTATION_USD + FEEDBACK_USD)


class FakeLLM:
    def __init__(self, *args, **kwargs):
        pass

    def get_cumulative_cost(self) -> float:
        return 0.0


class FakeProblemHandler:
    honor_agent_stop = False
    maximize_scoring = True

    def get_problem_context(self) -> str:
        return "the problem"


class FakeKnowledgeSearch:
    def close(self) -> None:
        pass


class Campaign:
    """One workspace and the bill its fake sessions charge."""

    def __init__(self, workspace: Path, monkeypatch: pytest.MonkeyPatch):
        self.workspace = workspace
        self.billed = 0.0
        # Lanes run on threads; the bill and GitPython's reader are not safe
        # to share between them.
        self.lane_lock = threading.Lock()
        self.repo = git.Repo.init(workspace)
        with self.repo.config_writer() as config:
            config.set_value("user", "name", "Cost Test")
            config.set_value("user", "email", "cost@example.com")
        workspace.joinpath("README.md").write_text("# Test\n")
        self.repo.git.add(["README.md"])
        self.repo.git.commit("-m", "initial")
        self.repo.git.branch("-M", "main")
        monkeypatch.setattr(orchestrator_module, "CliInference", FakeLLM)
        monkeypatch.setattr(
            orchestrator_module,
            "load_config",
            lambda *args, **kwargs: fake_config(
                search_strategy={"type": "generic", "params": {}}
            ),
        )
        monkeypatch.setattr(
            OrchestratorAgent,
            "_create_feedback_generator",
            lambda agent: object(),
        )
        monkeypatch.setattr(
            OrchestratorAgent,
            "_create_search_strategy",
            lambda agent, workspace_dir, start_from_checkpoint: (
                self._strategy()
            ),
        )

    def _charge(self, usd: float) -> Dict[str, float]:
        with self.lane_lock:
            self.billed += usd
        return {"cost_usd": usd, "duration_seconds": 1.0}

    def _strategy(self) -> GenericSearch:
        strategy = GenericSearch.__new__(GenericSearch)
        strategy.__dict__.update(
            parent_policy="best",
            registered_evaluator_id="",
            registered_evaluation_manifest={},
            scores_evaluator_id="",
            evaluator_transition=None,
            fidelity_decision=None,
            problem_handler=FakeProblemHandler(),
            node_history=[],
            previous_errors=[],
            node_expansion_value=LANES,
            expansion_lane_env=None,
            iteration_count=0,
            workspace_dir=str(self.workspace),
            last_round_seconds=None,
            finishing_cut=None,
            harness=None,
            cost_config=normalize_cost_config(),
            workspace=SimpleNamespace(
                workspace_dir=str(self.workspace),
                get_cumulative_cost=lambda: 0.0,
                repo=self.repo,
            ),
            inbox_settings={"enabled": False},
        )
        strategy._generate_solution = self._generate_solution
        strategy._implement = self._implement
        strategy._get_code_diff = lambda branch_name, parent_branch: "diff"
        strategy._extract_agent_result = lambda output: {}
        strategy.enforce_evaluation_integrity = lambda node: True
        strategy._generate_feedback = self._generate_feedback
        strategy._record_evaluation_attempt = lambda node: None
        strategy._ensure_technical_difficulties = lambda node: None
        return strategy

    def _generate_solution(self, problem, parent_branch):
        telemetry = {**self._charge(IDEATION_USD), "repo_memory_calls": 3.0}
        return [f"solution {lane}" for lane in range(LANES)], [], telemetry

    def _implement(self, **kwargs):
        telemetry = self._charge(IMPLEMENTATION_USD)
        with self.lane_lock:
            self.repo.create_head(kwargs["branch_name"])
        return "output", telemetry, None, None, None

    def _generate_feedback(self, node):
        node.phase_telemetry["feedback"] = self._charge(FEEDBACK_USD)
        node.score = 0.5
        node.feedback = "fine"
        return node

    def open(self, *, resume: bool = False) -> OrchestratorAgent:
        return OrchestratorAgent(
            FakeProblemHandler(),
            workspace_dir=str(self.workspace),
            resume=resume,
            knowledge_search=FakeKnowledgeSearch(),
            goal="the goal",
        )

    def checkpoint(self):
        return RunCheckpointStore(str(self.workspace)).load()


@pytest.fixture
def campaign(tmp_path, monkeypatch) -> Campaign:
    workspace = tmp_path / "campaign"
    workspace.mkdir()
    return Campaign(workspace, monkeypatch)


def test_the_ledger_matches_the_bill_across_lanes(campaign):
    result = campaign.open().solve(experiment_max_iter=1)

    assert campaign.billed == pytest.approx(ROUND_USD)
    assert result.total_cost == pytest.approx(campaign.billed)
    stored = campaign.checkpoint()
    assert stored.cumulative_cost == pytest.approx(campaign.billed)
    assert stored.cost_by_component["ideation"] == pytest.approx(IDEATION_USD)


def test_the_ledger_matches_the_bill_after_a_resume(campaign):
    campaign.open().solve(experiment_max_iter=1)

    result = campaign.open(resume=True).solve(experiment_max_iter=1)

    assert campaign.billed == pytest.approx(2 * ROUND_USD)
    assert result.total_cost == pytest.approx(campaign.billed)
    assert campaign.checkpoint().cost_by_component["ideation"] == pytest.approx(
        2 * IDEATION_USD
    )


def test_the_cost_budget_sees_the_bill_not_a_multiple_of_ideation(campaign):
    # The first round fits with room to spare only if ideation is paid once.
    budget = ROUND_USD + (LANES - 1) * IDEATION_USD / 2

    result = campaign.open().solve(experiment_max_iter=2, cost_budget=budget)

    assert result.iterations_run == 2
    assert result.stop_detail == "cost_budget"
    assert campaign.billed == pytest.approx(2 * ROUND_USD)


def test_per_node_totals_add_up_to_the_campaign(campaign):
    orchestrator = campaign.open()
    orchestrator.solve(experiment_max_iter=1)
    nodes = orchestrator.search_strategy.get_experiment_history()

    assert len(nodes) == LANES
    assert sum(node.cost_usd for node in nodes) == pytest.approx(campaign.billed)
    # The campaign summary merges every node's phases the same way, so the
    # gate calls of the one ideation session are counted once as well.
    campaign_phases = {
        f"{node.node_id}:{phase}": values
        for node in nodes
        for phase, values in node.phase_telemetry.items()
    }
    assert knowledge_summary(campaign_phases) == "repo_memory 3"
