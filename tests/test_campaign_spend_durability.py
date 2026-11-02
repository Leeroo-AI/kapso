"""What a campaign paid must stay on its ledger across a resume.

The harness runs the real orchestrator over the real GenericSearch with only
the paid sessions faked. Each fake session charges one Bill, the money that
actually left the account, so every test can compare the ledger with the bill
instead of with a figure worked out by hand.
"""

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import git
import pytest

import kapso.execution.orchestrator as orchestrator_module
from kapso.execution.inbox import file_requests, inbox_path, record_reply
from kapso.execution.orchestrator import OrchestratorAgent
from kapso.execution.run_checkpoint import RunCheckpointStore
from kapso.execution.search_strategies.generic.implementation import (
    SuspendedSession,
)
from kapso.execution.cost_estimator import normalize_cost_config
from kapso.execution.search_strategies.generic.strategy import GenericSearch

from harness_support import fake_config

IDEATION_USD = 0.1
IMPLEMENTATION_USD = 0.5
FEEDBACK_USD = 0.2
ROUND_USD = IDEATION_USD + IMPLEMENTATION_USD + FEEDBACK_USD

ENTRY = {
    "kind": "env",
    "key": "env:OPENAI_API_KEY",
    "hit": "the key is missing",
    "tried": "looked in the environment",
    "fix": "add the key",
    "next_steps": "run the evaluation",
}


class SimulatedKill(BaseException):
    """The process dying mid-round; not an Exception, so nothing absorbs it."""


class Bill:
    """The money the fake sessions charged, shared by every process slice."""

    def __init__(self) -> None:
        self.total = 0.0

    def charge(self, usd: float) -> Dict[str, float]:
        self.total += usd
        return {"cost_usd": usd, "duration_seconds": 1.0}


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
    """One workspace, its bill, and the fault to inject into its sessions."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self.workspace = tmp_path / "campaign"
        self.workspace.mkdir()
        self.repo = git.Repo.init(self.workspace)
        # Lanes run on threads; GitPython's object reader and the bill are not
        # safe to share between them.
        self.lane_lock = threading.Lock()
        with self.repo.config_writer() as config:
            config.set_value("user", "name", "Spend Test")
            config.set_value("user", "email", "spend@example.com")
        self.workspace.joinpath("README.md").write_text("# Test\n")
        self.repo.git.add(["README.md"])
        self.repo.git.commit("-m", "initial")
        self.repo.git.branch("-M", "main")
        self.bill = Bill()
        self.inbox = inbox_path(self.workspace)
        self.inbox_enabled = False
        # Called inside a session, once its charge is on the bill.
        self.during_implementation = lambda node_id: None
        self.during_feedback = lambda node_id: None
        self.solutions = ["solution"]
        self.ask_on_node: Optional[int] = None
        self.orchestrator: Optional[OrchestratorAgent] = None
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
            node_expansion_value=len(self.solutions),
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
            inbox_settings={
                "enabled": self.inbox_enabled,
                "path": str(self.inbox),
                "stop_grace_seconds": 1,
            },
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
        return list(self.solutions), [], self.bill.charge(IDEATION_USD)

    def _implement(self, **kwargs):
        node_id = kwargs["node_id"]
        branch_name = kwargs["branch_name"]
        with self.lane_lock:
            telemetry = self.bill.charge(IMPLEMENTATION_USD)
            if branch_name not in {head.name for head in self.repo.heads}:
                self.repo.create_head(branch_name)
        if kwargs.get("continuation") is None and node_id == self.ask_on_node:
            ((request_id, _),) = file_requests(
                self.inbox,
                node=node_id,
                session="sid",
                entries=[ENTRY],
            )
            suspended = SuspendedSession([request_id], "cli-1")
            return "partial", telemetry, None, suspended, None
        self.during_implementation(node_id)
        return "output", telemetry, None, None, None

    def _generate_feedback(self, node):
        node.phase_telemetry["feedback"] = self.bill.charge(FEEDBACK_USD)
        node.score = 0.5
        node.feedback = "fine"
        self.during_feedback(node.node_id)
        return node

    def open(self, *, resume: bool = False) -> OrchestratorAgent:
        self.orchestrator = OrchestratorAgent(
            FakeProblemHandler(),
            workspace_dir=str(self.workspace),
            resume=resume,
            knowledge_search=FakeKnowledgeSearch(),
            goal="the goal",
        )
        return self.orchestrator

    def checkpoint(self):
        return RunCheckpointStore(str(self.workspace)).load()


@pytest.fixture
def campaign(tmp_path, monkeypatch) -> Campaign:
    return Campaign(tmp_path, monkeypatch)


def test_a_continuation_after_a_resume_is_on_the_ledger(campaign):
    campaign.inbox_enabled = True
    campaign.ask_on_node = 0
    assert campaign.open().solve(experiment_max_iter=1).stopped_reason == (
        "waiting_for_user"
    )
    assert campaign.checkpoint().cumulative_cost == pytest.approx(
        IDEATION_USD + IMPLEMENTATION_USD
    )
    record_reply(campaign.inbox, 1, "added the key")

    result = campaign.open(resume=True).solve(experiment_max_iter=1)

    # The same node took a second session and its judge after the restart.
    assert result.total_cost == pytest.approx(campaign.bill.total)
    stored = campaign.checkpoint()
    assert stored.cumulative_cost == pytest.approx(campaign.bill.total)
    assert stored.cost_by_component["implementation"] == pytest.approx(
        2 * IMPLEMENTATION_USD
    )
    assert stored.cost_by_component["feedback"] == pytest.approx(FEEDBACK_USD)


def test_a_continuation_after_a_resume_trips_the_cost_budget(campaign):
    campaign.inbox_enabled = True
    campaign.ask_on_node = 0
    campaign.open().solve(experiment_max_iter=1)
    record_reply(campaign.inbox, 1, "added the key")
    budget = IDEATION_USD + 2 * IMPLEMENTATION_USD

    result = campaign.open(resume=True).solve(experiment_max_iter=3, cost_budget=budget)

    assert result.stopped_reason == "budget_exhausted"
    assert result.stop_detail == "cost_budget"
    assert result.iterations_run == 1
    assert campaign.bill.total == pytest.approx(budget + FEEDBACK_USD)


def test_restored_nodes_are_not_charged_twice(campaign):
    campaign.open().solve(experiment_max_iter=1)

    result = campaign.open(resume=True).solve(experiment_max_iter=1)

    assert campaign.bill.total == pytest.approx(2 * ROUND_USD)
    assert result.total_cost == pytest.approx(campaign.bill.total)


def _kill_second_round(campaign: Campaign, phase: str, *, after_heartbeat: bool):
    """Die in node 1's ``phase``. ``after_heartbeat`` models SIGKILL: the last
    write the process makes is a heartbeat, and no ``finally`` runs after it."""
    store = RunCheckpointStore(str(campaign.workspace))
    last_beat = {}

    def die(node_id: int) -> None:
        if node_id != 1:
            return
        if after_heartbeat:
            campaign.orchestrator._persist_budget_totals()
            last_beat["bytes"] = store.path.read_bytes()
        raise SimulatedKill()

    setattr(campaign, f"during_{phase}", die)
    with pytest.raises(SimulatedKill):
        campaign.open().solve(experiment_max_iter=5)
    if after_heartbeat:
        store.path.write_bytes(last_beat["bytes"])
    setattr(campaign, f"during_{phase}", lambda node_id: None)


@pytest.mark.parametrize("after_heartbeat", [True, False])
def test_a_round_killed_midway_keeps_what_it_paid(campaign, after_heartbeat):
    _kill_second_round(campaign, "feedback", after_heartbeat=after_heartbeat)

    # The boundary checkpoint knows one node; the disk knows two rounds' bills.
    stored = campaign.checkpoint()
    assert stored.completed_iterations == 1
    assert len(stored.strategy_state["node_history"]) == 1
    assert campaign.bill.total == pytest.approx(2 * ROUND_USD)
    assert stored.cumulative_cost == pytest.approx(campaign.bill.total)
    assert sum(stored.cost_by_component.values()) == pytest.approx(campaign.bill.total)

    # The redo rebuilds node 1 under the same id and is charged once more.
    result = campaign.open(resume=True).solve(experiment_max_iter=1)
    assert campaign.bill.total == pytest.approx(3 * ROUND_USD)
    assert result.total_cost == pytest.approx(campaign.bill.total)
    assert campaign.checkpoint().cumulative_cost == pytest.approx(campaign.bill.total)
    assert [
        node.node_id for node in campaign.orchestrator.search_strategy.node_history
    ] == [0, 1]


def test_a_kill_during_the_build_keeps_the_ideation_already_paid(campaign):
    _kill_second_round(campaign, "implementation", after_heartbeat=True)

    # The running session has not reported (it is the one being killed in a
    # real run), so the fake's charge for it is not expected on the ledger.
    assert campaign.checkpoint().cumulative_cost == pytest.approx(
        ROUND_USD + IDEATION_USD
    )


def test_a_killed_round_counts_against_the_cost_budget_on_resume(campaign):
    _kill_second_round(campaign, "feedback", after_heartbeat=True)
    paid = campaign.bill.total

    result = campaign.open(resume=True).solve(experiment_max_iter=5, cost_budget=paid)

    assert result.stopped_reason == "budget_exhausted"
    assert result.stop_detail == "cost_budget"
    assert result.iterations_run == 0
    assert campaign.bill.total == pytest.approx(paid)


def test_a_finished_lane_stays_paid_when_its_sibling_dies(campaign):
    campaign.solutions = ["one", "two"]
    charged: List[int] = []

    def die_in_second_lane(node_id: int) -> None:
        charged.append(node_id)
        if node_id == 1:
            raise SimulatedKill()

    campaign.during_implementation = die_in_second_lane
    campaign._generate_solution = lambda problem, parent: (
        list(campaign.solutions),
        [],
        {},
    )
    with pytest.raises(SimulatedKill):
        campaign.open().solve(experiment_max_iter=1)

    # Lane 0 returned its bill; lane 1 died before returning one.
    assert sorted(charged) == [0, 1]
    assert campaign.checkpoint().cost_by_component["implementation"] == pytest.approx(
        IMPLEMENTATION_USD
    )


class _StopAfterOneBeat:
    def __init__(self) -> None:
        self.waits = 0

    def wait(self, timeout: float) -> bool:
        self.waits += 1
        return self.waits > 1

    def set(self) -> None:
        pass


def test_the_heartbeat_thread_writes_the_running_rounds_spend(campaign):
    seen = {}

    def beat_once_then_die(node_id: int) -> None:
        if node_id != 1:
            return
        # The daemon loop itself: one beat, then its stop event ends it.
        orchestrator = campaign.orchestrator
        orchestrator._heartbeat_stop = _StopAfterOneBeat()
        orchestrator._run_checkpoint_heartbeat(0.0)
        seen["cost"] = campaign.checkpoint().cumulative_cost
        raise SimulatedKill()

    campaign.during_feedback = beat_once_then_die
    with pytest.raises(SimulatedKill):
        campaign.open().solve(experiment_max_iter=5)

    assert seen["cost"] == pytest.approx(2 * ROUND_USD)
