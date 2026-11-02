"""Every paid session reaches the ledger when it ends, whatever comes after.

The campaign tests run the real orchestrator over the real GenericSearch, the
real run_implementation and the real difficulties fallback, with only the agent
faked. Each fake session charges the shared Bill, so the ledger is compared with
what was actually charged. A process slice is a new orchestrator.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

import kapso.execution.coding_agents.factory as factory_module
import kapso.execution.evaluation_maintainer.maintainer as maintainer_module
import kapso.gated_mcp as gated_mcp_module
from kapso.core.roles import SessionRole
from kapso.execution.evaluation_maintainer import EvaluationMaintainerError
from kapso.execution.experiment_workspace.experiment_session import SessionPushError
from kapso.execution.inbox import file_requests, record_reply
from kapso.execution.memories.repo_memory import RepoMemoryManager
from kapso.execution.run_checkpoint import RunCheckpointStore
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.feedback_flow import generate_feedback
from kapso.execution.search_strategies.generic.implementation import (
    ContinuationFailed,
    run_implementation,
)

from test_campaign_spend_durability import (
    ENTRY,
    FEEDBACK_USD,
    IDEATION_USD,
    Campaign,
    SimulatedKill,
)
from test_ensemble_ideation import (
    CLAUDE_MEMBER,
    CODEX_MEMBER,
    SELECTOR,
    _plan,
    make_ensemble_strategy,
)
from implementation_support import MCP_TOOL, FakeWorkspace, no_staging
from tests.test_evaluation_maintainer_wiring import (
    ScriptedMaintainerAgent,
    manifest_stdout,
    patch_maintainer_environment,
    write_entrypoint,
)
from tests.test_run_checkpoint import (
    _init_git_workspace,
    _orchestrator,
    _patch_orchestrator,
)

IMPLEMENTATION_ROLE = SessionRole(
    name="implementation",
    cli="claude_code",
    model="the-model",
    effort=None,
    timeout=600,
    settings={"auth_mode": "oauth"},
    web=False,
)
BUILD_USD = 0.5
CONTINUATION_USD = 0.3
DIFFICULTIES_USD = 0.15


class RealBuildCampaign(Campaign):
    """Campaign whose build goes through the real run_implementation."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        super().__init__(tmp_path, monkeypatch)
        self.tmp_path = tmp_path
        self.resume_succeeds = True
        self.push_error: Optional[str] = None
        campaign = self

        class FakeAgent:
            def __init__(self, config):
                self.cost = 0.0
                # The difficulties fallback is the read-and-bash session.
                self.reconstructs = config.agent_specific.get("allowed_tools") == [
                    "Read",
                    "Bash",
                ]

            def initialize(self, workspace):
                pass

            def _result(self, success=True, **metadata):
                return SimpleNamespace(
                    success=success,
                    output="the build's report" if success else "",
                    error=None if success else "the CLI could not resume",
                    metadata={"tool_calls": [], **metadata},
                )

            def generate_code(self, prompt):
                if self.reconstructs:
                    self.cost = campaign.bill.charge(DIFFICULTIES_USD)["cost_usd"]
                    return self._result()
                self.cost = campaign.bill.charge(BUILD_USD)["cost_usd"]
                if campaign.ask_on_node is None:
                    return self._result()
                ((request_id, _),) = file_requests(
                    campaign.inbox, node=0, session="sid", entries=[ENTRY]
                )
                campaign.ask_on_node = None
                return self._result(
                    stopped_for_inbox=True,
                    inbox_request_ids=[request_id],
                    cli_session_id="cli-1",
                )

            def resume(self, cli_session_id, prompt):
                self.cost = campaign.bill.charge(CONTINUATION_USD)["cost_usd"]
                return self._result(success=campaign.resume_succeeds)

            def get_cumulative_cost(self):
                return self.cost

            def cleanup(self):
                pass

        monkeypatch.setattr(
            factory_module.CodingAgentFactory,
            "create",
            classmethod(lambda cls, config: FakeAgent(config)),
        )
        monkeypatch.setattr(
            gated_mcp_module, "get_mcp_config", lambda **kw: ({}, [MCP_TOOL])
        )
        monkeypatch.setattr(
            RepoMemoryManager,
            "ensure_exists_in_worktree",
            classmethod(lambda cls, folder: {}),
        )
        monkeypatch.setattr(
            RepoMemoryManager,
            "render_summary_and_toc",
            classmethod(lambda cls, doc, max_chars=2500: "memory"),
        )

    def _strategy(self):
        strategy = super()._strategy()
        strategy.__dict__.update(
            implementation_role=IMPLEMENTATION_ROLE,
            env_strip=[],
            stage_harness_for=no_staging,
            _finishing_deadline=lambda *args, **kwargs: (60, None),
            _session_stream_path=lambda branch: str(self.tmp_path / f"{branch}.jsonl"),
        )
        del strategy.__dict__["_ensure_technical_difficulties"]
        return strategy

    def _implement(self, **kwargs):
        branch_name = kwargs["branch_name"]
        if branch_name not in {head.name for head in self.repo.heads}:
            self.repo.create_head(branch_name)
        campaign = self

        class Workspace(FakeWorkspace):
            def create_experiment_session(
                self, branch_name, parent_branch_name, llm=None, continue_branch=False
            ):
                return super().create_experiment_session(
                    branch_name, parent_branch_name, llm
                )

            def finalize_session(self, session):
                if campaign.push_error:
                    raise SessionPushError(campaign.push_error)

        strategy = self.orchestrator.search_strategy
        continuation = kwargs.get("continuation")
        return run_implementation(
            stage_harness_for=no_staging,
            solution=kwargs["solution"],
            problem=kwargs["problem"],
            branch_name=branch_name,
            parent_branch_name=kwargs["parent_branch_name"],
            ideation_repo_memory_sections_consulted=None,
            lane_index=0,
            workspace=Workspace(str(self.tmp_path)),
            llm=None,
            registered_evaluation_manifest=None,
            sync_registered_evaluation=lambda folder: None,
            implementation_gates=["repo_memory"],
            gate_failure_policy="skip",
            implementation_role=IMPLEMENTATION_ROLE,
            implementation_fallback_model=None,
            env_strip=[],
            env_defaults={},
            lane_env=None,
            clamped_timeout=lambda seconds: seconds,
            session_stream_path=lambda branch: str(self.tmp_path / f"{branch}.jsonl"),
            build_prompt=lambda **kw: "the prompt",
            previous_errors_text="",
            lane_brief="",
            note_session_started=lambda: None,
            note_session_end_facts=lambda facts: None,
            note_session_spend=lambda telemetry: strategy._note_session_spend(
                kwargs["node_id"], telemetry, continued=continuation is not None
            ),
            await_registered_evaluation=lambda output: None,
            node_id=kwargs["node_id"],
            continuation=continuation,
        )


@pytest.fixture
def campaign(tmp_path, monkeypatch) -> RealBuildCampaign:
    return RealBuildCampaign(tmp_path, monkeypatch)


def _paused_on_the_inbox(campaign: RealBuildCampaign) -> None:
    campaign.inbox_enabled = True
    campaign.ask_on_node = 0
    assert campaign.open().solve(experiment_max_iter=1).stopped_reason == (
        "waiting_for_user"
    )
    record_reply(campaign.inbox, 1, "added the key")


def test_a_failed_continuation_stays_on_the_ledger(campaign):
    _paused_on_the_inbox(campaign)
    campaign.resume_succeeds = False

    with pytest.raises(ContinuationFailed):
        campaign.open(resume=True).solve(experiment_max_iter=1)

    assert campaign.bill.total == pytest.approx(
        IDEATION_USD + BUILD_USD + CONTINUATION_USD
    )
    assert campaign.checkpoint().cumulative_cost == pytest.approx(campaign.bill.total)

    # The retry is paid for too, and nothing is counted twice.
    campaign.resume_succeeds = True
    result = campaign.open(resume=True).solve(experiment_max_iter=1)
    assert campaign.bill.total == pytest.approx(
        IDEATION_USD
        + BUILD_USD
        + 2 * CONTINUATION_USD
        + DIFFICULTIES_USD
        + FEEDBACK_USD
    )
    assert result.total_cost == pytest.approx(campaign.bill.total)
    assert campaign.checkpoint().cumulative_cost == pytest.approx(campaign.bill.total)


def test_failed_continuations_count_against_the_cost_budget(campaign):
    _paused_on_the_inbox(campaign)
    campaign.resume_succeeds = False
    for _ in range(2):
        with pytest.raises(ContinuationFailed):
            campaign.open(resume=True).solve(experiment_max_iter=1)
    campaign.resume_succeeds = True
    paid = campaign.bill.total

    # The two failures alone spent the budget, so nothing more is bought.
    result = campaign.open(resume=True).solve(experiment_max_iter=5, cost_budget=paid)

    assert result.stopped_reason == "budget_exhausted"
    assert result.iterations_run == 0
    assert campaign.bill.total == pytest.approx(paid)


def test_a_continuation_whose_push_fails_stays_on_the_ledger(campaign):
    _paused_on_the_inbox(campaign)
    campaign.push_error = "origin rejected the push"

    with pytest.raises(SessionPushError):
        campaign.open(resume=True).solve(experiment_max_iter=1)

    assert campaign.bill.total == pytest.approx(
        IDEATION_USD + BUILD_USD + CONTINUATION_USD
    )
    assert campaign.checkpoint().cumulative_cost == pytest.approx(campaign.bill.total)


def test_a_continuation_counts_once(campaign):
    _paused_on_the_inbox(campaign)

    campaign.open(resume=True).solve(experiment_max_iter=1)

    node = campaign.orchestrator.search_strategy.node_history[0]
    assert node.phase_telemetry["implementation"]["cost_usd"] == pytest.approx(
        BUILD_USD + CONTINUATION_USD
    )
    assert campaign.checkpoint().cost_by_component["implementation"] == pytest.approx(
        BUILD_USD + CONTINUATION_USD
    )


def test_the_difficulties_fallback_is_on_the_ledger(campaign):
    result = campaign.open().solve(experiment_max_iter=1)

    assert campaign.bill.total == pytest.approx(
        IDEATION_USD + BUILD_USD + DIFFICULTIES_USD + FEEDBACK_USD
    )
    assert result.total_cost == pytest.approx(campaign.bill.total)
    stored = campaign.checkpoint()
    assert stored.cumulative_cost == pytest.approx(campaign.bill.total)
    assert stored.cost_by_component["technical_difficulties"] == pytest.approx(
        DIFFICULTIES_USD
    )
    node = campaign.orchestrator.search_strategy.node_history[0]
    assert node.cost_usd == pytest.approx(campaign.bill.total)


def test_the_difficulties_fallback_counts_against_the_cost_budget(campaign):
    budget = IDEATION_USD + BUILD_USD + FEEDBACK_USD + DIFFICULTIES_USD / 2

    result = campaign.open().solve(experiment_max_iter=3, cost_budget=budget)

    assert result.stop_detail == "cost_budget"
    assert result.iterations_run == 1


class FakeJudge:
    agent_type = "claude_code"

    def attach_harness(self, staged):
        pass

    def generate(self, **kwargs):
        return SimpleNamespace(
            unjudged=None,
            cards_load_bearing=[],
            feedback="fine",
            evaluation_valid=True,
            score=0.5,
            stop=False,
            cost_usd=FEEDBACK_USD,
            duration_seconds=1.0,
        )


def test_a_judge_whose_bookkeeping_raises_keeps_its_bill():
    """generate_feedback absorbs the error, so nothing else would notice."""

    def label_fails(record, valid):
        raise OSError("the run archive is gone")

    node = SearchNode(node_id=0, branch_name="generic_exp_0", workspace_dir="/tmp/none")
    generate_feedback(
        node,
        feedback_generator=FakeJudge(),
        goal="the goal",
        axes="",
        session_end_facts="",
        deadline_seconds=60,
        retry_deadline=lambda: (60, None),
        manifest_of_record=lambda n: {"score": 0.5},
        finalize_run_selection=label_fails,
        stage_harness_for=no_staging,
    )

    assert node.feedback == "Error generating feedback: the run archive is gone"
    assert node.phase_telemetry["feedback"]["cost_usd"] == FEEDBACK_USD


def _ensemble(tmp_path, monkeypatch):
    return make_ensemble_strategy(
        tmp_path,
        monkeypatch,
        ensemble=[dict(CODEX_MEMBER), dict(CLAUDE_MEMBER)],
        selector=dict(SELECTOR),
        claude_output=f"<solution>{_plan('claude')}</solution>",
        codex_output=f"<solution>{_plan('codex')}</solution>",
        selector_output=f"<solution_1>{_plan('picked')}</solution_1>",
    )


def test_ideation_bills_are_known_before_the_round_has_a_node(tmp_path, monkeypatch):
    strategy, _ = _ensemble(tmp_path, monkeypatch)

    def selector_dies(*args, **kwargs):
        raise SimulatedKill()

    strategy._select_from_candidates = selector_dies
    with pytest.raises(SimulatedKill):
        strategy._generate_solution("problem", "main")

    # The claude member's session ended. Codex reports no cost.
    assert strategy.ideation_bills == [1.0]


def test_ideation_bills_add_up_to_the_phase_the_node_takes_over(tmp_path, monkeypatch):
    strategy, _ = _ensemble(tmp_path, monkeypatch)

    _, _, telemetry = strategy._generate_solution("problem", "main")

    assert len(strategy.ideation_bills) >= 2  # a member and the selector
    assert sum(strategy.ideation_bills) == telemetry["cost_usd"]


def _ideation_noting_bills(campaign: Campaign, *, die: bool):
    def generate(problem, parent):
        strategy = campaign.orchestrator.search_strategy
        strategy.ideation_bills = []
        strategy._note_ideation_spend(campaign.bill.charge(IDEATION_USD)["cost_usd"])
        if die:
            raise RuntimeError("lens planner session failed: boom")
        return (
            list(campaign.solutions),
            [],
            {"cost_usd": IDEATION_USD, "duration_seconds": 1.0},
        )

    return generate


def test_ideation_that_raises_stays_on_the_ledger(tmp_path, monkeypatch):
    campaign = Campaign(tmp_path, monkeypatch)
    campaign._generate_solution = _ideation_noting_bills(campaign, die=True)

    with pytest.raises(RuntimeError, match="lens planner"):
        campaign.open().solve(experiment_max_iter=1)

    assert campaign.checkpoint().cumulative_cost == pytest.approx(IDEATION_USD)


def test_a_node_taking_over_the_ideation_bill_does_not_double_it(tmp_path, monkeypatch):
    campaign = Campaign(tmp_path, monkeypatch)
    campaign.solutions = ["one", "two"]
    campaign._generate_solution = _ideation_noting_bills(campaign, die=False)
    observed = []
    campaign.during_implementation = lambda node_id: observed.append(
        campaign.orchestrator._live_phase_costs().get("ideation")
    )

    result = campaign.open().solve(experiment_max_iter=1)

    # A heartbeat at any point of the round reads one ideation bill.
    assert observed == [pytest.approx(IDEATION_USD)] * 2
    assert result.total_cost == pytest.approx(campaign.bill.total)


REJECTED = "<change_verdict>reject</change_verdict><reason>candidate error</reason>"
ACCEPTED = "<change_verdict>accept</change_verdict><reason>real defect</reason>"
CHANGE_REQUEST = (
    "<evaluation_change_request>the harness crashes</evaluation_change_request>"
)
MAINTAINER_SESSION_USD = 0.25


def _edit_on_the_change_request(root: Path) -> None:
    write_entrypoint(root)
    if len(ScriptedMaintainerAgent.calls) > 1:
        (root / "kapso_evaluation" / "kapso_eval.py").write_text("V2 = True\n")


def _calibration_failing_after_setup(monkeypatch, failure) -> None:
    runs = {"count": 0}

    def run(command, cwd, capture_output, text, errors, timeout):
        runs["count"] += 1
        if runs["count"] > 1:
            return failure()
        return SimpleNamespace(returncode=0, stdout=manifest_stdout(), stderr="")

    monkeypatch.setattr(maintainer_module, "subprocess", SimpleNamespace(run=run))


@pytest.fixture
def maintained_workspace(tmp_path, monkeypatch) -> Path:
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)
    patch_maintainer_environment(
        monkeypatch,
        ScriptedMaintainerAgent(_edit_on_the_change_request, output=ACCEPTED),
    )
    return workspace


def _maintainer_spend_on_disk(workspace: Path) -> float:
    stored = RunCheckpointStore(str(workspace)).load()
    return stored.cost_by_component["evaluation_maintenance"]


def test_a_change_request_that_fails_calibration_keeps_its_bill(
    maintained_workspace, monkeypatch
):
    _calibration_failing_after_setup(
        monkeypatch,
        lambda: SimpleNamespace(returncode=1, stdout="", stderr="it crashed"),
    )
    orchestrator = _orchestrator(maintained_workspace)
    orchestrator.search_strategy.next_agent_output = CHANGE_REQUEST

    with pytest.raises(EvaluationMaintainerError, match="Calibration run failed"):
        orchestrator.solve(experiment_max_iter=1)

    assert _maintainer_spend_on_disk(maintained_workspace) == pytest.approx(
        2 * MAINTAINER_SESSION_USD
    )


def test_a_kill_during_calibration_keeps_the_session_that_came_before(
    maintained_workspace, monkeypatch
):
    """Calibration can run for many minutes: a heartbeat lands during it."""
    store = RunCheckpointStore(str(maintained_workspace))
    last_beat = {}

    def die():
        orchestrator._persist_budget_totals()
        last_beat["bytes"] = store.path.read_bytes()
        raise SimulatedKill()

    _calibration_failing_after_setup(monkeypatch, die)
    orchestrator = _orchestrator(maintained_workspace)
    orchestrator.search_strategy.next_agent_output = CHANGE_REQUEST
    with pytest.raises(SimulatedKill):
        orchestrator.solve(experiment_max_iter=1)
    store.path.write_bytes(last_beat["bytes"])

    assert _maintainer_spend_on_disk(maintained_workspace) == pytest.approx(
        2 * MAINTAINER_SESSION_USD
    )


def test_a_rejected_change_request_is_billed_once(maintained_workspace, monkeypatch):
    patch_maintainer_environment(
        monkeypatch, ScriptedMaintainerAgent(write_entrypoint, output=REJECTED)
    )
    orchestrator = _orchestrator(maintained_workspace)
    orchestrator.search_strategy.next_agent_output = CHANGE_REQUEST

    orchestrator.solve(experiment_max_iter=1)

    assert _maintainer_spend_on_disk(maintained_workspace) == pytest.approx(
        2 * MAINTAINER_SESSION_USD
    )
