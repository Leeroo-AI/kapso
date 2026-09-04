"""The orchestrator and the inbox (design v4 §4.3, §10 scenarios 1, 17–20).

What must hold: a session that asked pauses the campaign — the result,
the checkpoint and the status file all say waiting_for_user and carry
the requests, no iteration is counted, the on_status hook saw it; a
resume with the request still open pauses again without running an
iteration; a resume after the reply runs the continuation and counts it
once; paused time is not campaign time; `kapso watch` renders the pause
after the process is gone.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import git
import pytest

from kapso.execution.inbox import file_requests, inbox_path, load_requests, record_reply
from kapso.execution.observability import OperationStatusView
from kapso.execution.orchestrator import OrchestratorAgent
from kapso.execution.run_checkpoint import RunCheckpointStore
from kapso.execution.search_strategies.base import SearchNode

ENTRY = {
    "key": "env:OPENAI_API_KEY",
    "hit": "openai.AuthenticationError at the embedding step",
    "tried": "OPENAI_KEY unset too; no .env in the repo",
    "fix": "add OPENAI_API_KEY=sk-... to /home/me/churn/.env",
    "next_steps": "embed the candidate texts, re-rank, evaluate",
}


def _init_git_workspace(path: Path) -> git.Repo:
    path.mkdir(parents=True)
    repo = git.Repo.init(path)
    with repo.config_writer() as config:
        config.set_value("user", "name", "Inbox Test")
        config.set_value("user", "email", "inbox@example.com")
    path.joinpath("README.md").write_text("# Test\n")
    repo.git.add(["README.md"])
    repo.git.commit("-m", "initial")
    repo.git.branch("-M", "main")
    return repo


class FakeLLM:
    def __init__(self, *args, **kwargs):
        pass

    def get_cumulative_cost(self) -> float:
        return 0.0

    def create_embedding(self, text, model=None):
        return []


class FakeProblemHandler:
    honor_agent_stop = True

    def get_problem_context(self) -> str:
        return "Solve the support problem"

    def deliverable_ready_reserve_seconds(self):
        return None


class FakeKnowledgeSearch:
    def close(self) -> None:
        pass


class FakeWorkspace:
    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir

    def get_cumulative_cost(self) -> float:
        return 0.0


class InboxFakeStrategy:
    """Asks the person on its first run; continues the suspended node on
    the next run once it is answered."""

    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self.workspace = FakeWorkspace(workspace_dir)
        self.node_history: List[SearchNode] = []
        self.runs = 0
        self.inbox = inbox_path(workspace_dir)
        self.scores_evaluator_id = ""
        self.evaluator_transition = None
        self.registered_evaluator_id = ""

    def run(self, context: str, budget_progress: float = 0.0) -> SearchNode:
        self.runs += 1
        waiting = [node for node in self.node_history if node.suspended]
        if waiting:
            node = waiting[0]
            node.suspended = False
            node.request_ids = []
            node.score = 0.4
            node.feedback = "continued and scored"
            return node
        node_id = len(self.node_history)
        branch_name = f"generic_exp_{node_id}"
        repo = git.Repo(self.workspace_dir)
        if branch_name not in {head.name for head in repo.heads}:
            repo.create_head(branch_name)
        (request_id, _), = file_requests(self.inbox, node=node_id, session="s", entries=[ENTRY])
        node = SearchNode(
            node_id=node_id, branch_name=branch_name,
            solution="# Core Idea\nre-rank candidates with text-embedding-3-large\n",
            suspended=True, request_ids=[request_id], cli_session_id="cli-1",
        )
        self.node_history.append(node)
        return node

    def waiting_requests(self):
        waiting = {node.node_id for node in self.node_history if node.suspended}
        if not waiting:
            return []
        return [r for r in load_requests(self.inbox).values() if r.open and r.node in waiting]

    def observe_budget(self, snapshot: Any) -> None:
        pass

    def observe_fidelity(self, decision: Any) -> None:
        pass

    def get_experiment_history(self, best_last: bool = False) -> List[SearchNode]:
        return self.node_history

    def get_best_experiment(self) -> Optional[SearchNode]:
        scored = [n for n in self.node_history if n.score is not None and not n.suspended]
        return scored[-1] if scored else None

    def get_deliverable_experiment(self) -> Optional[SearchNode]:
        return self.get_best_experiment()

    def dump_state(self) -> Dict[str, Any]:
        return {
            "node_history": [node.to_dict() for node in self.node_history],
            "scores_evaluator_id": self.scores_evaluator_id,
            "evaluator_transition": self.evaluator_transition,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.node_history = [SearchNode.from_dict(item) for item in state.get("node_history", [])]
        self.scores_evaluator_id = state.get("scores_evaluator_id", "")
        self.evaluator_transition = state.get("evaluator_transition")


@pytest.fixture
def patched(monkeypatch, tmp_path):
    import kapso.execution.orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module, "CliInference", FakeLLM)
    monkeypatch.setattr(
        orchestrator_module, "load_mode_config",
        lambda config_path, mode: {
            "search_strategy": {"type": "generic", "params": {}},
            "inbox": {"enabled": True, "stop_grace_seconds": 1, "registry": str(tmp_path / "registry.jsonl")},
        },
    )
    monkeypatch.setattr(OrchestratorAgent, "_create_feedback_generator", lambda self, coding_agent=None: object())
    strategies: List[InboxFakeStrategy] = []

    def create_strategy(self, coding_agent, workspace_dir, start_from_checkpoint):
        strategy = InboxFakeStrategy(workspace_dir)
        strategies.append(strategy)
        return strategy

    monkeypatch.setattr(OrchestratorAgent, "_create_search_strategy", create_strategy)
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    return workspace, strategies


def _orchestrator(workspace: Path, *, resume: bool = False) -> OrchestratorAgent:
    return OrchestratorAgent(
        FakeProblemHandler(), workspace_dir=str(workspace), resume=resume,
        knowledge_search=FakeKnowledgeSearch(), goal="Improve support",
    )


def test_a_session_that_asks_pauses_the_campaign(patched, capsys):
    workspace, strategies = patched
    seen: List[Dict[str, Any]] = []
    orchestrator = _orchestrator(workspace)
    assert orchestrator.strategy_params["inbox"]["enabled"] is True

    result = orchestrator.solve(experiment_max_iter=3, on_status=lambda status: seen.append(dict(status)))

    assert result.stopped_reason == "waiting_for_user"
    assert result.cumulative_iterations == 0 and result.iterations_run == 1
    assert [r["key"] for r in result.requests] == ["env:OPENAI_API_KEY"]
    assert result.requests[0]["for"] == "re-rank candidates with text-embedding-3-large"
    assert strategies[0].runs == 1

    checkpoint = RunCheckpointStore(str(workspace)).load()
    assert checkpoint.status == "running" and checkpoint.last_stop == "waiting_for_user"
    assert checkpoint.completed_iterations == 0
    assert checkpoint.strategy_state["node_history"][0]["suspended"] is True

    status = json.loads((workspace / ".kapso" / "status.json").read_text())
    assert status["state"] == "done" and status["stopped_reason"] == "waiting_for_user"
    assert status["requests"][0]["fix"] == ENTRY["fix"]
    done_payloads = [s for s in seen if s.get("state") == "done"]
    assert len(done_payloads) == 1 and done_payloads[0]["requests"][0]["id"] == 1

    out = capsys.readouterr().out
    assert "kapso evolve — waiting on you" in out
    assert "#1  env:OPENAI_API_KEY" in out and "reply with   kapso inbox reply" in out
    assert "tried  OPENAI_KEY unset too" in out


def test_resume_with_the_request_still_open_pauses_without_running(patched):
    workspace, strategies = patched
    _orchestrator(workspace).solve(experiment_max_iter=3)

    result = _orchestrator(workspace, resume=True).solve(experiment_max_iter=3)

    assert result.stopped_reason == "waiting_for_user"
    assert strategies[1].runs == 0
    assert result.requests[0]["id"] == 1
    assert RunCheckpointStore(str(workspace)).load().last_stop == "waiting_for_user"


def test_resume_after_the_reply_continues_and_counts_once(patched):
    workspace, strategies = patched
    _orchestrator(workspace).solve(experiment_max_iter=3)
    record_reply(inbox_path(workspace), 1, "added the key")

    result = _orchestrator(workspace, resume=True).solve(experiment_max_iter=1)

    assert result.stopped_reason == "max_iterations"
    assert result.cumulative_iterations == 1 and result.iterations_run == 1
    assert result.requests == []
    assert strategies[1].runs == 1
    node = strategies[1].node_history[0]
    assert node.suspended is False and node.score == 0.4
    checkpoint = RunCheckpointStore(str(workspace)).load()
    assert checkpoint.completed_iterations == 1 and checkpoint.last_stop is None
    assert checkpoint.strategy_state["node_history"][0]["suspended"] is False


def test_paused_time_is_not_campaign_time(patched):
    workspace, strategies = patched
    _orchestrator(workspace).solve(experiment_max_iter=3)
    saved = RunCheckpointStore(str(workspace)).load().elapsed_seconds
    time.sleep(0.3)
    resumed = _orchestrator(workspace, resume=True)
    assert resumed.get_elapsed_seconds() < saved + 0.1


def test_watch_renders_the_pause_after_the_process_is_gone(patched):
    workspace, strategies = patched
    _orchestrator(workspace).solve(experiment_max_iter=3)
    screen = OperationStatusView(workspace).explain()
    assert "WAITING ON YOU · 1 request" in screen
    assert "#1  env:OPENAI_API_KEY" in screen
    assert ENTRY["fix"] in screen
    assert "DONE" in screen
