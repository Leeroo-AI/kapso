"""The generic strategy and the inbox (design v4 §4.3–4.4).

What must hold: a session that asked the person suspends its node — no
judge, no difficulties reconstruction, never a parent; an iteration with
that request still open raises instead of running; once every request is
answered the next run() continues the same node in place — no new
ideation, no iteration counted, the stored session resumed with a
follow-up that carries the reply — and the judge sees it once; asking
again suspends again; the node's inbox fields round-trip the checkpoint
and are validated; parallel lanes turn the inbox off.
"""

from types import SimpleNamespace

import pytest

from kapso.execution.inbox import (
    InboxOpenError,
    file_requests,
    inbox_path,
    load_requests,
    record_reply,
    write_launch_record,
)
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.implementation import (
    Continuation,
    SuspendedSession,
)
from kapso.execution.search_strategies.generic.strategy import (
    GenericSearch,
    resolve_inbox_settings,
)

ENTRY = {
    "key": "env:OPENAI_API_KEY",
    "hit": "openai.AuthenticationError at the embedding step",
    "tried": "OPENAI_KEY unset too; no .env in the repo",
    "fix": "add OPENAI_API_KEY=sk-... to .env",
    "next_steps": "embed the candidate texts, re-rank, run the evaluation",
}


class Harness:
    """A GenericSearch built through __new__ (the suite's convention) with
    an `_implement` that asks on its first call and continues on its
    second, and counters for what must or must not run."""

    def __init__(self, tmp_path):
        self.inbox = inbox_path(tmp_path)
        self.calls = {"ideation": 0, "feedback": 0, "difficulties": 0}
        self.implement_kwargs = []
        self.ask_again_on_continue = False

        strategy = GenericSearch.__new__(GenericSearch)
        strategy.parent_policy = "best"
        strategy.registered_evaluator_id = ""
        strategy.fidelity_decision = None
        strategy.problem_handler = SimpleNamespace(maximize_scoring=True)
        strategy.node_history = []
        strategy.node_expansion_value = 1
        strategy.expansion_lane_env = None
        strategy.iteration_count = 0
        strategy.workspace_dir = str(tmp_path)
        strategy.inbox_settings = {
            "enabled": True, "path": str(self.inbox), "stop_grace_seconds": 1,
        }
        strategy._generate_solution = self._generate_solution
        strategy._implement = self._implement
        strategy._get_code_diff = lambda branch_name, parent_branch: "diff"
        strategy._extract_agent_result = lambda output: {}
        strategy.enforce_evaluation_integrity = lambda node: True
        strategy._generate_feedback = self._generate_feedback
        strategy._record_evaluation_attempt = lambda node: None
        strategy._ensure_technical_difficulties = self._ensure_difficulties
        self.strategy = strategy

    def _generate_solution(self, problem, parent_branch):
        self.calls["ideation"] += 1
        return ["solution"], [], {"cost_usd": 0.1, "duration_seconds": 1.0}

    def _implement(self, **kwargs):
        self.implement_kwargs.append(kwargs)
        telemetry = {"cost_usd": 0.5, "duration_seconds": 2.0}
        continuation = kwargs.get("continuation")
        if continuation is None or self.ask_again_on_continue:
            (request_id, _), = file_requests(
                self.inbox, node=kwargs["node_id"], session="inbox-sid",
                entries=[ENTRY if continuation is None else {**ENTRY, "key": "data/x.csv"}],
            )
            return "partial work", telemetry, None, SuspendedSession([request_id], "cli-1")
        return "finished output", telemetry, None, None

    def _generate_feedback(self, node):
        self.calls["feedback"] += 1
        node.score = 0.5
        node.feedback = "fine"
        return node

    def _ensure_difficulties(self, node):
        self.calls["difficulties"] += 1


def test_a_session_that_asks_suspends_the_node(tmp_path):
    harness = Harness(tmp_path)
    strategy = harness.strategy
    node = strategy.run("problem")
    assert node.suspended is True
    assert node.request_ids == [1]
    assert node.cli_session_id == "cli-1"
    assert node.score is None and node.feedback == ""
    assert harness.calls == {"ideation": 1, "feedback": 0, "difficulties": 0}
    assert strategy.node_history == [node]
    assert strategy.iteration_count == 1
    assert harness.implement_kwargs[0]["node_id"] == 0
    assert strategy.get_best_experiment() is None
    # Never a parent while it waits, even though it has a committed diff.
    assert strategy._select_parent().branch_name == "main"


def test_an_open_request_stops_iteration(tmp_path):
    harness = Harness(tmp_path)
    strategy = harness.strategy
    strategy.run("problem")
    with pytest.raises(InboxOpenError, match="node 0 waits"):
        strategy.run("problem")
    assert harness.calls["ideation"] == 1


def test_an_answered_request_continues_the_same_node(tmp_path):
    harness = Harness(tmp_path)
    strategy = harness.strategy
    first = strategy.run("problem")
    record_reply(harness.inbox, 1, "added the key")

    node = strategy.run("problem")

    assert node is first
    assert node.suspended is False and node.request_ids == []
    assert node.score == 0.5
    assert harness.calls == {"ideation": 1, "feedback": 1, "difficulties": 1}
    assert strategy.iteration_count == 1
    assert strategy.node_history == [node]
    continuation = harness.implement_kwargs[1]["continuation"]
    assert isinstance(continuation, Continuation)
    assert continuation.cli_session_id == "cli-1"
    assert "Request #1 — env:OPENAI_API_KEY" in continuation.follow_up
    assert 'their reply: "added the key"' in continuation.follow_up
    assert ENTRY["tried"] in continuation.follow_up
    assert ENTRY["next_steps"] in continuation.follow_up
    assert harness.implement_kwargs[1]["branch_name"] == first.branch_name
    assert load_requests(harness.inbox)[1].state == "continued"
    assert node.phase_telemetry["implementation"]["cost_usd"] == pytest.approx(1.0)
    assert node.duration_seconds is not None and node.cost_usd == pytest.approx(1.1)


def test_asking_again_after_a_reply_suspends_again(tmp_path):
    harness = Harness(tmp_path)
    harness.ask_again_on_continue = True
    strategy = harness.strategy
    strategy.run("problem")
    record_reply(harness.inbox, 1, "")
    node = strategy.run("problem")
    assert node.suspended is True and node.request_ids == [2]
    assert harness.calls["feedback"] == 0
    assert load_requests(harness.inbox)[2].key == "data/x.csv"


def test_inbox_off_never_looks_at_the_file(tmp_path):
    harness = Harness(tmp_path)
    strategy = harness.strategy
    strategy.inbox_settings = {"enabled": False}
    strategy._implement = lambda **kwargs: ("out", {"cost_usd": 0.0, "duration_seconds": 0.0}, None)
    node = strategy.run("problem")
    assert node.suspended is False
    assert harness.calls["feedback"] == 1


def test_node_inbox_fields_round_trip_and_validate():
    node = SearchNode(node_id=3, suspended=True, request_ids=[1, 2], cli_session_id="cli-1")
    restored = SearchNode.from_dict(node.to_dict())
    assert (restored.suspended, restored.request_ids, restored.cli_session_id) == (True, [1, 2], "cli-1")
    with pytest.raises(ValueError, match="request_ids"):
        SearchNode.from_dict({"node_id": 3, "request_ids": ["a"]})
    with pytest.raises(ValueError, match="suspended must be a boolean"):
        SearchNode.from_dict({"node_id": 3, "suspended": "yes"})
    with pytest.raises(ValueError, match="cli_session_id"):
        SearchNode.from_dict({"node_id": 3, "cli_session_id": 7})


def test_resolve_inbox_settings_mounts_the_gate_or_turns_off_for_lanes(capsys):
    gates = ["research", "repo_memory"]
    settings = {"enabled": True, "path": "/x/inbox.jsonl", "stop_grace_seconds": 120}
    resolved = {**settings, "dotenv_path": ""}
    assert resolve_inbox_settings(settings, 1, gates) == resolved
    assert gates == ["research", "repo_memory", "inbox"]
    assert resolve_inbox_settings(settings, 1, gates) == resolved
    assert gates.count("inbox") == 1

    gates = ["research"]
    resolved = resolve_inbox_settings(settings, 2, gates)
    assert resolved["enabled"] is False and "inbox" not in gates
    assert "node expansion above 1" in capsys.readouterr().out

    off = {"enabled": False}
    assert resolve_inbox_settings(off, 1, gates) == off and "inbox" not in gates
    assert resolve_inbox_settings(None, 1, gates) is None


def test_resolve_inbox_settings_derives_an_absolute_path_and_the_dotenv(tmp_path, monkeypatch):
    """A relative campaign dir (the CLI's --output campaign) still yields
    an absolute inbox path, and the launch record's dotenv path rides
    along for the prompt (empty without a record)."""
    monkeypatch.chdir(tmp_path)
    settings = {"enabled": True, "stop_grace_seconds": 120}
    resolved = resolve_inbox_settings(settings, 1, [], "campaign")
    assert resolved["path"] == str(tmp_path.resolve() / "campaign" / ".kapso" / "inbox.jsonl")
    assert resolved["dotenv_path"] == ""
    assert settings == {"enabled": True, "stop_grace_seconds": 120}
    write_launch_record(tmp_path / "campaign", {"dotenv_path": str(tmp_path / ".env")})
    assert resolve_inbox_settings(settings, 1, [], "campaign")["dotenv_path"] == str(tmp_path / ".env")
