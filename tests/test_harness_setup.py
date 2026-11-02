"""The harness setup session: once per fresh campaign, at the campaign
root, when the harness has setup content; its domain statement and
prepared resources reach the task statement, the checkpoint and the
status file; a session that touches the pinned evaluation fails the build
naming the harness; a harness without setup content runs no session; a
resume after setup does not repeat it; a session that asks the person
pauses the campaign before its first experiment, and the reply lets the
next run finish setup."""

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, List, Optional

import pytest

import kapso.execution.orchestrator as orchestrator_module
from kapso.execution.coding_agents.base import CodingResult
from kapso.execution.inbox import file_requests, load_requests, record_reply
from kapso.handles import Budget
from kapso.harness import HarnessError
from kapso.kapso import Kapso
from tests.handles_support import git_workspace
from tests.harness_support import patch_engine

TRADING = Path(__file__).resolve().parent.parent / "examples" / "harnesses" / "trading"
ENTRY = {
    "key": "data:prices",
    "hit": "the task statement names a price history and kapso_datasets/ holds none",
    "tried": "looked under kapso_datasets/ and prepared/",
    "fix": "drop the bars as kapso_datasets/bars/<instrument>.csv",
    "next_steps": "build the snapshot under prepared/market-data, then write the statement",
}


class FakeSetupAgent:
    """What a setup session does, scripted: a domain statement, a prepared
    resource, and optionally a pinned-directory edit or an inbox request."""
    calls: List[dict] = []
    script: Callable[[Path, str], None] = staticmethod(lambda workspace, prompt: None)

    def __init__(self, config):
        self.config = config
        self.workspace: Optional[Path] = None

    def initialize(self, workspace: str) -> None:
        self.workspace = Path(workspace)

    def generate_code(self, prompt: str, debug_mode: bool = False, timeout_seconds=None) -> CodingResult:
        FakeSetupAgent.calls.append({"workspace": self.workspace, "prompt": prompt, "config": self.config})
        FakeSetupAgent.script(self.workspace, prompt)
        return CodingResult(success=True, output="prepared market-data")

    def get_cumulative_cost(self) -> float:
        return 0.25

    def cleanup(self) -> None:
        pass


def prepare_market_data(workspace: Path, prompt: str) -> None:
    statement = workspace / ".kapso" / "harness_setup" / "statement.md"
    statement.parent.mkdir(parents=True, exist_ok=True)
    statement.write_text("Instruments BTC and ETH, hourly bars, 5 bp costs; snapshot under prepared/market-data.\n")
    store = workspace / "prepared" / "market-data"
    store.mkdir(parents=True, exist_ok=True)
    (store / "MANIFEST.json").write_text(json.dumps({"instruments": ["BTC", "ETH"]}))


@pytest.fixture
def engine(monkeypatch, tmp_path):
    FakeSetupAgent.calls = []
    FakeSetupAgent.script = staticmethod(prepare_market_data)
    strategies = patch_engine(monkeypatch)
    # After patch_engine: the setup session runs as this scripted agent.
    monkeypatch.setattr(orchestrator_module.CodingAgentFactory, "create", staticmethod(lambda config: FakeSetupAgent(config)))
    harness_dir = tmp_path / "harness" / "trading"
    shutil.copytree(TRADING, harness_dir)
    workspace = tmp_path / "campaign"
    git_workspace(workspace)
    evaluation = workspace / "kapso_evaluation"
    evaluation.mkdir()
    (evaluation / "evaluate.py").write_text("print('sharpe: 1.0')\n")
    return workspace, harness_dir, strategies


def kapso_with(harness_dir: Path, tmp_path: Path) -> Kapso:
    kapso = Kapso()
    kapso._harness_settings["store"] = str(tmp_path / "store")
    kapso.load_harness(str(harness_dir))
    return kapso


def test_setup_fills_the_statement_and_is_recorded_everywhere(engine, tmp_path):
    workspace, harness_dir, strategies = engine
    solution = kapso_with(harness_dir, tmp_path).build("Raise the Sharpe above 1", output=str(workspace), budget=Budget(experiments=1))

    [call] = FakeSetupAgent.calls
    assert call["workspace"] == workspace
    prompt = call["prompt"]
    assert "# Harness setup" in prompt and f"{workspace / 'kapso_task.md'}" in prompt
    assert ".kapso/harness_setup/statement.md" in prompt and "`prepared/<name>/`" in prompt
    assert (harness_dir / "instructions" / "setup.md").read_text().strip() in prompt
    assert "market-data" in prompt   # the setup module's server and skill
    assert "request_from_user" not in prompt   # the inbox is off in this campaign
    agent_specific = call["config"].agent_specific
    assert agent_specific["timeout"] == 20 * 60 and "mcp__market-data" in agent_specific["allowed_tools"]
    assert "market-data" in agent_specific["mcp_servers"]

    task = (workspace / "kapso_task.md").read_text()
    assert "## Domain statement\n\nInstruments BTC and ETH, hourly bars, 5 bp costs" in task
    assert "## Prepared resources\n\n- `prepared/market-data`\n" in task
    assert "Raise the Sharpe above 1" in task
    checkpoint = json.loads((workspace / ".kapso" / "run_state.json").read_text())
    assert checkpoint["harness_setup"]["done"] is True and checkpoint["harness_setup"]["prepared"] == ["market-data"]
    assert checkpoint["harness_setup"]["cost_usd"] == 0.25
    status = json.loads((workspace / ".kapso" / "status.json").read_text())
    assert status["harness_setup"] == {"prepared": ["market-data"]}
    assert "harness_setup" in strategies[0].phases
    assert (workspace / ".kapso" / "sessions" / "harness_setup" / "session.json").is_file()
    assert solution.harness.name == "trading"


def test_a_session_that_touches_the_pinned_evaluation_fails_the_build(engine, tmp_path):
    workspace, harness_dir, _ = engine

    def tamper(workspace_dir: Path, prompt: str) -> None:
        prepare_market_data(workspace_dir, prompt)
        (workspace_dir / "kapso_evaluation" / "evaluate.py").write_text("print('sharpe: 9.9')\n")

    FakeSetupAgent.script = staticmethod(tamper)
    with pytest.raises(HarnessError, match="changed kapso_evaluation/, the campaign's pinned evaluation"):
        kapso_with(harness_dir, tmp_path).build("g", output=str(workspace), budget=Budget(experiments=1))


def test_a_harness_without_setup_content_runs_no_session(engine, tmp_path):
    workspace, harness_dir, _ = engine
    (harness_dir / "instructions" / "setup.md").unlink()
    manifest = harness_dir / "harness.yaml"
    text = manifest.read_text()
    start = text.index("  setup:\n")
    end = text.index("  ideation:\n")
    manifest.write_text(text[:start] + text[end:])
    # The market-data skill now has no module naming it on a page it is
    # listed on except implementation and deployment, which do.
    solution = kapso_with(harness_dir, tmp_path).build("g", output=str(workspace), budget=Budget(experiments=1))
    assert FakeSetupAgent.calls == []
    assert "## Prepared resources\n\n(none yet)" in (workspace / "kapso_task.md").read_text()
    assert json.loads((workspace / ".kapso" / "run_state.json").read_text())["harness_setup"] is None
    assert solution.harness.name == "trading"


def test_a_resume_after_setup_does_not_repeat_it(engine, tmp_path):
    workspace, harness_dir, _ = engine
    kapso_with(harness_dir, tmp_path).build("g", output=str(workspace), budget=Budget(experiments=1))
    assert len(FakeSetupAgent.calls) == 1
    solution = kapso_with(harness_dir, tmp_path).build(output=str(workspace), resume=True, budget=Budget(experiments=1))
    assert len(FakeSetupAgent.calls) == 1
    assert solution.experiments_total == 2
    assert "- `prepared/market-data`" in (workspace / "kapso_task.md").read_text()


def test_a_session_that_asks_pauses_the_campaign_and_the_reply_lets_it_finish(monkeypatch, tmp_path):
    FakeSetupAgent.calls = []
    inbox_path = tmp_path / "campaign" / ".kapso" / "inbox.jsonl"
    inbox = {"enabled": True, "path": str(inbox_path), "stop_grace_seconds": 5.0, "dotenv_path": ""}
    strategies = patch_engine(monkeypatch, inbox=inbox)
    monkeypatch.setattr(orchestrator_module.CodingAgentFactory, "create", staticmethod(lambda config: FakeSetupAgent(config)))
    harness_dir = tmp_path / "harness" / "trading"
    shutil.copytree(TRADING, harness_dir)
    workspace = tmp_path / "campaign"
    git_workspace(workspace)

    def ask_once(workspace_dir: Path, prompt: str) -> None:
        answered = [r for r in load_requests(inbox_path).values() if r.reply]
        if not answered:
            # The gate files under the id the session was given, which the
            # host also receives as its session id: Claude Code wants a UUID.
            session = FakeSetupAgent.calls[-1]["config"].agent_specific["session_id"]
            assert str(uuid.UUID(session)) == session
            file_requests(inbox_path, node=-1, session=session, entries=[ENTRY])
            return
        prepare_market_data(workspace_dir, prompt)

    FakeSetupAgent.script = staticmethod(ask_once)
    first = kapso_with(harness_dir, tmp_path).build("g", output=str(workspace), budget=Budget(experiments=1))
    assert first.stopped_reason == "waiting_for_user"
    assert [request["key"] for request in first.requests] == ["data:prices"]
    assert first.experiments == 0 and strategies[0].node_history == []
    prompt = FakeSetupAgent.calls[0]["prompt"]
    assert "request_from_user" in prompt and "the campaign pauses" in prompt
    assert "mcp__gated-knowledge__request_from_user" in FakeSetupAgent.calls[0]["config"].agent_specific["allowed_tools"]
    assert json.loads((workspace / ".kapso" / "run_state.json").read_text())["harness_setup"] is None

    [request] = load_requests(inbox_path).values()
    record_reply(inbox_path, request.id, "dropped under kapso_datasets/bars/")
    second = kapso_with(harness_dir, tmp_path).build(output=str(workspace), resume=True, budget=Budget(experiments=1))
    assert len(FakeSetupAgent.calls) == 2
    assert "dropped under kapso_datasets/bars/" in FakeSetupAgent.calls[1]["prompt"]
    assert second.stopped_reason != "waiting_for_user" and second.experiments == 1
    assert json.loads((workspace / ".kapso" / "run_state.json").read_text())["harness_setup"]["prepared"] == ["market-data"]


def test_a_reply_the_setup_session_never_got_can_be_retried(monkeypatch, tmp_path):
    """The setup session is no node, so nothing holds it suspended: until the
    checkpoint says setup is done, its answered requests still owe a resume."""
    FakeSetupAgent.calls = []
    inbox_path = tmp_path / "campaign" / ".kapso" / "inbox.jsonl"
    inbox = {"enabled": True, "path": str(inbox_path), "stop_grace_seconds": 5.0, "dotenv_path": ""}
    patch_engine(monkeypatch, inbox=inbox)
    monkeypatch.setattr(orchestrator_module.CodingAgentFactory, "create", staticmethod(lambda config: FakeSetupAgent(config)))
    harness_dir = tmp_path / "harness" / "trading"
    shutil.copytree(TRADING, harness_dir)
    workspace = tmp_path / "campaign"
    git_workspace(workspace)

    def ask_once(workspace_dir: Path, prompt: str) -> None:
        if not load_requests(inbox_path):
            file_requests(inbox_path, node=-1, session="setup", entries=[ENTRY])
            return
        prepare_market_data(workspace_dir, prompt)

    FakeSetupAgent.script = staticmethod(ask_once)
    kapso_with(harness_dir, tmp_path).build("g", output=str(workspace), budget=Budget(experiments=1))
    assert Kapso.awaiting_resume(str(workspace)) == []

    with monkeypatch.context() as resume_dies:
        def dies(self, **kwargs):
            raise RuntimeError("preflight: claude not on PATH")

        resume_dies.setattr(Kapso, "__init__", lambda self, config=None, bank=None: None)
        resume_dies.setattr(Kapso, "build", dies)
        with pytest.raises(RuntimeError, match="preflight"):
            Kapso.reply(str(workspace), 1, "dropped under kapso_datasets/bars/")
    assert [request.id for request in Kapso.awaiting_resume(str(workspace))] == [1]

    with monkeypatch.context() as resumes:
        resumes.setattr(Kapso, "__init__", lambda self, config=None, bank=None: None)
        resumes.setattr(Kapso, "build", lambda self, **kwargs: "resumed")
        assert Kapso.reply(str(workspace), 1) == "resumed"
    assert load_requests(inbox_path)[1].reply == "dropped under kapso_datasets/bars/"

    kapso_with(harness_dir, tmp_path).build(output=str(workspace), resume=True, budget=Budget(experiments=1))
    assert "dropped under kapso_datasets/bars/" in FakeSetupAgent.calls[1]["prompt"]
    assert json.loads((workspace / ".kapso" / "run_state.json").read_text())["harness_setup"]["prepared"] == ["market-data"]
    assert Kapso.awaiting_resume(str(workspace)) == []
    with pytest.raises(ValueError, match="already answered"):
        Kapso.reply(str(workspace), 1)
