"""Staging a module into a session: the tree each host discovers, the
hooks merged in manifest order with the plugin root expanded, the servers
beside the gates with the environment forwarded by name on codex, the
delegates carrying the module's guidance, the record and the stream path;
idempotent, two hosts in one directory, a tracked collision refused, a
changed harness refused."""

import json
import shutil
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from kapso.core.config import load_config
from kapso.execution.experiment_workspace.experiment_workspace import ExperimentWorkspace
from kapso.execution.coding_agents.adapters.claude_code_agent import ClaudeCodeCodingAgent
from kapso.execution.coding_agents.adapters.codex_agent import CodexCodingAgent
from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.execution.coding_agents.session_record import EXIT_ERROR, append_run, run_entry
from kapso.execution.coding_agents.transcripts import digest_claude
from kapso.harness import (
    GUIDANCE_HEADING,
    HARNESS_ROOT_VARIABLE,
    HarnessError,
    load_harness,
    stage_harness,
)
import kapso.execution.search_strategies.generic.feedback_generator.feedback_generator as feedback_module
from kapso.execution.search_strategies.generic.feedback_generator.feedback_generator import FeedbackGenerator
from kapso.learning.trajectory_store import STAGING_DIRECTORIES
from tests.handles_support import git, git_workspace
from tests.harness_support import NoopAgent

TRADING = Path(__file__).resolve().parent.parent / "examples" / "harnesses" / "trading"


@pytest.fixture
def harness(tmp_path):
    copy = tmp_path / "h" / "trading"
    shutil.copytree(TRADING, copy)
    settings = dict(load_config()["harness"], store=str(tmp_path / "store"))
    return load_harness(str(copy), settings=settings, say=lambda line: None)


def campaign_with(tmp_path: Path, *, statement: bool = False, prepared: bool = False) -> Path:
    campaign = tmp_path / "campaign"
    campaign.mkdir(exist_ok=True)
    if statement:
        (campaign / "kapso_task.md").write_text("# Task\n\nRaise the Sharpe.\n")
    if prepared:
        (campaign / "prepared" / "market-data").mkdir(parents=True)
        (campaign / "prepared" / "market-data" / "MANIFEST.json").write_text("{}")
    return campaign


def stage(harness, module, host, tmp_path, *, session="sessions/generic_exp_0", session_id="generic_exp_0", **campaign_options):
    campaign = campaign_with(tmp_path, **campaign_options)
    session_dir = campaign / session
    session_dir.mkdir(parents=True, exist_ok=True)
    return stage_harness(harness, module, session_dir=session_dir, campaign_dir=campaign, session_id=session_id, host=host)


def tree(root: Path):
    return sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*")
        if (p.is_file() or p.is_symlink()) and ".git" not in p.parts
    )


def test_the_implementation_module_on_claude_code(harness, tmp_path):
    staged = stage(harness, "implementation", "claude_code", tmp_path)
    session_dir = staged.session_dir
    assert tree(session_dir) == [
        ".claude/harness-agents.json",
        ".claude/harness-hooks.json",
        ".claude/skills/backtesting/SKILL.md",
        ".claude/skills/lookahead-scan/SKILL.md",
        ".claude/skills/lookahead-scan/scripts/rules.py",
        ".claude/skills/lookahead-scan/scripts/scan.py",
        ".claude/skills/market-data/SKILL.md",
        ".claude/skills/risk-limits/SKILL.md",
    ]
    hooks_text = (session_dir / ".claude" / "harness-hooks.json").read_text()
    hooks = json.loads(hooks_text)["hooks"]
    assert "${" not in hooks_text and str(harness.path) in hooks_text
    assert list(hooks) == ["SessionStart", "PreToolUse", "PostToolUse", "Stop"]
    # The engine's protect-harness hook first, then the module's.
    assert [entry.get("matcher") for entry in hooks["PreToolUse"]] == ["Edit|Write|MultiEdit", "Bash", "WebSearch|WebFetch", "Edit|Write|MultiEdit", "mcp__market-data__snapshot"]
    assert hooks["PreToolUse"][0]["hooks"][0]["command"].endswith("engine_hooks/protect_harness/protect_harness.py")
    agents = json.loads((session_dir / ".claude" / "harness-agents.json").read_text())
    assert list(agents) == ["lookahead-auditor"]
    assert agents["lookahead-auditor"]["description"].startswith("Audits one attempt")
    assert agents["lookahead-auditor"]["prompt"].startswith(f"{GUIDANCE_HEADING} (harness trading 0.1.0)")
    assert "You audit one attempt" in agents["lookahead-auditor"]["prompt"]

    assert staged.agent_specific == {
        "stream_artifact_path": str(tmp_path / "campaign" / ".kapso" / "sessions" / "generic_exp_0" / "stream.jsonl"),
        "session_record_path": str(tmp_path / "campaign" / ".kapso" / "sessions" / "generic_exp_0" / "session.json"),
        "settings_path": str(session_dir / ".claude" / "harness-hooks.json"),
        "agents_path": str(session_dir / ".claude" / "harness-agents.json"),
    }
    assert staged.mcp_servers == {"market-data": {
        "command": "python3",
        "args": [str(harness.path / "servers" / "market_data.py")],
        "cwd": str(session_dir),
        "env": {HARNESS_ROOT_VARIABLE: str(harness.path)},
    }}
    assert staged.allowed_tools == ["mcp__market-data"]
    assert staged.env == {HARNESS_ROOT_VARIABLE: str(harness.path)}

    guidance = staged.guidance
    assert (harness.path / "instructions" / "guardrails.md").read_text().strip() in guidance
    assert (harness.path / "instructions" / "implementation.md").read_text().strip() in guidance
    assert "- backtesting — Build and run a backtest" in guidance and "`/backtesting`" in guidance
    assert '- lookahead-auditor — Audits one attempt' in guidance and 'subagent_type "lookahead-auditor"' in guidance
    assert "- market-data — its tools are named mcp__market-data__<tool>" in guidance
    # The campaign has no task statement yet (a build writes it before its
    # sessions; a learning run never has one): no line names it.
    assert "Task statement" not in guidance
    assert f"- Harness root: ${HARNESS_ROOT_VARIABLE} = {harness.path}" in guidance

    record = json.loads((tmp_path / "campaign" / ".kapso" / "sessions" / "generic_exp_0" / "session.json").read_text())
    assert record["module"] == "implementation" and record["host"] == "claude_code"
    assert record["harness"] == harness.ref_dict()
    assert record["skills"] == ["backtesting", "market-data", "risk-limits", "lookahead-scan"]
    assert record["hooks"][-1] == "require-results" and record["pages"] == ["guardrails.md", "implementation.md"]
    assert record["runs"] == [] and record["stream"] == staged.agent_specific["stream_artifact_path"]
    assert staged.agent_specific["session_record_path"] == str(staged.record_path)
    assert staged.record == record


def test_the_same_module_on_codex(harness, tmp_path):
    mcp = harness.path / "mcp.json"
    data = json.loads(mcp.read_text())
    data["mcpServers"]["market-data"]["env"] = {"MARKET_TOKEN": "${MARKET_TOKEN}", "REGION": "eu"}
    mcp.write_text(json.dumps(data, indent=2))
    settings = dict(load_config()["harness"], store=str(tmp_path / "store"))
    harness = load_harness(str(harness.path), settings=settings, say=lambda line: None)

    staged = stage(harness, "implementation", "codex", tmp_path)
    session_dir = staged.session_dir
    assert tree(session_dir) == [
        ".agents/skills/backtesting/SKILL.md",
        ".agents/skills/lookahead-scan/SKILL.md",
        ".agents/skills/lookahead-scan/scripts/rules.py",
        ".agents/skills/lookahead-scan/scripts/scan.py",
        ".agents/skills/market-data/SKILL.md",
        ".agents/skills/risk-limits/SKILL.md",
        ".codex/harness-agents/lookahead-auditor.toml",
        ".codex/hooks.json",
    ]
    role = (session_dir / ".codex" / "harness-agents" / "lookahead-auditor.toml").read_text()
    assert role.startswith('developer_instructions = "')
    assert json.loads(role.split(" = ", 1)[1]).startswith(f"{GUIDANCE_HEADING} (harness trading 0.1.0)")
    assert staged.agent_specific["hooks_trust_bypass"] is True
    assert "settings_path" not in staged.agent_specific and "agents_path" not in staged.agent_specific
    assert staged.agent_specific["agents"] == {"lookahead-auditor": {
        "description": "Audits one attempt for look-ahead and leakage in signal timing, joins, resampling and data access, and reports pass, fail or inconclusive with evidence.",
        "config_file": str(session_dir / ".codex" / "harness-agents" / "lookahead-auditor.toml"),
    }}
    server = staged.mcp_servers["market-data"]
    assert server["approval_mode"] == "approve"
    assert server["env_vars"] == ["MARKET_TOKEN"]
    assert server["env"] == {"REGION": "eu", HARNESS_ROOT_VARIABLE: str(harness.path)}
    assert "`$backtesting`" in staged.guidance and 'spawn the agent role "lookahead-auditor"' in staged.guidance

    # Claude keeps the reference for the host to expand.
    claude = stage(harness, "implementation", "claude_code", tmp_path, session="sessions/other", session_id="other")
    assert claude.mcp_servers["market-data"]["env"]["MARKET_TOKEN"] == "${MARKET_TOKEN}"
    assert "env_vars" not in claude.mcp_servers["market-data"]


def test_a_codex_role_with_an_emoji_reaches_codex_intact(harness, tmp_path, monkeypatch):
    agent = harness.path / "agents" / "lookahead-auditor.md"
    agent.write_text(
        agent.read_text(encoding="utf-8").replace("Audits one attempt", "\U0001F50E Audits one attempt", 1) + "Then stop \U0001F6D1.\n",
        encoding="utf-8",
    )
    settings = dict(load_config()["harness"], store=str(tmp_path / "store"))
    harness = load_harness(str(harness.path), settings=settings, say=lambda line: None)
    staged = stage(harness, "implementation", "codex", tmp_path)
    role = tomllib.loads((staged.session_dir / ".codex" / "harness-agents" / "lookahead-auditor.toml").read_text(encoding="utf-8"))
    assert role["developer_instructions"].endswith("Then stop \U0001F6D1.\n")

    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    command = CodexCodingAgent(CodingAgentConfig(
        agent_type="codex", model="m", debug_model="m", agent_specific=staged.merged_agent_specific({}),
    ))._exec_command("m", "/tmp/last.txt")
    override = next(arg for arg in command if arg.startswith("agents.lookahead-auditor.description="))
    assert tomllib.loads(override.replace("agents.lookahead-auditor.", "", 1))["description"].startswith("\U0001F50E Audits")


def test_ideation_gets_its_own_lists_and_nothing_more(harness, tmp_path):
    staged = stage(harness, "ideation", "claude_code", tmp_path)
    assert tree(staged.session_dir) == [
        ".claude/harness-hooks.json",
        ".claude/skills/market-structure/SKILL.md",
        ".claude/skills/strategy-families/SKILL.md",
    ]
    assert staged.mcp_servers == {} and staged.allowed_tools == []
    assert "agents_path" not in staged.agent_specific
    assert "### Delegates" not in staged.guidance and "### Tool servers" not in staged.guidance
    assert (harness.path / "instructions" / "ideation.md").read_text().strip() in staged.guidance


def test_a_module_without_an_entry_still_gets_its_pages(harness, tmp_path):
    staged = stage(harness, "learning", "codex", tmp_path)
    assert tree(staged.session_dir) == [".codex/hooks.json"]   # the engine's own hooks, nothing of the module's
    assert (harness.path / "instructions" / "guardrails.md").read_text().strip() in staged.guidance
    assert (harness.path / "instructions" / "learning.md").read_text().strip() in staged.guidance
    assert staged.agent_specific == {
        "hooks_trust_bypass": True,
        "stream_artifact_path": str(tmp_path / "campaign" / ".kapso" / "sessions" / "generic_exp_0" / "stream.jsonl"),
        "session_record_path": str(tmp_path / "campaign" / ".kapso" / "sessions" / "generic_exp_0" / "session.json"),
    }


def test_no_harness_stages_nothing_and_keeps_the_record(tmp_path):
    """Without a harness nothing is staged, and the session still has its
    record (`harness: null`) for the adapter's run entries to land in."""
    staged = stage_harness(None, "implementation", session_dir=tmp_path, campaign_dir=tmp_path, session_id="x", host="codex")
    assert staged.guidance == "" and staged.mcp_servers == {} and staged.env == {}
    assert tree(tmp_path) == [".kapso/sessions/x/session.json"]
    record = json.loads(staged.record_path.read_text())
    assert record["harness"] is None and record["module"] == "implementation" and record["host"] == "codex"
    assert record["skills"] == [] and record["hooks"] == [] and record["pages"] == [] and record["runs"] == []
    assert record["stream"] == str(tmp_path / ".kapso" / "sessions" / "x" / "stream.jsonl")
    assert staged.agent_specific == {
        "stream_artifact_path": record["stream"],
        "session_record_path": str(tmp_path / ".kapso" / "sessions" / "x" / "session.json"),
    }
    # Without a harness a session keeps its own keys beside the two paths.
    assert staged.merged_agent_specific({"timeout": 5}) == {**staged.agent_specific, "timeout": 5}
    # A learning run's records live under its own directory.
    elsewhere = stage_harness(
        None, "learning", session_dir=tmp_path, campaign_dir=tmp_path, session_id="lead", host="claude_code",
        records_dir=tmp_path / "run" / ".kapso" / "sessions",
    )
    assert elsewhere.record_path == tmp_path / "run" / ".kapso" / "sessions" / "lead" / "session.json"


def test_the_judge_attaches_its_session_record_without_a_harness(tmp_path, monkeypatch):
    """The judge made its agent again with the staged keys only when a
    harness gave guidance, so a campaign with no harness judged its
    attempts and recorded no run (the WP6 proof run's churn build). The
    record and the stream are the session's, harness or not."""
    created = []
    monkeypatch.setattr(
        feedback_module.CodingAgentFactory, "create",
        classmethod(lambda cls, config: created.append(dict(config.agent_specific)) or NoopAgent(config)),
    )
    generator = FeedbackGenerator(CodingAgentConfig(
        agent_type="claude_code", model="m", debug_model="m", agent_specific={"streaming": True, "timeout": 5},
    ))
    staged = stage_harness(None, "feedback", session_dir=tmp_path, campaign_dir=tmp_path, session_id="feedback/generic_exp_0", host="claude_code")
    generator.attach_harness(staged)
    assert created[-1]["session_record_path"] == str(staged.record_path)
    assert created[-1]["stream_artifact_path"] == str(tmp_path / ".kapso" / "sessions" / "feedback" / "generic_exp_0" / "stream.jsonl")
    assert created[-1]["timeout"] == 5 and created[-1]["streaming"] is True
    # The next attempt's staging merges onto the judge's own settings,
    # not onto the previous attempt's record.
    again = stage_harness(None, "feedback", session_dir=tmp_path, campaign_dir=tmp_path, session_id="feedback/generic_exp_1", host="claude_code")
    generator.attach_harness(again)
    assert created[-1]["session_record_path"] == str(again.record_path) and len(created) == 3


def test_a_session_staged_again_keeps_the_runs_already_recorded(harness, tmp_path):
    """A phase re-run after a crash, or a resume, stages the same label
    again: the record keeps the invocation that failed, so the trace shows
    how the first one ended (wp6-observability.md, change 11)."""
    failed = run_entry(
        digest_claude([]), model="claude-opus-5", cli_session_id="s1", exit=EXIT_ERROR, error="Quota exceeded",
        files_changed=[], started="2026-10-25T10:00:00+00:00", ended="2026-10-25T10:00:30+00:00", duration_seconds=30.0,
    )
    for loaded in (None, harness):
        campaign = tmp_path / ("bare" if loaded is None else "with")
        git_workspace(campaign)
        first = stage_harness(loaded, "implementation", session_dir=campaign, campaign_dir=campaign, session_id="generic_exp_0", host="claude_code")
        append_run(first.record_path, failed)
        again = stage_harness(loaded, "implementation", session_dir=campaign, campaign_dir=campaign, session_id="generic_exp_0", host="claude_code")
        record = json.loads(again.record_path.read_text())
        assert [run["exit"] for run in record["runs"]] == [EXIT_ERROR] and record["runs"][0]["error"] == "Quota exceeded"


def test_every_host_stages_into_the_directories_a_harvest_leaves_out(harness, tmp_path):
    # The contract between staging and the trajectory store: a harvested
    # bundle holds none of a session's staging, so the mining crew can own
    # these names at the bundle root. A host that staged elsewhere would
    # leak its files into the raw inventory.
    for host in ("claude_code", "codex"):
        staged = stage(harness, "implementation", host, tmp_path, session=f"sessions/{host}", session_id=host)
        roots = {path.split("/")[0] for path in tree(staged.session_dir)}
        assert roots and roots <= set(STAGING_DIRECTORIES)


def test_the_statement_and_prepared_resources_reach_a_session_directory(harness, tmp_path):
    staged = stage(harness, "implementation", "claude_code", tmp_path, statement=True, prepared=True)
    session_dir = staged.session_dir
    assert (session_dir / "kapso_task.md").read_text() == "# Task\n\nRaise the Sharpe.\n"
    assert (session_dir / "prepared").is_symlink()
    assert (session_dir / "prepared" / "market-data" / "MANIFEST.json").is_file()
    assert "- Prepared resources: prepared/market-data" in staged.guidance
    assert staged.record["prepared"] == ["market-data"]

    # At the campaign root nothing is copied or linked, and the statement
    # path is the campaign's own.
    campaign = tmp_path / "campaign"
    at_root = stage_harness(harness, "feedback", session_dir=campaign, campaign_dir=campaign, session_id="feedback/x", host="claude_code")
    assert not (campaign / "prepared").is_symlink()
    assert f"- Task statement: {campaign / 'kapso_task.md'}" in at_root.guidance


def test_merged_agent_specific_keeps_the_sessions_own_keys(harness, tmp_path):
    staged = stage(harness, "implementation", "claude_code", tmp_path)
    merged = staged.merged_agent_specific({
        "stream_artifact_path": "/own/stream.jsonl",
        "env_overrides": {"LANE": "1"},
        "mcp_servers": {"gated-knowledge": {"command": "python"}},
        "timeout": 60,
    })
    assert merged["stream_artifact_path"] == "/own/stream.jsonl" and merged["timeout"] == 60
    assert merged["env_overrides"] == {HARNESS_ROOT_VARIABLE: str(harness.path), "LANE": "1"}
    assert set(merged["mcp_servers"]) == {"gated-knowledge", "market-data"}
    assert merged["settings_path"] == staged.agent_specific["settings_path"]


def test_staging_is_idempotent_and_two_hosts_share_a_directory(harness, tmp_path):
    first = stage(harness, "implementation", "claude_code", tmp_path)
    again = stage(harness, "implementation", "claude_code", tmp_path)
    assert tree(first.session_dir) == tree(again.session_dir) and first.record == again.record
    codex = stage(harness, "implementation", "codex", tmp_path)
    listing = tree(codex.session_dir)
    assert ".claude/harness-hooks.json" in listing and ".codex/hooks.json" in listing
    assert ".claude/skills/backtesting/SKILL.md" in listing and ".agents/skills/backtesting/SKILL.md" in listing


def test_a_tracked_collision_is_refused_by_name(harness, tmp_path):
    campaign = campaign_with(tmp_path)
    session_dir = campaign / "sessions" / "generic_exp_0"
    git_workspace(session_dir)
    own = session_dir / ".claude" / "skills" / "backtesting" / "SKILL.md"
    own.parent.mkdir(parents=True)
    own.write_text("---\nname: backtesting\ndescription: mine\n---\n")
    git(session_dir, "add", ".claude")
    git(session_dir, "commit", "-q", "-m", "my own skill")
    with pytest.raises(HarnessError) as caught:
        stage_harness(harness, "implementation", session_dir=session_dir, campaign_dir=campaign, session_id="generic_exp_0", host="claude_code")
    assert "tracks .claude/skills/backtesting" in str(caught.value)
    assert not (session_dir / ".claude" / "harness-hooks.json").exists()
    assert own.read_text().endswith("description: mine\n---\n")


def test_a_harness_changed_since_the_load_is_refused(harness, tmp_path):
    page = harness.path / "instructions" / "guardrails.md"
    page.write_text(page.read_text() + "\n- one more rule\n")
    (harness.path / "instructions" / "notes.md").write_text("new\n")
    with pytest.raises(HarnessError, match="the harness changed since launch") as caught:
        stage(harness, "implementation", "claude_code", tmp_path)
    # Named: the changed and the added files, and the way out.
    assert "changed instructions/guardrails.md; added instructions/notes.md" in str(caught.value)
    assert "restore the recorded content" in str(caught.value)


def test_the_adapters_carry_the_staged_keys(harness, tmp_path, monkeypatch):
    # The adapters look for their CLIs at construction; the test machine
    # need not have either, this test is about the commands they build.
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    claude = stage(harness, "implementation", "claude_code", tmp_path)
    command = ClaudeCodeCodingAgent(CodingAgentConfig(
        agent_type="claude_code", model="m", debug_model="m",
        agent_specific=claude.merged_agent_specific({"allowed_tools": ["Read", *claude.allowed_tools]}),
    ))._build_command("m")
    settings = json.loads(command[command.index("--settings") + 1])
    assert settings["hooks"] == json.loads(Path(claude.agent_specific["settings_path"]).read_text())["hooks"]
    assert settings["enabledPlugins"] == {"kapso@kapso": False}
    assert command[command.index("--agents") + 1] == claude.agent_specific["agents_path"]
    assert "mcp__market-data" in command[command.index("--allowedTools") + 1]

    codex = stage(harness, "implementation", "codex", tmp_path, session="sessions/codex", session_id="codex")
    command = CodexCodingAgent(CodingAgentConfig(
        agent_type="codex", model="m", debug_model="m",
        agent_specific=codex.merged_agent_specific({"add_dirs": ["/campaign/.kapso/verification/x"]}),
    ))._exec_command("m", "/tmp/last.txt")
    assert "--dangerously-bypass-hook-trust" in command
    assert command[command.index("--add-dir") + 1] == "/campaign/.kapso/verification/x"
    joined = " ".join(command)
    assert 'mcp_servers.market-data.default_tools_approval_mode="approve"' in joined
    assert f'mcp_servers.market-data.cwd="{codex.session_dir}"' in joined
    assert 'agents.lookahead-auditor.description="Audits one attempt' in joined
    assert f'agents.lookahead-auditor.config_file="{codex.session_dir / ".codex" / "harness-agents" / "lookahead-auditor.toml"}"' in joined


def test_staging_another_module_into_the_same_directory_leaves_only_its_attachments(harness, tmp_path):
    """Sessions at the campaign root follow one another (setup, the judge,
    a maintainer session): each sees exactly its module's skills, hooks
    and delegates, nothing inherited from the one before."""
    first = stage(harness, "setup", "claude_code", tmp_path, session="", session_id="harness_setup")
    root = first.session_dir
    assert (root / ".claude" / "skills" / "market-data").is_dir()
    assert json.loads((root / ".claude" / "harness-hooks.json").read_text())["hooks"].keys() >= {"SessionStart", "PreToolUse"}

    second = stage(harness, "feedback", "claude_code", tmp_path, session="", session_id="feedback/generic_exp_0")
    assert sorted(p.name for p in (root / ".claude" / "skills").iterdir()) == ["risk-limits"]
    assert json.loads((root / ".claude" / "harness-agents.json").read_text()).keys() == {"risk-reviewer"}
    assert second.record["skills"] == ["risk-limits"] and second.record["hooks"] == ["protect-harness", "session-brief", "deny-web-fetch", "protect-prepared"]

    third = stage(harness, "learning", "claude_code", tmp_path, session="", session_id="learning")
    # A module with no hooks of its own still gets the engine's.
    assert list(json.loads((root / ".claude" / "harness-hooks.json").read_text())["hooks"]) == ["PreToolUse"]
    assert not (root / ".claude" / "harness-agents.json").exists()
    assert list((root / ".claude" / "skills").iterdir()) == []
    assert third.record["skills"] == [] and third.record["hooks"] == ["protect-harness"] and third.record["agents"] == []


def test_the_record_names_the_stream_a_session_writes(harness, tmp_path):
    staged = stage(harness, "ideation", "claude_code", tmp_path, session=".kapso/ideation/iter1", session_id="ideation/iter1/claude_code")
    record_path = staged.record_path
    assert record_path == tmp_path / "campaign" / ".kapso" / "sessions" / "ideation" / "iter1" / "claude_code" / "session.json"
    assert json.loads(record_path.read_text())["stream"] == staged.agent_specific["stream_artifact_path"]
    # An ideation session streams beside its artifacts; the record follows it.
    elsewhere = str(tmp_path / "campaign" / ".kapso" / "ideation" / "iter1" / "claude_code_opus.stream.txt")
    merged = staged.merged_agent_specific({"stream_artifact_path": elsewhere})
    assert merged["stream_artifact_path"] == elsewhere
    assert json.loads(record_path.read_text())["stream"] == elsewhere


def test_the_ignore_writer_matches_whole_lines(tmp_path):
    """A seed whose ignore file mentions `.claude/` in a comment or ignores
    `data/prepared/` has not ignored the staged files; the writer adds the
    lines a session's closing add-all must not sweep up."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / ".gitignore").write_text("# prepared resources live under data/prepared/\n.claude/settings.local.json\n")
    commits = []
    fake = SimpleNamespace(
        workspace_dir=str(workspace),
        repo=SimpleNamespace(git=SimpleNamespace(add=lambda paths: None, commit=lambda *args, **kwargs: commits.append(args))),
    )
    ExperimentWorkspace._ensure_workspace_gitignore(fake)
    lines = [line.strip() for line in (workspace / ".gitignore").read_text().splitlines()]
    assert "prepared" in lines and ".claude/" in lines and ".agents/" in lines and ".codex/" in lines and "kapso_task.md" in lines
    # Written once: a second pass adds nothing.
    before = (workspace / ".gitignore").read_text()
    ExperimentWorkspace._ensure_workspace_gitignore(fake)
    assert (workspace / ".gitignore").read_text() == before and len(commits) == 1
