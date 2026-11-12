"""`kapso harness export` (docs/plans/redesign/wp7-plugin.md, change 7)
on the trading example: for each host the files an ordinary session
discovers, the union over the modules in first-occurrence order, the
servers with `${PLUGIN_ROOT}` expanded and the harness root in their
environment, the standing rules as a page with the include line printed,
hooks and the module pages left behind and said, the person's own files
never written, and one existing destination refusing the whole export."""

import json
import shlex
import shutil
import sys
import tomllib
from pathlib import Path

import pytest

import kapso.cli.harness as harness_cli
from kapso.cli import main
from kapso.core.front_matter import read_front_matter
from kapso.harness.export import HarnessExportError, _codex_server_lines, export_harness

TRADING = Path(__file__).resolve().parent.parent / "examples" / "harnesses" / "trading"
SKILLS_IN_ORDER = ("market-data", "strategy-families", "market-structure", "backtesting", "risk-limits", "lookahead-scan", "paper-trading")


def test_the_claude_code_export_writes_the_sessions_files_and_leaves_hooks_behind(tmp_path):
    into = tmp_path.resolve()
    export = export_harness(TRADING, "claude_code", into)
    assert [path.as_posix() for path in export.written] == [
        *(f".claude/skills/{name}" for name in SKILLS_IN_ORDER),
        ".claude/agents/lookahead-auditor.md",
        ".claude/agents/risk-reviewer.md",
        ".mcp.json",
        "kapso-harness-trading.md",
    ]
    assert (into / ".claude" / "skills" / "lookahead-scan" / "scripts" / "scan.py").is_file()
    data, _ = read_front_matter(into / ".claude" / "agents" / "lookahead-auditor.md")
    assert data["name"] == "lookahead-auditor" and data["description"]
    servers = json.loads((into / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]
    assert set(servers) == {"market-data", "broker"}
    assert servers["market-data"]["args"] == [str(TRADING / "servers" / "market_data.py")]
    assert servers["broker"]["env"] == {"KAPSO_HARNESS_ROOT": str(TRADING)}
    assert "${PLUGIN_ROOT}" not in (into / ".mcp.json").read_text(encoding="utf-8")
    assert (into / "kapso-harness-trading.md").read_text(encoding="utf-8") == (TRADING / "instructions" / "guardrails.md").read_text(encoding="utf-8")
    assert not list(into.rglob("hook.json")) and not (into / ".claude" / "settings.json").exists()
    assert not (into / "CLAUDE.md").exists()
    text = export.render()
    assert text.startswith(f"exported trading 0.1.0 for claude_code into {into}\n  .claude/skills/market-data/\n")
    assert "left behind: 7 hook(s) (session-brief, deny-web-fetch, protect-prepared, deny-web, deny-snapshot, scan-writes, require-results): campaign-bound" in text
    assert "left behind: the module page(s) (setup, ideation, implementation, verification, feedback, learning, deployment) and the routing per module" in text
    assert "next:\n  add to CLAUDE.md (never written by the export): @kapso-harness-trading.md\n" in text
    assert f"  export KAPSO_HARNESS_ROOT={TRADING} in the shell these skills run from (lookahead-scan" in text


def test_the_codex_export_writes_toml_roles_and_prints_the_config_and_server_lines(tmp_path):
    into = tmp_path.resolve()
    export = export_harness(TRADING, "codex", into)
    assert (into / ".agents" / "skills" / "backtesting" / "SKILL.md").is_file()
    role_file = into / ".codex" / "agents" / "lookahead-auditor.toml"
    _, body = read_front_matter(TRADING / "agents" / "lookahead-auditor.md")
    assert tomllib.loads(role_file.read_text(encoding="utf-8")) == {"developer_instructions": body.strip() + "\n"}
    assert not (into / ".mcp.json").exists() and not (into / "AGENTS.md").exists()
    text = export.render()
    assert "add to AGENTS.md (never written by the export): Read kapso-harness-trading.md before any work in this domain." in text
    assert f"[agents.lookahead-auditor]\n    description = " in text
    assert f"config_file = {json.dumps(str(role_file))}" in text
    assert shlex.join([
        "codex", "mcp", "add", "market-data", "--env", f"KAPSO_HARNESS_ROOT={TRADING}",
        "--", "python3", str(TRADING / "servers" / "market_data.py"),
    ]) in text
    assert "codex mcp add broker " in text


def test_the_codex_role_is_valid_toml_when_the_agent_has_an_emoji(tmp_path):
    harness = tmp_path / "trading"
    shutil.copytree(TRADING, harness)
    agent = harness / "agents" / "lookahead-auditor.md"
    agent.write_text(
        agent.read_text(encoding="utf-8").replace("Audits one attempt", "\U0001F50E Audits one attempt", 1) + "Report \u2705 or \u274C, then stop \U0001F6D1.\n",
        encoding="utf-8",
    )
    into = (tmp_path / "out").resolve()
    export = export_harness(harness, "codex", into)
    data, body = read_front_matter(agent)
    role = tomllib.loads((into / ".codex" / "agents" / "lookahead-auditor.toml").read_text(encoding="utf-8"))
    assert role == {"developer_instructions": body.strip() + "\n"}
    lines = export.render().split("[agents.lookahead-auditor]\n", 1)[1].splitlines()[:2]
    assert tomllib.loads("\n".join(line.strip() for line in lines))["description"] == data["description"]


def test_one_existing_destination_refuses_the_whole_export_by_name(tmp_path):
    (tmp_path / ".claude" / "skills" / "backtesting").mkdir(parents=True)
    (tmp_path / "kapso-harness-trading.md").write_text("mine\n", encoding="utf-8")
    with pytest.raises(HarnessExportError, match=r"^nothing written: \.claude/skills/backtesting, kapso-harness-trading\.md in the way under "):
        export_harness(TRADING, "claude_code", tmp_path)
    assert not (tmp_path / ".mcp.json").exists()
    assert not (tmp_path / ".claude" / "skills" / "market-data").exists()
    assert (tmp_path / "kapso-harness-trading.md").read_text(encoding="utf-8") == "mine\n"


def test_an_unknown_host_is_refused_by_name(tmp_path):
    with pytest.raises(HarnessExportError, match="unknown host 'cursor'; the hosts are claude_code, codex"):
        export_harness(TRADING, "cursor", tmp_path)


@pytest.fixture
def settings(monkeypatch, tmp_path):
    values = {"store": str(tmp_path / "store"), "page_warning_lines": 40, "setup_minutes": 20,
              "verification_minutes": 10, "verification_attempts": 2, "server_tool_timeout_seconds": 600}
    monkeypatch.setattr(harness_cli, "load_config", lambda config=None: {"harness": dict(values)})
    monkeypatch.chdir(tmp_path)
    return values


def test_the_verb_exports_a_directory_and_refuses_a_harness_that_fails_check(settings, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, "argv", ["kapso", "harness", "export", str(TRADING), "--host", "codex", "--to", str(tmp_path / "out")])
    main()
    out = capsys.readouterr().out
    assert out.startswith(f"exported trading 0.1.0 for codex into {(tmp_path / 'out').resolve()}\n")
    assert (tmp_path / "out" / ".agents" / "skills" / "market-data" / "SKILL.md").is_file()
    broken = tmp_path / "broken"
    shutil.copytree(TRADING, broken)
    shutil.rmtree(broken / "skills" / "backtesting")
    monkeypatch.setattr(sys, "argv", ["kapso", "harness", "export", str(broken), "--host", "codex", "--to", str(tmp_path / "out2")])
    with pytest.raises(SystemExit) as exit_:
        main()
    assert exit_.value.code == 1
    assert "fail" in capsys.readouterr().out
    assert not (tmp_path / "out2").exists()


def test_a_file_on_the_way_to_a_destination_refuses_the_export_before_any_write(tmp_path):
    into = tmp_path.resolve()
    (into / ".claude").mkdir()
    (into / ".claude" / "skills").write_text("a file where the skills directory must be")
    with pytest.raises(HarnessExportError, match="in the way under"):
        export_harness(TRADING, "claude_code", into)
    assert not (into / ".claude" / "agents").exists() and not (into / "kapso-harness-trading.md").exists()


def test_the_codex_lines_for_an_http_server_are_toml():
    lines = _codex_server_lines("api", {"url": "https://api.example", "headers": {"X-Key": "k1", "Authorization": "${TOKEN}"}})
    assert lines[0] == "codex mcp add api --url https://api.example"
    tables = [line.split("config.toml: ", 1)[1] for line in lines[1:]]
    parsed = [tomllib.loads(f"[mcp_servers.api]\n{table}") for table in tables]
    assert parsed[0]["mcp_servers"]["api"]["http_headers"] == {"X-Key": "k1"}
    assert parsed[1]["mcp_servers"]["api"]["env_http_headers"] == {"Authorization": "TOKEN"}
