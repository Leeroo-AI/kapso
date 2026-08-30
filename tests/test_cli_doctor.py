"""kapso doctor — hermetic pins (onboarding audit 2026-08-26).

The doctor is the first thing a new engineer runs; a check that lies
(false OK, or a failure without its fix hint) costs exactly the support
round-trip the command exists to prevent.
"""

import shutil
import socket
import subprocess
from types import SimpleNamespace

import pytest

import kapso.cli as cli_module
from kapso.cli import _doctor_checks, cmd_doctor


def fake_environment(monkeypatch, *, binaries, ports=(), claude_logged_in=True,
                     openai_key=True, codex_auth=True, oauth_token=False,
                     bank_origin_url=None):
    monkeypatch.setattr(
        shutil, "which", lambda name: f"/usr/bin/{name}" if name in binaries else None
    )

    def fake_run(cmd, **kwargs):
        if cmd[0] == "git":  # the doctor's bank-origin probe
            if bank_origin_url:
                return SimpleNamespace(
                    returncode=0, stdout=bank_origin_url + "\n", stderr="")
            return SimpleNamespace(
                returncode=2, stdout="", stderr="error: No such remote")
        assert cmd[:3] == ["claude", "auth", "status"]
        return SimpleNamespace(
            returncode=0,
            stdout='{"loggedIn": %s}' % ("true" if claude_logged_in else "false"),
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_create_connection(address, timeout=None):
        if address[1] in ports:
            return FakeConnection()
        raise OSError("refused")

    monkeypatch.setattr(socket, "create_connection", fake_create_connection)
    monkeypatch.setattr(
        cli_module.Path, "home",
        classmethod(lambda cls: FakeHome(codex_auth)),
    )
    if openai_key:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    else:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    if oauth_token:
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "tok")
    else:
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)


class FakeHome:
    def __init__(self, codex_auth):
        self._codex_auth = codex_auth

    def __truediv__(self, part):
        if part == ".codex":
            return self
        return SimpleNamespace(is_file=lambda: self._codex_auth)


def by_label(checks):
    return {label: (ok, required) for ok, required, label, _ in checks}


def test_all_present_passes_and_optional_infra_is_not_required(monkeypatch):
    fake_environment(
        monkeypatch,
        binaries={"node", "codex", "claude", "docker"},
        ports={8080},  # weaviate up, neo4j down
    )
    checks = by_label(_doctor_checks())
    assert checks["codex CLI"] == (True, True)
    assert checks["codex authenticated"] == (True, True)
    assert checks["claude authenticated"] == (True, True)
    assert checks["OPENAI_API_KEY (embeddings)"] == (True, True)
    assert checks["Weaviate (localhost:8080)"] == (True, False)
    # Down infra is reported but NOT required — a fresh engineer without
    # docker must still get a passing doctor for the core loop.
    assert checks["Neo4j (localhost:7687)"] == (False, False)


def test_missing_clis_fail_with_install_hints(monkeypatch):
    fake_environment(monkeypatch, binaries={"node"}, openai_key=False)
    checks = _doctor_checks()
    failed = {label: hint for ok, required, label, hint in checks
              if not ok and required}
    assert "npm install -g @openai/codex" in failed["codex CLI"]
    assert "claude" in failed["claude CLI"]
    assert "OPENAI_API_KEY" in failed["OPENAI_API_KEY (embeddings)"]
    # No CLI on PATH -> no auth probes (would be noise on top of the miss)
    labels = [label for _, _, label, _ in checks]
    assert "codex authenticated" not in labels
    assert "claude authenticated" not in labels


def test_oauth_token_counts_as_claude_auth_without_subprocess(monkeypatch):
    fake_environment(
        monkeypatch, binaries={"node", "codex", "claude"},
        claude_logged_in=False, oauth_token=True,
    )
    # If the token path did not short-circuit, the fake `claude auth
    # status` above would report logged out and this would fail.
    assert by_label(_doctor_checks())["claude authenticated"] == (True, True)


def test_cmd_doctor_exits_nonzero_on_required_failure(monkeypatch, capsys):
    fake_environment(monkeypatch, binaries=set(), openai_key=False)
    with pytest.raises(SystemExit) as excinfo:
        cmd_doctor(SimpleNamespace(models=False, config=None))
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "[FAIL] codex CLI" in out
    assert "fix:" in out

    fake_environment(
        monkeypatch, binaries={"node", "codex", "claude", "docker"},
        ports={8080, 7687},
    )
    cmd_doctor(SimpleNamespace(models=False, config=None))  # no SystemExit
    assert "Required checks passed" in capsys.readouterr().out


def test_bank_rows_nudge_without_ever_failing_doctor(monkeypatch, tmp_path):
    # Regression (onboarding E2E finding #1 nudges): the bank line is a
    # discoverability nudge, never a required failure — absent bank says
    # "created on first learn()", origin-less bank names `kapso bank
    # connect`, attached bank shows its URL as OK.
    fake_environment(monkeypatch, binaries={"node", "codex", "claude"})
    monkeypatch.chdir(tmp_path)  # bank path resolves against CWD

    checks = by_label(_doctor_checks())
    assert checks["bank (none yet — created on first learn())"] == (False, False)

    (tmp_path / "data" / "kapso-bank.git").mkdir(parents=True)
    hints = {label: hint for _, _, label, hint in _doctor_checks()}
    assert "kapso bank connect" in hints["bank remote: not configured — local-only"]
    assert by_label(_doctor_checks())[
        "bank remote: not configured — local-only"] == (False, False)

    fake_environment(monkeypatch, binaries={"node", "codex", "claude"},
                     bank_origin_url="https://github.com/acme/kapso-bank.git")
    monkeypatch.chdir(tmp_path)
    assert by_label(_doctor_checks())[
        "bank remote: https://github.com/acme/kapso-bank.git"] == (True, False)


def test_collect_model_pairs_covers_the_packaged_config():
    # Regression / drift-catcher (onboarding E2E finding #5): every model
    # the SHIPPED config can spawn must surface in the probe list — a
    # model rename that dodges this list dodges the preflight.
    from kapso.cli import _collect_model_pairs
    from kapso.core.config import load_config
    from kapso.kapso import DEFAULT_CONFIG_PATH

    pairs = set(_collect_model_pairs(load_config(DEFAULT_CONFIG_PATH)))
    assert ("claude_code", "claude-fable-5") in pairs      # learning crews
    assert ("claude_code", "claude-opus-5") in pairs       # evolve sessions
    assert ("codex", "gpt-5.6-sol") in pairs               # codex roles
    # embeddings are the OPENAI_API_KEY row, never a session probe
    assert all("embedding" not in model for _, model in pairs)


def test_collect_model_pairs_resolves_cli_three_ways():
    # Regression: explicit `cli` beats the name prefix; implementation_cli
    # pairs with implementation_model; bare *_model keys fall back to the
    # claude-* prefix rule. Non-model keys never produce pairs.
    from kapso.cli import _collect_model_pairs

    config = {
        "explicit": {"cli": "codex", "model": "claude-named-but-codex"},
        "prefix": {"idea_generation_model": "claude-opus-5"},
        "sibling": {"implementation_model": "gpt-x",
                    "implementation_cli": "codex"},
        "nested": [{"judge": {"cli": "claude_code",
                              "model": "claude-fable-5"}}],
        "not_models": {"embedding": "text-embedding-3-small", "mode": "m"},
    }
    assert set(_collect_model_pairs(config)) == {
        ("codex", "claude-named-but-codex"),
        ("claude_code", "claude-opus-5"),
        ("codex", "gpt-x"),
        ("claude_code", "claude-fable-5"),
    }


def test_probe_model_reports_the_clis_own_cap_message(monkeypatch):
    # Regression: a capped model must fail the probe with the CLI's own
    # first line (the E2E's real case: TOKEN2's Fable-5 window), and the
    # claude/codex command shapes must match the CLIs' one-shot forms.
    from kapso.cli import _probe_model_access
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        if "claude-fable-5" in cmd:
            return SimpleNamespace(
                returncode=1,
                stdout="You've reached your Fable 5 limit. Switch model.\n",
                stderr="")
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ok, detail = _probe_model_access("claude_code", "claude-fable-5")
    assert (ok, detail) == (False, "You've reached your Fable 5 limit. Switch model.")
    assert seen[0][:3] == ["claude", "-p", "--model"]
    ok, detail = _probe_model_access("codex", "gpt-5.6-sol")
    assert (ok, detail) == (True, "")
    assert seen[1][:2] == ["codex", "exec"] and "--sandbox" in seen[1]
    # codex must run in untrusted (non-git) user dirs — the live smoke
    # failed without this flag
    assert "--skip-git-repo-check" in seen[1]


def test_probe_model_takes_codex_reason_from_last_stderr_line(monkeypatch):
    # Regression: codex prefixes stderr with noise ("Reading additional
    # input from stdin...") — the probe must surface the real reason on
    # the last line, not the noise.
    from kapso.cli import _probe_model_access

    def fake_run(cmd, **kwargs):
        return SimpleNamespace(
            returncode=1, stdout="",
            stderr="Reading additional input from stdin...\n"
                   "stream error: model not found\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ok, detail = _probe_model_access("codex", "nope-model")
    assert (ok, detail) == (False, "stream error: model not found")


def test_cmd_doctor_models_gates_exit_on_a_capped_model(
    monkeypatch, capsys, tmp_path
):
    # Regression: --models turns a capped model into a doctor failure
    # (exit 1) with the cap message inline; with every probe OK the
    # doctor passes. Probes run only under --models.
    fake_environment(monkeypatch, binaries={"node", "codex", "claude"})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "crew:\n  cli: claude_code\n  model: claude-fable-5\n"
        "worker:\n  model: gpt-5.6-sol\n"
    )
    probes = {("claude_code", "claude-fable-5"): (False, "Fable 5 limit"),
              ("codex", "gpt-5.6-sol"): (True, "")}
    monkeypatch.setattr(cli_module, "_probe_model_access",
                        lambda cli, model: probes[(cli, model)])

    with pytest.raises(SystemExit) as excinfo:
        cmd_doctor(SimpleNamespace(models=True, config=str(config_path)))
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "[FAIL] claude_code: claude-fable-5 — Fable 5 limit" in out
    assert "[OK ] codex: gpt-5.6-sol" in out

    probes[("claude_code", "claude-fable-5")] = (True, "")
    cmd_doctor(SimpleNamespace(models=True, config=str(config_path)))
    assert "Required checks passed" in capsys.readouterr().out
