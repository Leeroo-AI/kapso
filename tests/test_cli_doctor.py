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
                     openai_key=True, codex_auth=True, oauth_token=False):
    monkeypatch.setattr(
        shutil, "which", lambda name: f"/usr/bin/{name}" if name in binaries else None
    )

    def fake_run(cmd, **kwargs):
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
        cmd_doctor(SimpleNamespace())
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "[FAIL] codex CLI" in out
    assert "fix:" in out

    fake_environment(
        monkeypatch, binaries={"node", "codex", "claude", "docker"},
        ports={8080, 7687},
    )
    cmd_doctor(SimpleNamespace())  # no SystemExit
    assert "Required checks passed" in capsys.readouterr().out
