"""`kapso doctor` — the command surface over the preflight resolver.

The doctor is the first thing a new engineer runs; a check that lies
(false OK, a failure without its fix, or a requirement the config never
asked for) costs exactly the support round-trip the command exists to
prevent. The resolver's own behaviour is pinned in test_preflight.py —
what lives here is the CLI contract: scope, exit code, and --config.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import kapso.core.preflight as preflight
from kapso.cli import DEFAULT_CONFIG_PATH, cmd_doctor


def args(verb=None, models=False, config=None):
    return SimpleNamespace(verb=verb, models=models, config=config)


@pytest.fixture
def full_machine(monkeypatch):
    monkeypatch.setattr(
        preflight.shutil, "which", lambda name, *a, **k: f"/usr/bin/{name}")
    monkeypatch.setattr(preflight, "port_open", lambda *a, **k: True)
    monkeypatch.setattr(preflight, "claude_logged_in", lambda: True)
    monkeypatch.setattr(preflight, "codex_authenticated", lambda: True)
    monkeypatch.setattr(preflight, "bank_origin", lambda home: None)
    for name in ("OPENAI_API_KEY", "GITHUB_PAT", "LEEROOPEDIA_API_KEY"):
        monkeypatch.setenv(name, "set")


@pytest.fixture
def bare_machine(monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda name, *a, **k: None)
    monkeypatch.setattr(preflight, "port_open", lambda *a, **k: False)
    monkeypatch.setattr(preflight, "claude_logged_in", lambda: False)
    monkeypatch.setattr(preflight, "codex_authenticated", lambda: False)
    monkeypatch.setattr(preflight, "bank_origin", lambda home: None)
    for name in ("OPENAI_API_KEY", "GITHUB_PAT", "LEEROOPEDIA_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_a_ready_machine_passes_and_optional_gaps_never_fail_it(
    full_machine, monkeypatch, capsys,
):
    # An engineer without modal or bentoml must still get a passing doctor
    # for the core loop — optional targets are reported, never required.
    real_which = preflight.shutil.which
    monkeypatch.setattr(
        preflight.shutil, "which",
        lambda name, *a, **k: None if name in ("modal", "bentoml")
        else real_which(name),
    )
    cmd_doctor(args())  # no SystemExit
    out = capsys.readouterr().out
    assert "Required checks passed" in out
    assert "[-- ] deploy target modal" in out


def test_missing_requirements_exit_nonzero_with_a_fix_for_each(
    bare_machine, capsys,
):
    with pytest.raises(SystemExit) as excinfo:
        cmd_doctor(args())
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "[FAIL] codex CLI" in out
    assert "npm install -g @openai/codex" in out
    # Every failed line is actionable: what wants it, and what to do.
    assert "needed by" in out and "fix" in out
    assert "required check(s) failed" in out


def test_naming_a_verb_narrows_the_report_to_that_verb(full_machine, capsys):
    """`kapso doctor research` must show research's requirements only —
    the packaged research role is codex-only, so a claude row here would
    mean the doctor and the verb disagree about what a call needs."""
    cmd_doctor(args(verb="research"))
    out = capsys.readouterr().out
    assert "kapso doctor research" in out
    assert "codex CLI" in out
    assert "claude CLI" not in out
    assert "Neo4j" not in out


def test_bare_doctor_reports_the_union_across_verbs(full_machine, capsys):
    cmd_doctor(args())
    out = capsys.readouterr().out
    # learn's claude+codex crews, learn_knowledge's stores, deploy's targets.
    for expected in ("codex CLI", "claude CLI", "Neo4j (localhost:7687)",
                     "deploy target docker"):
        assert expected in out


def test_config_flag_drives_the_requirements_including_the_bank_row(
    full_machine, monkeypatch, tmp_path, capsys,
):
    """Regression: the bank rows used to read the PACKAGED config even
    under --config, so `kapso doctor --config mine.yaml` reported someone
    else's bank. Everything the doctor says must come from the config it
    was handed."""
    config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
    config["learning"]["bank"]["local_path"] = str(tmp_path / "mine.git")
    (tmp_path / "mine.git").mkdir()
    # An all-codex config: the claude rows must disappear with it.
    mode = config["modes"]["GENERIC"]
    mode["search_strategy"]["params"]["idea_generation_model"] = "gpt-5.6-sol"
    mode["search_strategy"]["params"]["implementation_model"] = "gpt-5.6-sol"
    mode["coding_agent"] = {"type": "codex", "model": "gpt-5.6-sol"}
    mode["feedback_generator"] = {"type": "codex", "model": "gpt-5.6-sol"}
    path = tmp_path / "mine.yaml"
    path.write_text(yaml.safe_dump(config))

    cmd_doctor(args(verb="evolve", config=str(path)))
    out = capsys.readouterr().out
    assert "codex CLI" in out
    assert "claude CLI" not in out

    capsys.readouterr()
    cmd_doctor(args(verb="learn", config=str(path)))
    assert str(tmp_path / "mine.git") in capsys.readouterr().out


def test_models_flag_adds_the_live_tier_and_can_fail_the_doctor(
    full_machine, monkeypatch, capsys,
):
    """A capped model is invisible to every static check — it is precisely
    why the live tier exists (onboarding E2E finding #5), so it has to be
    able to fail the doctor."""
    monkeypatch.setattr(
        preflight, "probe_model_access",
        lambda cli, model: (False, "You've reached your Fable 5 limit"),
    )
    with pytest.raises(SystemExit) as excinfo:
        cmd_doctor(args(verb="research", models=True))
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "codex can serve gpt-5.6-sol" in out
    assert "You've reached your Fable 5 limit" in out


def test_without_models_no_live_probe_is_ever_spent(full_machine, monkeypatch):
    """The static tier is free; the live tier costs quota. A plain
    `kapso doctor` must never spend it."""
    monkeypatch.setattr(
        preflight, "probe_model_access",
        lambda cli, model: pytest.fail("live probe fired without --models"),
    )
    cmd_doctor(args())
