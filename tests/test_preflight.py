"""Per-verb preflight — the resolver behind both the facade verbs and
`kapso doctor [verb]`.

What these pin: that requirements are DERIVED from the active config
rather than hardcoded (the whole point — an all-codex config asked for
`claude` is exactly the false failure this replaced), and that a failing
row is actionable, carrying the config key that wants it and one fix. A
row without those two is a support round-trip.
"""

import copy
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import kapso.core.preflight as preflight
from kapso.core.config import load_config
from kapso.core.preflight import (
    PreflightError,
    Requirement,
    configured_bank_home,
    dedupe,
    render,
    requirements_for,
    session_specs,
    summarize_origins,
)


PACKAGED = str(Path(preflight.__file__).resolve().parent.parent / "config.yaml")


@pytest.fixture
def packaged():
    return load_config(PACKAGED)


@pytest.fixture
def full_machine(monkeypatch):
    """Everything installed, everything authenticated, both stores up —
    so any FAIL a test sees comes from the config, not the box."""
    monkeypatch.setattr(
        preflight.shutil, "which",
        lambda name, *a, **k: f"/usr/bin/{name}",
    )
    monkeypatch.setattr(preflight, "port_open", lambda host, port, timeout=2.0: True)
    monkeypatch.setattr(preflight, "claude_logged_in", lambda: True)
    monkeypatch.setattr(preflight, "codex_authenticated", lambda: True)
    for name in ("OPENAI_API_KEY", "GITHUB_PAT", "LEEROOPEDIA_API_KEY"):
        monkeypatch.setenv(name, "set")
    monkeypatch.setattr(
        preflight, "bank_origin", lambda home: None,
    )


def labels(requirements):
    return [item.label for item in requirements]


def by_label(requirements):
    return {item.label: item for item in requirements}


# =============================================================================
# The headline: requirements follow the config, not a hardcoded union
# =============================================================================

def test_all_codex_evolve_never_asks_for_claude(packaged, full_machine):
    """The bug this design replaced: the old doctor demanded `claude` from
    everyone, including a config that can only ever spawn codex."""
    config = copy.deepcopy(packaged)
    mode = config["modes"]["GENERIC"]
    mode["search_strategy"]["params"]["idea_generation_model"] = "gpt-5.6-sol"
    mode["search_strategy"]["params"]["implementation_model"] = "gpt-5.6-sol"
    mode["coding_agent"] = {"type": "codex", "model": "gpt-5.6-sol"}
    mode["feedback_generator"] = {"type": "codex", "model": "gpt-5.6-sol"}

    found = labels(requirements_for("evolve", config))
    assert "codex CLI" in found
    assert "claude CLI" not in found
    assert "claude authenticated" not in found


def test_research_needs_one_cli_and_learn_needs_both(packaged, full_machine):
    """Not a coincidence worth losing: the grading crew is deliberately
    cross-model (report_writer on codex, verifier on claude), so `learn`
    genuinely needs both CLIs while `research` needs only codex."""
    research = labels(requirements_for("research", packaged))
    assert "codex CLI" in research
    assert "claude CLI" not in research

    learn = labels(requirements_for("learn", packaged))
    assert "codex CLI" in learn and "claude CLI" in learn


def test_skip_merge_drops_the_merger_and_both_kg_stores(packaged, full_machine):
    """`skip_merge=True` stops after page extraction — demanding Neo4j for
    a run that never writes to it is exactly the false requirement a
    config-blind checker produces."""
    merging = labels(requirements_for("learn_knowledge", packaged, skip_merge=False))
    assert "Weaviate (localhost:8080)" in merging
    assert "Neo4j (localhost:7687)" in merging
    assert "OPENAI_API_KEY (embeddings)" in merging

    extract_only = labels(
        requirements_for("learn_knowledge", packaged, skip_merge=True)
    )
    assert "Weaviate (localhost:8080)" not in extract_only
    assert "Neo4j (localhost:7687)" not in extract_only
    assert "OPENAI_API_KEY (embeddings)" not in extract_only


def test_codify_target_decides_whether_gcloud_is_required(packaged, full_machine):
    """`learning.codify.target: gcp_ephemeral` shells out to gcloud — a
    requirement no user would guess, and one that must vanish when the
    target is local."""
    assert "gcloud CLI" in labels(requirements_for("learn", packaged))

    local = copy.deepcopy(packaged)
    local["learning"]["codify"]["target"] = "local"
    assert "gcloud CLI" not in labels(requirements_for("learn", local))


def test_kg_rows_appear_only_when_an_index_is_connected(packaged, full_machine):
    """Knowledge search is opt-in via Kapso(kg_index=...); a campaign
    without one must not be blocked on stores it never opens."""
    without = labels(requirements_for("evolve", packaged))
    assert "Neo4j (localhost:7687)" not in without

    with_index = labels(
        requirements_for("evolve", packaged, kg_index="data/indexes/ml.index")
    )
    assert "Neo4j (localhost:7687)" in with_index
    assert "OPENAI_API_KEY (embeddings)" in with_index


def test_caller_passed_coding_agent_overrides_the_mode(packaged, full_machine):
    """evolve(coding_agent=...) changes which CLI actually runs, so the
    requirement must follow the argument, not the config's default."""
    found = labels(requirements_for("evolve", packaged, coding_agent="codex"))
    assert "codex CLI" in found


# =============================================================================
# Gates — the config's own failure policy decides whether a gap blocks
# =============================================================================

def test_gate_policy_decides_whether_a_missing_gate_blocks(
    packaged, full_machine, monkeypatch,
):
    monkeypatch.delenv("LEEROOPEDIA_API_KEY", raising=False)
    real_which = preflight.shutil.which
    monkeypatch.setattr(
        preflight.shutil, "which",
        lambda name, *a, **k: None if name == "leeroopedia-mcp"
        else real_which(name),
    )

    warn = by_label(requirements_for("evolve", packaged))
    assert warn["MCP gate 'leeroopedia'"].required is False

    strict = copy.deepcopy(packaged)
    strict["modes"]["GENERIC"]["search_strategy"]["params"][
        "gate_failure_policy"] = "error"
    blocking = by_label(requirements_for("evolve", strict))
    assert blocking["MCP gate 'leeroopedia'"].required is True
    # `error` is the one policy where downgrading is a real option, so it
    # is the one that says so.
    assert "gate_failure_policy: warn" in blocking["MCP gate 'leeroopedia'"].fix


def test_platform_injected_gate_env_is_never_reported_as_user_setup(
    packaged, full_machine,
):
    """KAPSO_BANK_DIR / EXPERIMENT_HISTORY_PATH / KG_INDEX_PATH are set by
    Kapso when it launches a session. Reporting them at preflight would be
    a guaranteed false positive on every clean machine — and the packaged
    config names the experiment_history gate, so this fires by default."""
    found = labels(requirements_for("evolve", packaged))
    assert "MCP gate 'experiment_history'" not in found


# =============================================================================
# Every failed row must be actionable
# =============================================================================

def test_every_failing_row_names_its_origin_and_a_fix(packaged, monkeypatch):
    """The contract that makes this useful rather than merely correct: a
    user reading a failure must learn what to do and why Kapso wants it."""
    monkeypatch.setattr(preflight.shutil, "which", lambda name, *a, **k: None)
    monkeypatch.setattr(preflight, "port_open", lambda *a, **k: False)
    monkeypatch.setattr(preflight, "claude_logged_in", lambda: False)
    monkeypatch.setattr(preflight, "codex_authenticated", lambda: False)
    for name in ("OPENAI_API_KEY", "GITHUB_PAT", "LEEROOPEDIA_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    seen = 0
    for verb in preflight.VERBS:
        for item in requirements_for(verb, packaged):
            if item.ok:
                continue
            seen += 1
            assert item.origin, f"{verb}/{item.label} has no origin"
            assert item.fix, f"{verb}/{item.label} has no fix"
    assert seen > 0


def test_origins_are_summarized_not_a_wall_of_near_identical_keys():
    """Six config keys all naming claude-opus-5 is unreadable; the summary
    keeps the first key, the count, and the model."""
    specs = [
        preflight.SessionSpec("claude_code", "claude-opus-5", f"a.b.k{i} = claude-opus-5")
        for i in range(6)
    ]
    summary = summarize_origins(specs)
    assert summary == "a.b.k0 (+5 more) = claude-opus-5"

    mixed = summarize_origins([
        preflight.SessionSpec("codex", "gpt-5.6-sol", "x.model = gpt-5.6-sol"),
        preflight.SessionSpec("codex", "gpt-5.4", "y.model = gpt-5.4"),
    ])
    assert "gpt-5.6-sol" in mixed and "gpt-5.4" in mixed


# =============================================================================
# Session extraction
# =============================================================================

def test_session_specs_resolve_the_cli_four_ways():
    """An explicit `cli`, an `implementation_cli` sibling, a `type` naming a
    registered agent, and the model-name prefix. A rename that dodges this
    silently probes the wrong CLI."""
    specs = session_specs({
        "explicit": {"cli": "claude_code", "model": "some-house-model"},
        "impl": {"implementation_cli": "codex",
                 "implementation_model": "claude-named-but-codex-run"},
        "typed": {"type": "codex", "model": "claude-shaped-name"},
        "prefixed": {"model": "claude-opus-5"},
        "fallback": {"model": "gpt-5.6-sol"},
    }, "root")
    resolved = {spec.model: spec.cli for spec in specs}
    assert resolved["some-house-model"] == "claude_code"
    assert resolved["claude-named-but-codex-run"] == "codex"
    assert resolved["claude-shaped-name"] == "codex"
    assert resolved["claude-opus-5"] == "claude_code"
    assert resolved["gpt-5.6-sol"] == "codex"
    assert all(spec.origin.startswith("root.") for spec in specs)


def test_embedding_models_are_not_sessions(packaged):
    """`models.embedding` runs on the API path, not as a CLI session — a
    walker that treats it as one would demand a codex probe for it."""
    specs = session_specs(packaged.get("defaults") or {}, "defaults")
    assert specs == []


def test_auth_mode_api_key_asks_for_the_key_not_the_login(monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda name, *a, **k: f"/b/{name}")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(preflight, "claude_logged_in", lambda: True)

    specs = session_specs(
        {"agent": {"type": "claude_code", "model": "claude-opus-5",
                   "agent_specific": {"auth_mode": "api_key"}}},
        "modes.X",
    )
    found = by_label(preflight.cli_requirements(specs))
    assert "ANTHROPIC_API_KEY" in found
    assert found["ANTHROPIC_API_KEY"].ok is False
    # A stored CLI login does not satisfy an api_key config.
    assert "claude authenticated" not in found


# =============================================================================
# The bank rows
# =============================================================================

def test_bank_push_is_verified_before_work_not_after(packaged, monkeypatch, tmp_path):
    """Onboarding E2E finding #1: an unreachable push destination must cost
    seconds at the start, never hours of crew work at the end."""
    monkeypatch.setattr(preflight.shutil, "which", lambda name, *a, **k: f"/b/{name}")
    monkeypatch.setattr(preflight, "claude_logged_in", lambda: True)
    monkeypatch.setattr(preflight, "codex_authenticated", lambda: True)
    home = tmp_path / "bank.git"
    home.mkdir()

    monkeypatch.setattr(
        preflight, "bank_origin", lambda h: "https://github.com/acme/bank.git")
    monkeypatch.setattr(
        preflight, "bank_remote_error",
        lambda h, url: "fatal: Authentication failed",
    )
    found = by_label(requirements_for("learn", packaged, bank_home=home, push=True))
    row = found["bank remote reachable (https://github.com/acme/bank.git)"]
    assert row.ok is False and row.required is True
    assert "fatal: Authentication failed" in row.detail


def test_push_false_never_probes_an_attached_remote(
    packaged, monkeypatch, tmp_path,
):
    """Caught by the suite during the build: an attached-but-unreachable
    origin blocked a `push=False` run, which by definition never touches
    it. A preflight that fails work it does not gate is worse than none."""
    monkeypatch.setattr(preflight.shutil, "which", lambda name, *a, **k: f"/b/{name}")
    monkeypatch.setattr(preflight, "claude_logged_in", lambda: True)
    monkeypatch.setattr(preflight, "codex_authenticated", lambda: True)
    monkeypatch.setattr(
        preflight, "bank_origin", lambda h: "https://github.com/acme/bank.git")
    monkeypatch.setattr(
        preflight, "bank_remote_error",
        lambda h, url: pytest.fail("probed a remote this run never pushes to"),
    )
    home = tmp_path / "bank.git"
    home.mkdir()
    found = labels(requirements_for("learn", packaged, bank_home=home, push=False))
    assert not any("bank remote" in label for label in found)


def test_local_only_bank_nudges_the_doctor_but_stays_out_of_every_run(
    packaged, monkeypatch, tmp_path,
):
    """`push=None` is the doctor's view (no call in flight) and earns the
    sharing nudge; a resolved `push=False` is a run that decided, and must
    not print the same nudge on every learn()."""
    monkeypatch.setattr(preflight.shutil, "which", lambda name, *a, **k: f"/b/{name}")
    monkeypatch.setattr(preflight, "claude_logged_in", lambda: True)
    monkeypatch.setattr(preflight, "codex_authenticated", lambda: True)
    monkeypatch.setattr(preflight, "bank_origin", lambda h: None)
    home = tmp_path / "bank.git"
    home.mkdir()

    doctor = by_label(requirements_for("learn", packaged, bank_home=home))
    assert doctor["bank remote (sharing)"].required is False
    assert "kapso bank connect" in doctor["bank remote (sharing)"].fix

    during_run = labels(
        requirements_for("learn", packaged, bank_home=home, push=False))
    assert "bank remote (sharing)" not in during_run


def test_configured_bank_home_follows_the_given_config():
    """The doctor's bank rows used to read the PACKAGED config even under
    --config, so they reported someone else's bank."""
    assert configured_bank_home({}) is None
    home = configured_bank_home(
        {"learning": {"bank": {"local_path": "~/elsewhere/bank.git"}}})
    assert home == Path("~/elsewhere/bank.git").expanduser()


# =============================================================================
# Enforcement + settings
# =============================================================================

def test_run_preflight_raises_with_the_report_as_its_message(packaged, monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda name, *a, **k: None)
    with pytest.raises(PreflightError) as excinfo:
        preflight.run_preflight("research", packaged)
    message = str(excinfo.value)
    assert "kapso research — preflight failed" in message
    assert "npm install -g @openai/codex" in message
    assert "kapso doctor research" in message


def test_disabling_preflight_skips_every_probe(packaged, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("preflight probed while disabled")

    monkeypatch.setattr(preflight.shutil, "which", explode)
    config = copy.deepcopy(packaged)
    config["preflight"] = {"enabled": False}
    assert preflight.run_preflight("evolve", config) == []


def test_settings_fall_back_to_the_packaged_defaults(packaged):
    """A config written before the `preflight:` block existed still gets
    the shipped behaviour rather than a silently-off preflight."""
    settings = preflight.preflight_settings({})
    assert settings["enabled"] is True
    assert settings["live_model_probe"] is False

    partial = preflight.preflight_settings({"preflight": {"live_model_probe": True}})
    assert partial["enabled"] is True
    assert partial["live_model_probe"] is True


def test_live_probe_runs_only_when_configured(packaged, monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda name, *a, **k: f"/b/{name}")
    monkeypatch.setattr(preflight, "claude_logged_in", lambda: True)
    monkeypatch.setattr(preflight, "codex_authenticated", lambda: True)
    monkeypatch.setattr(preflight, "port_open", lambda *a, **k: True)
    probed = []

    def fake_probe(cli, model):
        probed.append((cli, model))
        return True, ""

    monkeypatch.setattr(preflight, "probe_model_access", fake_probe)

    preflight.run_preflight("research", packaged)
    assert probed == []

    config = copy.deepcopy(packaged)
    config["preflight"] = {"enabled": True, "live_model_probe": True}
    preflight.run_preflight("research", config)
    assert probed == [("codex", "gpt-5.6-sol")]


def test_live_probe_surfaces_the_clis_own_cap_message(packaged, monkeypatch):
    """Onboarding E2E finding #5: the CLI's own words are the useful part —
    "You've reached your Fable 5 limit" tells the user far more than a
    generic failure would."""
    monkeypatch.setattr(preflight.shutil, "which", lambda name, *a, **k: f"/b/{name}")
    monkeypatch.setattr(
        preflight, "probe_model_access",
        lambda cli, model: (False, "You've reached your Fable 5 limit"),
    )
    rows = preflight.live_model_requirements("research", packaged)
    assert len(rows) == 1
    assert rows[0].ok is False
    assert rows[0].detail == "You've reached your Fable 5 limit"
    assert "Choosing models" in rows[0].fix


def test_live_probe_skips_pairs_whose_cli_is_missing(packaged, monkeypatch):
    """Probing a model through a CLI that is not installed would report a
    second, derivative failure on top of the static row that already said
    the binary is absent."""
    monkeypatch.setattr(preflight.shutil, "which", lambda name, *a, **k: None)
    monkeypatch.setattr(
        preflight, "probe_model_access",
        lambda cli, model: pytest.fail("probed a missing CLI"),
    )
    assert preflight.live_model_requirements("learn", packaged) == []


def test_probe_model_takes_codex_reason_from_the_last_stderr_line(monkeypatch):
    """codex prefixes stderr with progress noise; the real reason is last.
    claude puts its cap message on stdout's first line."""
    monkeypatch.setattr(
        preflight.subprocess, "run",
        lambda *a, **k: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="Reading prompt...\nnoise\nERROR: model not found",
        ),
    )
    ok, detail = preflight.probe_model_access("codex", "gpt-nope")
    assert ok is False and detail == "ERROR: model not found"

    monkeypatch.setattr(
        preflight.subprocess, "run",
        lambda *a, **k: SimpleNamespace(
            returncode=1, stdout="You've reached your Fable 5 limit\nmore",
            stderr="",
        ),
    )
    ok, detail = preflight.probe_model_access("claude_code", "claude-fable-5")
    assert ok is False and detail == "You've reached your Fable 5 limit"


# =============================================================================
# Rendering
# =============================================================================

def test_render_separates_blocking_from_advisory():
    rows = [
        Requirement("all good", True, "", "somewhere"),
        Requirement("hard stop", False, "do this", "config.key"),
        Requirement("nice to have", False, "optional thing", "other.key",
                    required=False),
    ]
    report = render("evolve", rows)
    assert "[FAIL] hard stop" in report
    assert "[warn] nice to have" in report
    assert "all good" not in report  # passing rows are not noise
    assert "1 of 3 requirements missing" in report

    passing = render("evolve", [Requirement("fine", True, "", "k")])
    assert "preflight passed" in passing


def test_dedupe_keeps_every_origin_and_the_worst_verdict():
    merged = dedupe([
        Requirement("claude CLI", True, "install", "a.model = m"),
        Requirement("claude CLI", False, "install", "b.model = m"),
    ])
    assert len(merged) == 1
    assert merged[0].ok is False
    assert "a.model = m" in merged[0].origin and "b.model = m" in merged[0].origin


def test_unknown_verb_fails_loud(packaged):
    with pytest.raises(ValueError, match="unknown verb"):
        requirements_for("evolve_but_typoed", packaged)
