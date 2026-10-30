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

from kapso.core.config import ConfigError, load_config
import kapso.core.preflight as preflight
import kapso.kapso as kapso_module
from kapso.kapso import Kapso
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
    for name in ("OPENAI_API_KEY", "GH_TOKEN", "LEEROOPEDIA_API_KEY"):
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

def test_all_codex_build_never_asks_for_claude(packaged, full_machine):
    """The bug this design replaced: the old doctor demanded `claude` from
    everyone, including a config that can only ever spawn codex."""
    config = copy.deepcopy(packaged)
    config["search_strategy"]["params"]["host"] = "codex"

    found = labels(requirements_for("build", config))
    assert "codex CLI" in found
    assert "claude CLI" not in found
    assert "claude authenticated" not in found


def test_research_needs_codex_only_when_asked_and_learn_needs_both_clis(packaged, full_machine):
    """Not a coincidence worth losing: the grading crew is deliberately
    cross-model (report_writer on codex, verifier on claude), so `learn`
    genuinely needs both CLIs; ingesting a corpus asks for the web-search
    session's CLI only when a question is researched."""
    researched = labels(requirements_for("experience", packaged, research=True))
    assert "codex CLI" in researched
    assert "codex CLI" not in labels(requirements_for("experience", packaged))

    learn = labels(requirements_for("learn", packaged))
    assert "codex CLI" in learn and "claude CLI" in learn


def test_ingesting_a_corpus_needs_the_stores_and_the_embeddings_key(packaged, full_machine):
    """An index is written into both graph stores and embeds every page:
    the stores and the key are requirements of `add_experience`, and of
    nothing else a machine without an index runs."""
    ingesting = labels(requirements_for("experience", packaged))
    assert "Weaviate (localhost:8080)" in ingesting
    assert "Neo4j (localhost:7687)" in ingesting
    assert "OPENAI_API_KEY (embeddings: https://api.openai.com/v1)" in ingesting
    assert "Weaviate (localhost:8080)" not in labels(requirements_for("learn", packaged))


def test_kg_store_rows_name_the_configured_endpoints(packaged, full_machine):
    """A test stack beside production (other ports) is what the config names,
    so the rows must check those endpoints, not localhost:8080/7687."""
    import copy
    config = copy.deepcopy(packaged)
    config["knowledge_search"]["params"] = {
        "weaviate_url": "http://localhost:18080",
        "neo4j_uri": "bolt://localhost:17687",
    }
    merging = labels(requirements_for("experience", config))
    assert "Weaviate (localhost:18080)" in merging
    assert "Neo4j (localhost:17687)" in merging
    assert "Weaviate (localhost:8080)" not in merging


def test_codify_target_decides_whether_gcloud_is_required(packaged, full_machine):
    """`learning.codify.target: gcp_ephemeral` shells out to gcloud — a
    requirement no user would guess, and one that must vanish when the
    target is local."""
    assert "gcloud CLI" in labels(requirements_for("learn", packaged))

    local = copy.deepcopy(packaged)
    local["learning"]["codify"]["target"] = "local"
    assert "gcloud CLI" not in labels(requirements_for("learn", local))


def test_kg_rows_appear_only_when_an_index_is_connected(packaged, full_machine):
    """The stores are a build's requirement exactly when an index is
    loaded; a campaign without one must not be blocked on stores it never
    opens."""
    without = labels(requirements_for("build", packaged))
    assert "Neo4j (localhost:7687)" not in without

    with_index = labels(
        requirements_for("build", packaged, index="org-x")
    )
    assert "Neo4j (localhost:7687)" in with_index
    assert (
        "OPENAI_API_KEY (embeddings: https://api.openai.com/v1)" in with_index
    )


def test_caller_passed_coding_agent_overrides_the_mode(packaged, full_machine):
    """build(coding_agent=...) changes which CLI actually runs, so the
    requirement must follow the argument, not the config's default."""
    found = labels(requirements_for("build", packaged, coding_agent="codex"))
    assert "codex CLI" in found


# =============================================================================
# Gates — the config's own failure policy decides whether a gap blocks
# =============================================================================

def test_gate_policy_decides_whether_a_missing_gate_blocks(
    packaged, full_machine, monkeypatch,
):
    monkeypatch.delenv("LEEROOPEDIA_API_KEY", raising=False)
    # The packaged lists name no gate the user sets up (the hosted gate
    # rides by loading, its key checked then); a config that names one
    # in its lists is checked under its own policy.
    assert not [
        label for label in by_label(requirements_for("build", packaged))
        if label.startswith("MCP gate")
    ]
    packaged = copy.deepcopy(packaged)
    params = packaged["search_strategy"]["params"]
    params["ideation"]["gates"] = list(params["ideation"]["gates"]) + ["leeroopedia"]

    warn = by_label(requirements_for("build", packaged))
    assert warn["MCP gate 'leeroopedia'"].required is False
    # Hosted gate: the fix is the key alone, never an install step.
    assert "LEEROOPEDIA_API_KEY" in warn["MCP gate 'leeroopedia'"].fix
    assert "install" not in warn["MCP gate 'leeroopedia'"].fix.replace(
        "nothing to install", ""
    )

    strict = copy.deepcopy(packaged)
    strict["search_strategy"]["params"]["gate_failure_policy"] = "error"
    blocking = by_label(requirements_for("build", strict))
    assert blocking["MCP gate 'leeroopedia'"].required is True
    # `error` is the one policy where downgrading is a real option, so it
    # is the one that says so.
    assert "gate_failure_policy: warn" in blocking["MCP gate 'leeroopedia'"].fix


def test_platform_injected_gate_env_is_never_reported_as_user_setup(
    packaged, full_machine,
):
    """KAPSO_BANKS / EXPERIMENT_HISTORY_PATH / KG_INDEX_PATH are set by
    Kapso when it launches a session. Reporting them at preflight would be
    a guaranteed false positive on every clean machine — and the packaged
    config names the experiment_history gate, so this fires by default."""
    found = labels(requirements_for("build", packaged))
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
    for name in ("OPENAI_API_KEY", "GH_TOKEN", "LEEROOPEDIA_API_KEY"):
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

def test_session_specs_resolve_the_cli_three_ways():
    """An explicit `cli`, a `type` naming a registered agent, and the
    model-name prefix; the build's own sessions are roles over the hosts
    table and do not walk this. A rename that dodges this silently probes
    the wrong CLI."""
    specs = session_specs({
        "explicit": {"cli": "claude_code", "model": "some-house-model"},
        "typed": {"type": "codex", "model": "claude-shaped-name"},
        "prefixed": {"model": "claude-opus-5"},
        "fallback": {"model": "gpt-5.6-sol"},
    }, "root")
    resolved = {spec.model: spec.cli for spec in specs}
    assert resolved["some-house-model"] == "claude_code"
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
    assert "kapso experience share" in doctor["bank remote (sharing)"].fix

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
        preflight.run_preflight("experience", packaged, research=True)
    message = str(excinfo.value)
    assert "kapso experience — preflight failed" in message
    assert "npm install -g @openai/codex" in message
    assert "kapso doctor experience" in message


def test_disabling_preflight_skips_every_probe(packaged, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("preflight probed while disabled")

    monkeypatch.setattr(preflight.shutil, "which", explode)
    config = copy.deepcopy(packaged)
    config["preflight"] = {"enabled": False}
    assert preflight.run_preflight("build", config) == []


def test_settings_fall_back_to_the_packaged_defaults(packaged):
    """A config written before the `preflight:` block existed still gets
    the shipped behaviour rather than a silently-off preflight."""
    settings = preflight.preflight_settings({})
    assert settings["enabled"] is True
    assert settings["live_model_probe"] is False

    partial = preflight.preflight_settings({"preflight": {"live_model_probe": True}})
    assert partial["enabled"] is True
    assert partial["live_model_probe"] is True


def test_live_probe_runs_only_when_configured(packaged, full_machine, monkeypatch):
    # the full machine, so the static tier passes everywhere (the index's
    # embeddings key included: a runner without one failed here)
    probed = []

    def fake_probe(cli, model):
        probed.append((cli, model))
        return True, ""

    monkeypatch.setattr(preflight, "probe_model_access", fake_probe)

    preflight.run_preflight("experience", packaged, research=True)
    assert probed == []

    config = copy.deepcopy(packaged)
    config["preflight"] = {"enabled": True, "live_model_probe": True}
    preflight.run_preflight("experience", config, research=True)
    assert ("codex", "gpt-5.6-sol") in probed


def test_live_probe_surfaces_the_clis_own_cap_message(packaged, monkeypatch):
    """Onboarding E2E finding #5: the CLI's own words are the useful part —
    "You've reached your Fable 5 limit" tells the user far more than a
    generic failure would."""
    monkeypatch.setattr(preflight.shutil, "which", lambda name, *a, **k: f"/b/{name}")
    monkeypatch.setattr(
        preflight, "probe_model_access",
        lambda cli, model: (False, "You've reached your Fable 5 limit"),
    )
    rows = preflight.live_model_requirements("experience", packaged, research=True)
    assert rows and all(row.ok is False for row in rows)
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
    report = render("build", rows)
    assert "[FAIL] hard stop" in report
    assert "[warn] nice to have" in report
    assert "all good" not in report  # passing rows are not noise
    assert "1 of 3 requirements missing" in report

    passing = render("build", [Requirement("fine", True, "", "k")])
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
        requirements_for("build_but_typoed", packaged)


def test_api_agent_checks_configured_keys_per_session(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PROVIDER_KEY", "test-key")
    specs = preflight.session_specs(
        {
            "writer": {
                "type": "openai_compatible",
                "model": "provider/model",
                "agent_specific": {"api_key_env": "PROVIDER_KEY"},
            },
            "local": {
                "type": "openai_compatible",
                "model": "local/model",
                "agent_specific": {"allow_missing_api_key": True},
            },
        },
        inference=True,
    )
    found = {r.label: r for r in preflight.cli_requirements(specs)}
    assert "OPENAI_API_KEY" not in found
    assert "node" not in found
    assert found["PROVIDER_KEY (https://api.openai.com/v1)"].ok
    monkeypatch.delenv("PROVIDER_KEY")
    found = {r.label: r for r in preflight.cli_requirements(specs)}
    assert not found["PROVIDER_KEY (https://api.openai.com/v1)"].ok


def test_research_preflight_inherits_nested_api_options(monkeypatch):
    monkeypatch.setenv("PROVIDER_KEY", "test-key")
    config = {
        "inference": {
            "default": {
                "cli": "openai_compatible",
                "model": "provider/model",
                "agent_specific": {"api_key_env": "PROVIDER_KEY"},
            },
            "roles": {"research": {"agent_specific": {"temperature": 0.2}}},
        }
    }
    found = {r.label: r for r in preflight.research_requirements(config)}
    assert found["PROVIDER_KEY (https://api.openai.com/v1)"].ok
    assert "OPENAI_API_KEY" not in found


def test_research_rejects_api_agent_without_web_tools():
    config = {
        "inference": {
            "default": {
                "cli": "openai_compatible",
                "model": "local/model",
                "agent_specific": {"allow_missing_api_key": True},
            }
        }
    }
    found = {r.label: r for r in preflight.research_requirements(config)}
    assert not found["research agent supports web search"].ok


def test_build_checks_api_inference_role_key(
    packaged, full_machine, monkeypatch
):
    monkeypatch.delenv("RERANK_KEY", raising=False)
    packaged["inference"]["roles"]["kg_rerank"] = {
        "cli": "openai_compatible",
        "model": "provider/model",
        "agent_specific": {"api_key_env": "RERANK_KEY"},
    }
    found = {
        r.label: r
        for r in requirements_for(
            "build",
            packaged,
            index="my-index",
        )
    }
    assert not found["RERANK_KEY (https://api.openai.com/v1)"].ok
    assert "inference.roles.kg_rerank" in found[
        "RERANK_KEY (https://api.openai.com/v1)"
    ].origin


def test_doctor_keeps_endpoints_separate_for_the_same_key(monkeypatch):
    monkeypatch.setenv("PROVIDER_KEY", "secret-not-for-display")
    specs = [preflight.SessionSpec(
        cli="openai_compatible", model="m", origin="inference.roles." + role,
        inference=True,
        agent_specific={"base_url": endpoint, "api_key_env": "PROVIDER_KEY"},
    ) for role, endpoint in [
        ("kg_rerank", "https://first.test/v1"),
        ("repo_memory", "https://second.test/v1"),
    ]]
    rows = preflight.dedupe(preflight.cli_requirements(specs))
    labels = [row.label for row in rows]
    assert "PROVIDER_KEY (https://first.test/v1)" in labels
    assert "PROVIDER_KEY (https://second.test/v1)" in labels
    assert "secret-not-for-display" not in repr(rows)


def test_preflight_rejects_plaintext_inference_endpoint():
    spec = preflight.SessionSpec(
        cli="openai_compatible", model="m", origin="inference.default",
        inference=True,
        agent_specific={"base_url": "http://remote.test/v1"},
    )
    with pytest.raises(ValueError, match="HTTPS"):
        preflight.cli_requirements([spec])


def test_preflight_uses_embedding_endpoint_from_config(packaged, full_machine):
    packaged["knowledge_search"]["params"] = {
        "embedding_base_url": "https://embeddings.test/v1",
    }
    rows = requirements_for("build", packaged, index="my-index")
    assert any(row.label ==
               "OPENAI_API_KEY (embeddings: https://embeddings.test/v1)"
               for row in rows)


@pytest.mark.parametrize("role", ["implementation", "judge"])
@pytest.mark.parametrize("with_model", [False, True])
def test_build_refuses_inference_agent_in_config(
    packaged, full_machine, role, with_model,
):
    """A role on an inference-only adapter (a host entry names it, the
    role picks it) is refused by the role's key: the session must write."""
    packaged["hosts"]["openai_compatible"] = {"model": "provider/model"}
    packaged["search_strategy"]["params"][role]["cli"] = "openai_compatible"
    if with_model:
        packaged["search_strategy"]["params"][role]["model"] = "provider/own-model"
    with pytest.raises(PreflightError) as error:
        preflight.run_preflight("build", packaged)
    failure = next(
        row for row in error.value.requirements
        if "inference-only" in row.label
    )
    assert failure.required and not failure.ok
    assert f"search_strategy.params.{role}" in failure.origin
    assert "coding-capable" in failure.fix
    assert "inference.roles" in failure.fix


@pytest.mark.preflight
def test_a_host_the_table_does_not_name_fails_before_workspace(
    packaged, full_machine, tmp_path, monkeypatch,
):
    """`--coding-agent` picks a host of the config's table; a name the
    table does not carry is refused by name, with the way in, before the
    workspace exists (a named host on an inference-only adapter is
    test_build_refuses_inference_agent_in_config's refusal)."""
    kapso = object.__new__(Kapso)
    # The launch is registered before its checks (wp6-observability.md,
    # change 8): the registry is a file of the test's own.
    packaged["inbox"]["registry"] = str(tmp_path / "registry.jsonl")
    kapso._config = packaged
    kapso.config_path = None
    kapso._harness = None
    kapso._harness_settings = packaged["harness"]
    kapso._experience = []
    kapso._bank_home = None
    output = tmp_path / "campaign"
    with pytest.raises(ValueError, match=r"search_strategy\.params\.host .*openai_compatible is not in hosts"):
        kapso.build(
            "test", coding_agent="openai_compatible",
            output=str(output),
        )
    assert not output.exists()


@pytest.mark.parametrize("verb,branch", [
    ("build", "search_strategy.params"),
    ("experience", "learner.ingestor"),
    ("learn", "mining"),
])
def test_other_non_inference_sessions_are_rejected(
    packaged, full_machine, verb, branch,
):
    if verb == "build":
        # The build's sessions are roles over the hosts table.
        packaged["hosts"]["openai_compatible"] = {"model": "provider/model"}
        packaged["search_strategy"]["params"]["host"] = "openai_compatible"
    else:
        node = packaged["learning"] if verb == "learn" else packaged
        for key in branch.split("."):
            node = node.setdefault(key, {})
        node["cli"] = "openai_compatible"
        node["model"] = "provider/model"
    with pytest.raises(PreflightError, match="inference-only"):
        preflight.run_preflight(verb, packaged)


def test_the_configured_target_is_a_required_row_with_its_probe(packaged, full_machine, monkeypatch):
    """`deployment.target: docker` on a machine without docker: the
    doctor's row is required and names the key; with docker present, the
    daemon's probe is a row of its own."""
    packaged["deployment"]["target"] = "docker"
    monkeypatch.setattr(preflight.shutil, "which", lambda name, *a, **k: None if name == "docker" else f"/usr/bin/{name}")
    rows = {row.label: row for row in preflight.deploy_requirements(packaged)}
    assert rows["docker (deploy target)"].required and not rows["docker (deploy target)"].ok
    assert rows["docker (deploy target)"].origin == "deployment.target = docker"
    assert "docker: `docker info` answers" not in rows and not rows["deploy target modal"].required
    monkeypatch.setattr(preflight.shutil, "which", lambda name, *a, **k: f"/usr/bin/{name}")
    monkeypatch.setattr(preflight, "probe_failure", lambda skill: "`docker info` fails: start Docker Desktop, or: sudo systemctl start docker")
    rows = {row.label: row for row in preflight.deploy_requirements(packaged, target="docker")}
    probe = rows["docker: `docker info` answers"]
    assert probe.required and not probe.ok and probe.origin == "deploy(target='docker')" and probe.fix == "start Docker Desktop, or: sudo systemctl start docker"
    assert "deploy target modal" not in rows   # a call in flight lists no advisory rows


@pytest.mark.parametrize("override", [False, True])
def test_deploy_rejects_an_inference_only_host_by_key(packaged, full_machine, override):
    """An inference-only adapter is never in the hosts table (it has no
    staging), so the deployment session's host is refused by the key that
    named it, from the config or the call."""
    context = {}
    if override:
        context["coding_agent"] = "openai_compatible"
    else:
        packaged["deployment"]["cli"] = "openai_compatible"
    with pytest.raises(ConfigError, match="deployment.cli: openai_compatible is not in hosts"):
        preflight.run_preflight("deploy", packaged, **context)


def test_text_inference_roles_remain_allowed(packaged, full_machine):
    packaged["inference"]["default"].update({
        "cli": "openai_compatible", "model": "provider/model",
    })
    rows = preflight.run_preflight("build", packaged)
    assert not any("inference-only" in row.label for row in rows)
    assert any("OPENAI_API_KEY (https://api.openai.com/v1)" == row.label
               for row in rows)


def test_inference_like_path_does_not_grant_capability():
    specs = session_specs(
        {"type": "openai_compatible"}, "inference.roles.fake",
    )
    assert any(not row.ok and "inference-only" in row.label
               for row in preflight.cli_requirements(specs))


def test_build_asks_for_the_openai_key_only_when_something_will_embed(packaged, full_machine, monkeypatch):
    """The shipped config leaves `models.embedding` null, so a stock campaign
    embeds nothing and doctor must not ask for the key (it used to pass and
    the campaign died at the first node's bookkeeping). Naming a model turns
    the row on, with the store as its reason."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    stock = by_label(requirements_for("build", packaged))
    assert not [label for label in stock if label.startswith("OPENAI_API_KEY")]

    embedding_on = copy.deepcopy(packaged)
    embedding_on["models"] = {"embedding": "text-embedding-3-small"}
    rows = by_label(requirements_for("build", embedding_on))
    row = rows["OPENAI_API_KEY (embeddings: https://api.openai.com/v1)"]
    assert row.ok is False and row.required is True
    assert "experiment store" in row.origin and "text-embedding-3-small" in row.origin


def test_build_finds_a_missing_notify_command_at_launch(packaged, full_machine, monkeypatch):
    monkeypatch.setattr(
        preflight.shutil, "which",
        lambda name, *a, **k: None if name == "/opt/hooks/ping.sh" else f"/usr/bin/{name}",
    )
    notifying = copy.deepcopy(packaged)
    notifying["notify"].update(events=["waiting_for_user"], command=["/opt/hooks/ping.sh"])
    row = by_label(requirements_for("build", notifying))["notify command /opt/hooks/ping.sh"]
    assert row.ok is False and row.required is True and row.origin == "notify.command"

    notifying["notify"]["events"] = []
    assert "notify command /opt/hooks/ping.sh" not in by_label(requirements_for("build", notifying))
