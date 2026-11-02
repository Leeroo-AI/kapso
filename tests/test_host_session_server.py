"""`kapso-mcp`: the bundled gates served to a host session — the engineer's
own coding-agent session — with the configuration derived from where the
session runs and from what `--experience` loads, the way get_mcp_config()
derives it for a campaign session.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

import kapso.gated_mcp.host_session as host_module
from kapso.execution.memories.repo_memory import RepoMemoryManager
from kapso.experience import (
    ExperienceError,
    ExperienceStore,
    IndexRecord,
    parse_experience_reference,
    resolve_source,
)
from kapso.gated_mcp.host_session import (
    DEFAULT_GATES,
    INDEX_GATES,
    campaign_directory,
    load_sources,
    prepare_host_gates,
    remove_session_dir,
    repository_root,
)
from kapso.gated_mcp.presets import resolve_gates
from kapso.core.config import load_config
from kapso.experience.sources import experience_settings
from tests.harness_support import fake_config
from tests.test_bank_retriever import card_text
from tests.test_experience_reference import settings_for
from tests.test_update_frame import seed_bank_home


def _settings(tmp_path):
    settings = settings_for(tmp_path)
    settings["bank_home"] = str(tmp_path / "bank-home.git")
    return settings


def _source(tmp_path, raw):
    settings = _settings(tmp_path)
    reference = parse_experience_reference(raw, indexes_root=Path(settings["indexes"]), cwd=tmp_path)
    return resolve_source(reference, settings=settings)


def _index(tmp_path, name):
    settings = _settings(tmp_path)
    store = ExperienceStore(settings["banks"], settings["indexes"])
    store.write_index_record(IndexRecord(
        name=name, id=f"id-{name}", created_at="2026-10-25T12:00:00+00:00", source="https://github.com/org/x",
        search_backend="kg_graph_search", backend_refs={"collection": name}, page_count=3, path=store.index_dir(name),
    ))
    return _source(tmp_path, name)


def _prepare(tmp_path, root, *, requested=DEFAULT_GATES, sources=(), embedding=None, harness_name=None):
    return prepare_host_gates(
        root=root,
        requested=list(requested),
        config=fake_config(models={"embedding": embedding}),
        sources=list(sources),
        session_dir=tmp_path / "sessions" / "s1",
        probe_budget=1,
        harness_name=harness_name,
    )


@pytest.fixture
def codex_on_path(monkeypatch):
    monkeypatch.setattr(host_module.shutil, "which", lambda name: "/usr/bin/codex")


def _git_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return path


# ------------------------------------------------------------ decisions


def test_an_empty_directory_gets_research_only_and_says_why(tmp_path, codex_on_path):
    root = tmp_path / "project"
    root.mkdir()
    gates = _prepare(tmp_path, root)
    assert gates.enabled == ["research", "campaign"] and gates.env == {}
    assert [note.split(":")[0] for note in gates.notes] == [
        "repo_memory", "experiment_history", "bank",
    ]


def test_without_the_codex_cli_research_is_off(tmp_path, monkeypatch):
    monkeypatch.setattr(host_module.shutil, "which", lambda name: None)
    gates = _prepare(tmp_path, tmp_path, requested=["research"])
    assert gates.enabled == [] and gates.notes == ["research: off (the codex CLI is not on PATH)"]


def test_repo_memory_rides_when_the_repository_has_one(tmp_path, codex_on_path):
    repo = _git_repo(tmp_path / "repo")
    memory = repo / RepoMemoryManager.MEMORY_REL_PATH
    memory.parent.mkdir(parents=True)
    memory.write_text(json.dumps({"schema_version": 2}))
    inside = repo / "src" / "pkg"
    inside.mkdir(parents=True)

    gates = _prepare(tmp_path, inside, requested=["repo_memory"])
    assert gates.enabled == ["repo_memory"]
    assert gates.env == {"REPO_MEMORY_ROOT": str(repo.resolve())}
    assert repository_root(inside) == repo.resolve()
    assert repository_root(tmp_path / "nowhere-git") == tmp_path / "nowhere-git"


def test_experiment_history_rides_inside_a_campaign_directory(tmp_path, codex_on_path):
    campaign = tmp_path / "campaign"
    (campaign / ".kapso").mkdir(parents=True)
    (campaign / ".kapso" / "experiment_history.json").write_text("[]")
    deep = campaign / "src"
    deep.mkdir()
    assert campaign_directory(deep) == campaign
    assert campaign_directory(tmp_path) is None

    gates = _prepare(
        tmp_path, deep, requested=["experiment_history"],
        embedding="text-embedding-3-small",
    )
    assert gates.enabled == ["experiment_history"]
    assert gates.env == {
        "EXPERIMENT_HISTORY_PATH": str(campaign / ".kapso" / "experiment_history.json"),
        "EXPERIMENT_EMBEDDING_MODEL": "text-embedding-3-small",
    }


def test_the_bank_is_staged_for_the_session_when_one_is_loaded(tmp_path, codex_on_path):
    seed_bank_home(tmp_path, {"a-card": card_text("a-card")})
    gates = _prepare(tmp_path, tmp_path, requested=["bank"], sources=[_source(tmp_path, "mine")])
    assert gates.enabled == ["bank"]
    env = gates.env
    session_dir = tmp_path / "sessions" / "s1"
    [entry] = json.loads(env["KAPSO_BANKS"])
    clone = Path(entry["dir"])
    assert clone.parent == session_dir / "banks" and clone.name.startswith("mine-")
    assert (clone / "insights" / "a-card.md").is_file()
    assert (session_dir / "serving-record.yaml").is_file()
    assert env["KAPSO_SERVING_PULL_LOG"] == str(session_dir / "serving-pull.jsonl")
    # a host session has no harness: its coordinate is `none`; its pull
    # events carry the host's own label
    assert env["KAPSO_TASK_HARNESS"] == "none" and env["KAPSO_PROBE_BUDGET"] == "1"
    assert env["KAPSO_SESSION"] == "host"
    head = subprocess.run(
        ["git", "-C", entry["dir"], "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert entry["head"] == head and entry["provenance"] == "measured by you"
    # The bundled server resolves every served gate from exactly this env.
    resolution = resolve_gates(gates.enabled, policy="error", env={**os.environ, **env})
    assert resolution.unavailable_gates == ()


def test_the_harness_is_the_coordinate_and_the_intro_is_the_servers_instructions(tmp_path, codex_on_path):
    seed_bank_home(tmp_path, {"a-card": card_text("a-card")})
    gates = _prepare(
        tmp_path, tmp_path, requested=["bank"], sources=[_source(tmp_path, "mine")], harness_name="trading",
    )
    env = gates.env
    assert env["KAPSO_TASK_HARNESS"] == "trading"
    intro = Path(env["KAPSO_SERVING_INTRO"])
    assert intro == tmp_path / "sessions" / "s1" / "intro.md"
    assert "## Experience (measured before)" in intro.read_text()
    assert "1 card measured by you, under the trading harness" in intro.read_text()
    # the staging goes with the server
    remove_session_dir(tmp_path / "sessions" / "s1")
    assert not (tmp_path / "sessions" / "s1").exists()
    remove_session_dir(tmp_path / "sessions" / "s1")   # gone already: nothing to do


def test_the_bank_stays_off_while_no_bank_is_loaded(tmp_path, codex_on_path):
    seed_bank_home(tmp_path, {"a-card": card_text("a-card")})
    off = _prepare(tmp_path, tmp_path, requested=["bank"])
    assert off.enabled == [] and off.notes == ["bank: off (no bank loaded: `kapso experience add mine` makes yours; `session_sources: [mine:all]` under `experience` in kapso-config.yaml serves it here)"]
    assert not (tmp_path / "sessions").exists()


def test_an_index_adds_the_search_gates(tmp_path, codex_on_path):
    source = _index(tmp_path, "org-x")
    gates = _prepare(tmp_path, tmp_path, requested=list(INDEX_GATES), sources=[source])
    assert gates.enabled == ["idea", "code"]
    assert gates.env["KG_INDEX_PATH"] == str(Path(_settings(tmp_path)["indexes"]) / "org-x" / "index.json")
    # the intro names the corpus and rides as the server's instructions,
    # bank or no bank
    intro = Path(gates.env["KAPSO_SERVING_INTRO"]).read_text()
    assert "the `org-x` corpus (3 pages), reported by others" in intro and "bank_index()" not in intro
    assert set(gates.env) == {"KG_INDEX_PATH", "KAPSO_SERVING_INTRO"}
    without = _prepare(tmp_path, tmp_path, requested=list(INDEX_GATES))
    assert without.enabled == [] and without.notes == [
        "idea: off (no index loaded: `kapso experience add --repo <url> --name <name>` makes one; `session_sources` serves it here)",
        "code: off (no index loaded: `kapso experience add --repo <url> --name <name>` makes one; `session_sources` serves it here)",
    ]


def test_the_hosted_source_is_a_reminder_not_a_gate(tmp_path, codex_on_path, monkeypatch):
    monkeypatch.setenv("LEEROOPEDIA_API_KEY", "kpsk_test")
    gates = _prepare(tmp_path, tmp_path, requested=["research"], sources=[_source(tmp_path, "leeroopedia")])
    assert gates.enabled == ["research"] and set(gates.env) == {"KAPSO_SERVING_INTRO"}
    assert "Leeroopedia's pages, reported by others" in Path(gates.env["KAPSO_SERVING_INTRO"]).read_text()
    assert gates.notes == [
        "leeroopedia: a hosted server, not a bundled gate — add it to the host "
        "as an HTTP MCP server with your key on the URL"
    ]


def test_load_sources_resolves_each_reference_and_refuses_a_missing_one(tmp_path):
    seed_bank_home(tmp_path, {"a-card": card_text("a-card")})
    _index(tmp_path, "org-x")
    sources = load_sources(" mine, org-x ", _settings(tmp_path))
    assert [(s.reference, s.kind) for s in sources] == [("mine", "bank"), ("org-x", "index")]
    assert load_sources(None, _settings(tmp_path)) == []
    with pytest.raises(ExperienceError, match="kapso experience add mine"):
        load_sources("mine", {**_settings(tmp_path), "bank_home": str(tmp_path / "absent.git")})


def test_unknown_and_campaign_only_gates_are_configuration_errors(tmp_path, codex_on_path):
    with pytest.raises(ValueError, match="unknown gate"):
        _prepare(tmp_path, tmp_path, requested=["telepathy"])
    with pytest.raises(ValueError, match="campaign-only"):
        _prepare(tmp_path, tmp_path, requested=["inbox"])


def test_duplicates_are_dropped_and_order_kept(tmp_path, codex_on_path):
    gates = _prepare(tmp_path, tmp_path, requested=["research", "research"])
    assert gates.enabled == ["research"]


# ------------------------------------------------------------------ main


def test_main_takes_the_harness_for_its_name(tmp_path, monkeypatch, capsys):
    trading = Path(__file__).resolve().parent.parent / "examples" / "harnesses" / "trading"
    captured = {}
    monkeypatch.setattr(
        host_module, "prepare_host_gates",
        lambda **kwargs: captured.update(kwargs) or host_module.HostGates(enabled=["research"], env={}),
    )
    monkeypatch.setattr(host_module, "run_server", _no_server)
    monkeypatch.chdir(tmp_path)
    host_module.main(["--harness", str(trading)])
    assert captured["harness_name"] == "trading" and captured["sources"] == []
    assert "kapso-mcp: serving research for" in capsys.readouterr().err
    # the session directory is removed when the server exits
    assert captured["session_dir"].parent == Path(experience_settings(load_config())["session_root"]).expanduser()


async def _no_server(**kwargs):
    return None


def test_the_bundled_server_serves_the_staged_intro_as_its_instructions(tmp_path, monkeypatch):
    import kapso.gated_mcp.server as server_module

    intro = tmp_path / "intro.md"
    intro.write_text("## Experience (measured before)\n\nYou are served:\n- 1 card measured by you\n")
    built = {}

    class FakeServer:
        def __init__(self, name, instructions=None):
            built.update(name=name, instructions=instructions)

        def list_tools(self):
            return lambda fn: fn

        def call_tool(self):
            return lambda fn: fn

    monkeypatch.setattr(server_module, "Server", FakeServer)
    monkeypatch.setattr(server_module, "_resolve_configuration", lambda gate_params=None: {})
    monkeypatch.setenv("KAPSO_SERVING_INTRO", str(intro))
    server_module.create_gated_mcp_server()
    assert built == {"name": "gated-knowledge", "instructions": intro.read_text()}
    monkeypatch.delenv("KAPSO_SERVING_INTRO")
    server_module.create_gated_mcp_server()
    assert built["instructions"] is None


def test_main_serves_what_it_can_and_connects_with_every_other_gate_off(tmp_path, monkeypatch, capsys):
    served = {}

    async def fake_server(**kwargs):
        served.update(gates=os.environ["MCP_ENABLED_GATES"],
                      policy=os.environ["MCP_GATE_FAILURE_POLICY"], **kwargs)

    monkeypatch.setattr(host_module, "run_server", fake_server)
    monkeypatch.setattr(host_module.shutil, "which", lambda name: "/usr/bin/codex")
    monkeypatch.chdir(tmp_path)
    for name in ("MCP_ENABLED_GATES", "MCP_GATE_FAILURE_POLICY"):
        monkeypatch.delenv(name, raising=False)

    host_module.main([])
    assert served["gates"] == "research,campaign" and served["policy"] == "error"
    assert served["gate_params"] == {
        "campaign": {"config_path": None, "cwd": str(tmp_path)},
        "idea": {"config_path": None}, "code": {"config_path": None},
    }
    assert served["instructions"].startswith(f"## Kapso tools in this session (kapso-mcp {host_module.__version__})\n- on: research, campaign\n")
    assert "- bank: off (no bank loaded: `kapso experience add mine` makes yours" in served["instructions"]
    err = capsys.readouterr().err
    assert err.startswith(f"kapso-mcp {host_module.__version__} (config: the packaged defaults)\n")
    assert "kapso-mcp: serving research, campaign for" in err and "bank: off" in err

    # No codex, no bank, no repository: the server still connects, with
    # the campaign tools and every reason in its brief.
    monkeypatch.setattr(host_module.shutil, "which", lambda name: None)
    host_module.main([])
    assert served["gates"] == "campaign"
    assert "- research: off (the codex CLI is not on PATH)" in served["instructions"]


def test_the_config_lists_what_a_session_is_served_and_a_missing_source_is_a_note(tmp_path):
    settings = _settings(tmp_path)
    index = _index(tmp_path, "org-x")
    settings["session_sources"] = ["mine", "org-x"]
    sources, notes = host_module.config_sources(settings)
    assert [source.reference for source in sources] == [index.reference]
    assert notes == [
        "mine: not found, so its gate is off (`kapso experience add mine` makes it, as a first learn does); it is listed under experience.session_sources",
    ]
    settings["session_sources"] = "mine"
    with pytest.raises(ExperienceError, match="experience.session_sources must be a list of references"):
        host_module.config_sources(settings)
    settings["session_sources"] = ["mine", 3]
    with pytest.raises(ExperienceError, match="must be a list"):
        host_module.config_sources(settings)


def test_main_reads_the_projects_config_and_its_sources_unless_experience_is_passed(tmp_path, monkeypatch, capsys):
    settings = _settings(tmp_path)
    _index(tmp_path, "org-x")
    (tmp_path / "kapso-config.yaml").write_text(
        f"experience:\n  banks: {settings['banks']}\n  indexes: {settings['indexes']}\n  session_sources: [org-x, mine]\n"
        f"learning:\n  bank:\n    local_path: {settings['bank_home']}\n",
        encoding="utf-8",
    )
    captured = {}
    monkeypatch.setattr(
        host_module, "prepare_host_gates",
        lambda **kwargs: captured.update(kwargs) or host_module.HostGates(enabled=["campaign"], env={}),
    )
    monkeypatch.setattr(host_module, "run_server", _no_server)
    monkeypatch.chdir(tmp_path)

    host_module.main([])
    assert [source.reference for source in captured["sources"]] == ["org-x"]
    assert captured["requested"] == [*DEFAULT_GATES, *INDEX_GATES]
    err = capsys.readouterr().err
    assert err.startswith(f"kapso-mcp {host_module.__version__} (config: {tmp_path / 'kapso-config.yaml'})\n")
    assert "kapso-mcp: mine: not found, so its gate is off" in err

    # Explicit, so the list is replaced: an empty string loads nothing.
    host_module.main(["--experience", ""])
    assert captured["sources"] == [] and captured["requested"] == list(DEFAULT_GATES)
    assert "not found" not in capsys.readouterr().err
