"""The experience reference grammar and the sources it resolves to:
each form in its order, the errors with the forms and the `add` line,
`mine` pinned after a pull, a bank you read fetched with its record beside
the clone, the hosted gate's key, an index at its id."""

import json
import os
import shutil
from pathlib import Path

import pytest

from kapso.core.config import load_config
from kapso.experience.sources import experience_settings
from kapso.experience import (
    ExperienceError,
    ExperienceReferenceError,
    ExperienceStore,
    IndexRecord,
    parse_experience_reference,
    probe_hosted_gate,
    recorded_reference,
    resolve_source,
)
from kapso.learning.update_frame import init_bank
from tests.handles_support import git
from tests.test_bank_retriever import card_text


def settings_for(tmp_path: Path) -> dict:
    return {
        "banks": str(tmp_path / "homes" / "banks"),
        "indexes": str(tmp_path / "homes" / "indexes"),
        "bank_home": str(tmp_path / "bank.git"),
        "hosted": {"url": "https://mcp.leeroopedia.com/mcp?token={LEEROOPEDIA_API_KEY}", "tool_timeout_seconds": 600},
        "session_root": str(tmp_path / "serving"),
        "probe_budget": 1,
        "research_modes": ["idea", "implementation"],
        "research_depth": "deep",
        "search_top_k": 10,
    }


def bank_home(tmp_path: Path, name: str = "bank.git", cards: int = 0) -> Path:
    """A bank home with the founding skeleton and `cards` cards on main."""
    home = tmp_path / name
    init_bank(str(home))
    if cards:
        work = tmp_path / f"work-{name}"
        git(tmp_path, "clone", "-q", str(home), str(work))
        for index in range(cards):
            (work / ("insights" if index % 2 == 0 else "procedures") / f"card-{index}.md").write_text(f"# card {index}\n")
        git(work, "add", "-A")
        git(work, "commit", "-q", "-m", "cards")
        git(work, "push", "-q", "origin", "main")
        shutil.rmtree(work)
    return home


# ------------------------------------------------------------ the grammar

def test_the_forms_in_their_order(tmp_path):
    indexes = tmp_path / "indexes"
    (indexes / "org-x").mkdir(parents=True)
    (indexes / "org-x" / "index.json").write_text("{}")
    (tmp_path / "team" / "bank.git").mkdir(parents=True)
    parse = lambda text: parse_experience_reference(text, indexes_root=indexes, cwd=tmp_path)

    assert parse("mine").kind == "mine" and parse("leeroopedia").kind == "hosted"
    path = parse("./team/bank.git")
    assert path.kind == "path" and path.path == (tmp_path / "team" / "bank.git").resolve()
    shorthand = parse("acme/kapso-bank")
    assert (shorthand.kind, shorthand.url, shorthand.rev) == ("git", "https://github.com/acme/kapso-bank.git", "HEAD")
    assert parse("acme/kapso-bank@v2").rev == "v2"
    url = parse("git+https://github.com/acme/kapso-bank@main")
    assert (url.kind, url.url, url.rev) == ("git", "https://github.com/acme/kapso-bank", "main")
    scp = parse("git@github.com:acme/kapso-bank.git@v2")
    assert (scp.url, scp.rev, scp.slug) == ("git@github.com:acme/kapso-bank.git", "v2", "github.com-acme-kapso-bank")
    index = parse("org-x")
    assert index.kind == "index" and index.name == "org-x" and index.path == indexes / "org-x"
    # A bare name is never a path: the checkout an index was made from,
    # beside the run, cannot shadow the index; a bare name that is only a
    # directory names nothing.
    (tmp_path / "org-x").mkdir()
    assert parse("org-x").kind == "index"
    (tmp_path / "just-a-dir").mkdir()
    with pytest.raises(ExperienceReferenceError, match="is not a reserved name, a path, a git reference, or an index"):
        parse("just-a-dir")
    assert parse("./just-a-dir").kind == "path"


def test_what_matches_nothing_lists_the_forms_and_the_add_line(tmp_path):
    indexes = tmp_path / "indexes"
    with pytest.raises(ExperienceReferenceError, match="an empty experience reference"):
        parse_experience_reference("  ", indexes_root=indexes, cwd=tmp_path)
    with pytest.raises(ExperienceReferenceError) as caught:
        parse_experience_reference("org-y", indexes_root=indexes, cwd=tmp_path)
    message = str(caught.value)
    assert "`kapso experience add --repo <url> --name org-y`" in message and "the forms are:" in message
    assert "  mine " in message and "  leeroopedia " in message
    with pytest.raises(ExperienceReferenceError, match="not a reserved name, a git reference or an index name"):
        parse_experience_reference("Some Bank", indexes_root=indexes, cwd=tmp_path)
    # written as a path, it is a path: a missing one is never read as a git reference
    with pytest.raises(ExperienceReferenceError, match="a reference written as a path must exist"):
        parse_experience_reference("./ghost/bank@v2", indexes_root=indexes, cwd=tmp_path)


# ------------------------------------------------------------ the sources

def test_mine_is_pinned_at_its_head_with_its_cards_and_your_provenance(tmp_path):
    settings = settings_for(tmp_path)
    home = bank_home(tmp_path, cards=3)
    source = resolve_source(parse_experience_reference("mine", indexes_root=settings["indexes"]), settings=settings)
    assert (source.kind, source.resolved, source.provenance, source.scope, source.cards) == ("bank", str(home), "measured by you", "harness", 3)
    assert len(source.head) == 40
    assert source.ref().head == source.head and source.ref().kind == "bank"
    assert source.label() == f"mine at {source.head[:12]}, 3 cards, measured by you"
    every = resolve_source(parse_experience_reference("mine:all", indexes_root=settings["indexes"]), settings=settings)
    assert every.reference == "mine" and every.scope == "all" and every.label().endswith(", every scope")
    assert every.ref().scope == "all"
    with pytest.raises(ExperienceReferenceError, match="`:all` is for `mine` alone"):
        parse_experience_reference("acme/kapso-bank:all", indexes_root=settings["indexes"])


def test_mine_that_does_not_exist_names_the_add_line(tmp_path):
    settings = settings_for(tmp_path)
    with pytest.raises(ExperienceError, match="`kapso experience add mine` creates it"):
        resolve_source(parse_experience_reference("mine", indexes_root=settings["indexes"]), settings=settings)


def test_mine_with_a_remote_is_your_teams_too_and_loading_never_pulls(tmp_path):
    settings = settings_for(tmp_path)
    home = bank_home(tmp_path, cards=1)
    local = git(home, "rev-parse", "main")
    # The team's copy moves ahead; mine has it as origin.
    remote = tmp_path / "team.git"
    git(tmp_path, "clone", "-q", "--bare", str(home), str(remote))
    work = tmp_path / "work-team"
    git(tmp_path, "clone", "-q", str(remote), str(work))
    (work / "insights" / "from-the-team.md").write_text("# a colleague's card\n")
    git(work, "add", "-A")
    git(work, "commit", "-q", "-m", "the team's card")
    git(work, "push", "-q", "origin", "main")
    ahead = git(work, "rev-parse", "main")
    git(home, "remote", "add", "origin", str(remote))

    # pinned as it is on disk: the team's commit waits for `update`
    source = resolve_source(parse_experience_reference("mine", indexes_root=settings["indexes"]), settings=settings)
    assert source.head == local != ahead and source.cards == 1
    assert source.provenance == "measured by you and your team"


def test_a_bank_by_path_is_read_in_place_with_every_scope(tmp_path):
    settings = settings_for(tmp_path)
    home = bank_home(tmp_path, name="team-bank.git", cards=2)
    source = resolve_source(parse_experience_reference(str(home), indexes_root=settings["indexes"]), settings=settings)
    assert (source.kind, source.provenance, source.scope, source.cards, source.resolved) == ("bank", "measured by others", "all", 2, str(home))
    plain = tmp_path / "not-a-bank"
    plain.mkdir()
    with pytest.raises(ExperienceError, match="is not a bank: a bank is a git repository whose main branch holds index.md"):
        resolve_source(parse_experience_reference(str(plain), indexes_root=settings["indexes"]), settings=settings)


def test_a_bank_by_git_reference_is_fetched_once_with_its_record_beside_the_clone(tmp_path):
    settings = settings_for(tmp_path)
    home = bank_home(tmp_path, name="acme-bank.git", cards=4)
    reference = parse_experience_reference(f"git+file://{home}@main", indexes_root=settings["indexes"])
    source = resolve_source(reference, settings=settings)
    store = ExperienceStore(settings["banks"], settings["indexes"])
    clone = store.clone_dir(reference)
    assert source.path == clone and clone.parent == Path(settings["banks"])
    assert source.cards == 4 and source.provenance == "measured by others" and source.scope == "all"
    record = json.loads(store.record_path(reference).read_text())
    assert record["rev"] == "main" and record["commit"] == source.head and record["remote"] == f"file://{home}"
    assert not list(clone.glob("*.json"))
    # Fetched once: a second load reads the clone, and the listing shows it.
    marker = store.record_path(reference).stat().st_mtime_ns
    assert resolve_source(reference, settings=settings).head == source.head
    assert store.record_path(reference).stat().st_mtime_ns == marker
    assert [bank["remote"] for bank in store.banks()] == [f"file://{home}"]


def test_the_hosted_gate_needs_its_key(tmp_path, monkeypatch):
    settings = settings_for(tmp_path)
    reference = parse_experience_reference("leeroopedia", indexes_root=settings["indexes"])
    monkeypatch.delenv("LEEROOPEDIA_API_KEY", raising=False)
    with pytest.raises(ExperienceError, match="leeroopedia needs LEEROOPEDIA_API_KEY: put LEEROOPEDIA_API_KEY=kpsk_... in .env"):
        resolve_source(reference, settings=settings, dotenv_path="")
    monkeypatch.setenv("LEEROOPEDIA_API_KEY", "kpsk_test")
    source = resolve_source(reference, settings=settings, dotenv_path="")
    assert (source.kind, source.head, source.provenance, source.scope) == ("hosted", None, "reported by others", "all")
    assert source.resolved == settings["hosted"]["url"] and "kpsk_test" not in source.resolved
    assert source.label() == "leeroopedia (hosted), reported by others"
    # The key in .env beside the run is enough too; an empty one is absent.
    monkeypatch.delenv("LEEROOPEDIA_API_KEY")
    env_file = tmp_path / ".env"
    env_file.write_text("LEEROOPEDIA_API_KEY=kpsk_from_env\n")
    assert resolve_source(reference, settings=settings, dotenv_path=str(env_file)).kind == "hosted"
    env_file.write_text("LEEROOPEDIA_API_KEY=\n")
    with pytest.raises(ExperienceError, match="leeroopedia needs LEEROOPEDIA_API_KEY"):
        resolve_source(reference, settings=settings, dotenv_path=str(env_file))


def test_the_hosted_probe_initialises_the_gate_with_the_key_and_reads_either_answer_shape(tmp_path, monkeypatch):
    # Regression: `add leeroopedia` is the one-time probe — the key goes
    # on the URL, an MCP initialize is sent, the server's name comes back
    # from a JSON or an event-stream answer; a refusal and an empty key
    # stop the add with the fixing line.
    settings = settings_for(tmp_path)
    calls = []

    def post(url, payload, timeout_seconds):
        calls.append((url, payload["method"], timeout_seconds))
        if "sse" in url:
            return 200, "text/event-stream", 'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"serverInfo":{"name":"Leeroopedia"}}}\n\n'
        if "refuse" in url:
            return 401, "application/json", "{}"
        return 200, "application/json", '{"jsonrpc":"2.0","id":1,"result":{"serverInfo":{"name":"Leeroopedia"}}}'

    monkeypatch.setenv("LEEROOPEDIA_API_KEY", "kpsk_test")
    assert probe_hosted_gate(settings, dotenv_path="", post=post) == "Leeroopedia"
    assert calls[-1] == ("https://mcp.leeroopedia.com/mcp?token=kpsk_test", "initialize", 600)
    sse = {**settings, "hosted": {**settings["hosted"], "url": "https://sse.example/mcp?token={LEEROOPEDIA_API_KEY}"}}
    assert probe_hosted_gate(sse, dotenv_path="", post=post) == "Leeroopedia"
    refusing = {**settings, "hosted": {**settings["hosted"], "url": "https://refuse.example/mcp?token={LEEROOPEDIA_API_KEY}"}}
    with pytest.raises(ExperienceError, match=r"leeroopedia refused the probe \(HTTP 401\)"):
        probe_hosted_gate(refusing, dotenv_path="", post=post)
    monkeypatch.setenv("LEEROOPEDIA_API_KEY", "")
    with pytest.raises(ExperienceError, match="leeroopedia needs LEEROOPEDIA_API_KEY"):
        probe_hosted_gate(settings, dotenv_path="", post=post)


def test_a_bank_at_a_tag_is_pinned_at_the_tag_not_at_main(tmp_path):
    # Regression: the store checks the revision out; the head recorded and
    # served must be that revision's commit, not main's, which moved on.
    settings = settings_for(tmp_path)
    home = bank_home(tmp_path, name="acme-bank.git", cards=2)
    tagged = git(tmp_path, "--git-dir", str(home), "rev-parse", "main").strip()
    git(tmp_path, "--git-dir", str(home), "tag", "v1", tagged)
    work = tmp_path / "work-acme"
    git(tmp_path, "clone", "-q", str(home), str(work))
    (work / "insights" / "later.md").write_text(card_text("later"))
    git(work, "add", "-A")
    git(work, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "later")
    git(work, "push", "-q", "origin", "main")
    assert git(tmp_path, "--git-dir", str(home), "rev-parse", "main").strip() != tagged
    reference = parse_experience_reference(f"git+file://{home}@v1", indexes_root=settings["indexes"])
    source = resolve_source(reference, settings=settings)
    assert source.head == tagged and source.cards == 2
    store = ExperienceStore(settings["banks"], settings["indexes"])
    assert json.loads(store.record_path(reference).read_text())["commit"] == tagged


def test_a_recorded_reference_re_loads_the_same_source_from_anywhere():
    # Regression: a bank read by path is recorded at its absolute
    # location, so a resume or a reply from another directory finds it;
    # `mine` at every scope is `mine:all`; a git reference stays itself.
    assert recorded_reference({"reference": "mine", "kind": "bank", "resolved": "/h/bank.git", "scope": "all"}) == "mine:all"
    assert recorded_reference({"reference": "mine", "kind": "bank", "resolved": "/h/bank.git", "scope": "harness"}) == "mine"
    assert recorded_reference({"reference": "./team/bank.git", "kind": "bank", "resolved": "/srv/team/bank.git", "scope": "all"}) == "/srv/team/bank.git"
    assert recorded_reference({"reference": "acme/kapso-bank@v2", "kind": "bank", "resolved": "/h/banks/x", "scope": "all"}) == "acme/kapso-bank@v2"
    assert recorded_reference({"reference": "org-x", "kind": "index", "resolved": "/h/indexes/org-x", "scope": "all"}) == "org-x"


def test_an_index_is_loaded_at_its_id(tmp_path):
    settings = settings_for(tmp_path)
    store = ExperienceStore(settings["banks"], settings["indexes"])
    store.write_index_record(IndexRecord(
        name="org-x", id="3f9a0c1d2e4b", created_at="2026-10-25T12:00:00+00:00", source="https://github.com/org/x",
        search_backend="kg_graph_search", backend_refs={"collection": "kapso_org_x"}, page_count=1204, path=store.index_dir("org-x"),
    ))
    source = resolve_source(parse_experience_reference("org-x", indexes_root=settings["indexes"]), settings=settings)
    assert (source.kind, source.head, source.cards, source.provenance, source.resolved) == ("index", "3f9a0c1d2e4b", 1204, "reported by others", str(store.index_dir("org-x")))
    assert source.label() == "org-x (index 3f9a0c1d2e4b, 1204 pages), reported by others"
    assert [record.name for record in store.indexes()] == ["org-x"]


def test_the_defaults_come_from_the_config():
    settings = experience_settings(load_config())
    assert settings["banks"].endswith("experience/banks") and settings["indexes"].endswith("experience/indexes")
    assert settings["bank_home"].endswith("bank.git") and settings["hosted"]["url"].startswith("https://mcp.leeroopedia.com/")
    assert settings["session_root"].endswith("serving")
