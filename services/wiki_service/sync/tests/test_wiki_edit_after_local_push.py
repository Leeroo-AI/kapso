"""A wiki edit made right after a local push must still reach the file.

After pushing a local file, the service ignored the next wiki change for
that page, taking it for the echo of its own push. The poller hands over
only the newest revision per page, so when a person edited the page
within one poll interval, their revision was the one ignored and the
edit never reached data/wikis/. The page then stayed one revision behind,
and the next local edit was reported as a conflict.
"""

import pytest

from ..config import SyncConfig
from ..service import WikiSyncService

TITLE = "Heuristic:Repro Page"  # RecentChanges reports titles with spaces


class FakeWiki:
    """Minimal stand-in for MWClient that keeps one page in memory."""

    def __init__(self):
        self.revid = 0
        self.content = None

    def edit(self, title, text, summary="", bot=False):
        self.revid += 1
        self.content = text
        return {"edit": {"result": "Success", "newrevid": self.revid}}

    def human_edit(self, text):
        """Simulate someone saving the page in the wiki UI."""
        self.revid += 1
        self.content = text
        return self.revid

    def _api(self, params=None, data=None, post=False):
        if self.content is None:
            return {"query": {"pages": {"-1": {}}}}
        revision = {
            "revid": self.revid,
            "slots": {"main": {"*": self.content}},
        }
        return {"query": {"pages": {"1": {"revisions": [revision]}}}}


@pytest.fixture
def service(tmp_path):
    wiki_dir = tmp_path / "wikis"
    (wiki_dir / "heuristics").mkdir(parents=True)
    config = SyncConfig(
        wiki_dir=wiki_dir,
        sync_state=tmp_path / "sync.json",
        refresh_delay=3600,
    )
    svc = WikiSyncService(config)
    svc.engine._mw = FakeWiki()
    yield svc
    if svc._refresh_timer is not None:
        svc._refresh_timer.cancel()


def push_local_file(service, text):
    """Write a local page and run the file watcher's callback for it."""
    path = service.config.wiki_dir / "heuristics" / "Repro_Page.md"
    path.write_text(text, encoding="utf-8")
    service._on_local_change(path)
    return path


def test_wiki_edit_right_after_local_push_reaches_file(service):
    path = push_local_file(service, "== Overview ==\nWritten on disk.\n")
    wiki = service.engine._mw
    edited = wiki.content.replace("Written on disk.", "Edited in the wiki.")

    # The poll sees the push and the human edit together and passes on
    # only the newest revision.
    revid = wiki.human_edit(edited)
    service._on_wiki_change(TITLE, revid, edited)

    assert path.read_text(encoding="utf-8") == edited
    file_state = service.state.get_file_state("heuristics/Repro_Page.md")
    assert file_state.wiki_revid == revid


def test_echo_of_own_push_does_not_rewrite_file(service):
    original = "== Overview ==\nWritten on disk.\n"
    path = push_local_file(service, original)
    wiki = service.engine._mw

    # The poll sees only the revision the push itself created.
    service._on_wiki_change(TITLE, wiki.revid, wiki.content)

    assert path.read_text(encoding="utf-8") == original
