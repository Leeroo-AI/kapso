"""Polling watch mode picks up edits that inotify never reports.

On macOS, Docker Desktop does not pass inotify events from the host into
the container, so files edited on the Mac never reached the wiki. The
watcher can now poll the directory instead (WATCH_MODE=polling).
"""

import threading
from pathlib import Path

import pytest
from watchdog.observers.polling import PollingObserver

from ..config import SyncConfig
from ..file_watcher import FileWatcher

WAIT_SECONDS = 5  # polling scans once a second, then debounces


@pytest.fixture
def page(tmp_path):
    (tmp_path / "heuristics").mkdir()
    path = tmp_path / "heuristics" / "Some_Page.md"
    path.write_text("== Overview ==\nOriginal.\n")
    return path


def start_polling_watcher(wiki_dir):
    """Start a polling watcher that records change and delete callbacks."""
    changed, deleted = [], []
    seen = threading.Event()

    def on_change(path):
        changed.append(Path(path))
        seen.set()

    def on_delete(path):
        deleted.append(Path(path))
        seen.set()

    watcher = FileWatcher(
        wiki_dir,
        debounce_ms=100,
        on_change=on_change,
        on_delete=on_delete,
        polling=True,
    )
    watcher.start()
    return watcher, changed, deleted, seen


def test_watch_mode_defaults_to_events():
    assert SyncConfig().watch_mode == "events"


def test_polling_watcher_uses_polling_observer(page):
    watcher, *_ = start_polling_watcher(page.parents[1])
    try:
        assert isinstance(watcher._observer, PollingObserver)
    finally:
        watcher.stop()


def test_polling_watcher_reports_in_place_edit(page):
    watcher, changed, deleted, seen = start_polling_watcher(page.parents[1])
    try:
        page.write_text("== Overview ==\nEdited in place.\n")
        assert seen.wait(WAIT_SECONDS)
    finally:
        watcher.stop()

    assert changed == [page]
    assert deleted == []


def test_editor_save_is_a_change_not_a_delete(page):
    # Editors save by writing a temp file and renaming it over the page.
    # Polling sees that as delete + create; it must not delete the page,
    # since DELETE_MODE=sync would delete it from the wiki too.
    watcher, changed, deleted, seen = start_polling_watcher(page.parents[1])
    try:
        temp = page.with_name(".Some_Page.md.swp")
        temp.write_text("== Overview ==\nSaved by an editor.\n")
        temp.replace(page)
        assert seen.wait(WAIT_SECONDS)
    finally:
        watcher.stop()

    assert changed == [page]
    assert deleted == []
