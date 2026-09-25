"""A failed MediaWiki login must not leave a half-built client behind.

The poller and the engine create their client lazily. Storing it before
login() succeeded meant one unreachable wiki (a container restart on
2026-03-19) left a client with no API URL, and every poll for six months
failed on "API URL not resolved yet" instead of logging in again.
"""

import pytest

from .. import sync_engine, wiki_poller
from ..config import SyncConfig
from ..state_manager import StateManager


class FlakyClient:
    """Fails its first login, succeeds afterwards."""

    attempts = 0

    def __init__(self, base, user, password):
        self.api_url = f"{base}/api.php"

    def login(self):
        type(self).attempts += 1
        if type(self).attempts == 1:
            raise RuntimeError("Could not obtain login token")


@pytest.fixture
def state(tmp_path):
    return StateManager(tmp_path / "sync.json")


@pytest.mark.parametrize(
    "module, build",
    [
        (wiki_poller, lambda cfg, st: wiki_poller.WikiPoller(cfg, st, lambda *a: None, lambda *a: None)),
        (sync_engine, lambda cfg, st: sync_engine.SyncEngine(cfg, st)),
    ],
    ids=["poller", "engine"],
)
def test_failed_login_is_retried_on_next_access(monkeypatch, state, module, build):
    FlakyClient.attempts = 0
    monkeypatch.setattr(module, "MWClient", FlakyClient)
    owner = build(SyncConfig(wiki_url="http://wiki:80", mw_pass="x"), state)

    with pytest.raises(RuntimeError, match="login token"):
        owner.mw
    assert owner._mw is None  # nothing stored: the next access logs in again

    assert owner.mw.api_url == "http://wiki:80/api.php"
    assert FlakyClient.attempts == 2
    assert owner.mw is owner._mw  # and the working client is kept
