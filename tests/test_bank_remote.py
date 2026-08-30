# bank_remote — git-native share-remote contracts (onboarding E2E
# finding #1: the remote lives IN the bank as `origin`, never in config;
# Rule 9: each test names its regression). All remotes here are local
# bare repos — real git, no network, no credentials.

import subprocess
from pathlib import Path

import pytest

from kapso.learning.bank_remote import (
    bank_origin,
    connect_bank,
    create_bank_repo,
    verify_bank_remote,
)
from kapso.learning.update_frame import init_bank


def make_home(tmp_path) -> Path:
    home = tmp_path / "bank.git"
    init_bank(str(home))
    return home


def make_share(tmp_path, name="share.git") -> Path:
    share = tmp_path / name
    subprocess.run(["git", "init", "--bare", "-q", str(share)], check=True)
    return share


def git_head(git_dir: Path) -> str:
    return subprocess.run(
        ["git", "--git-dir", str(git_dir), "rev-parse", "main"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def test_origin_absent_then_connect_roundtrip(tmp_path):
    # Regression: a fresh bank is local-only (origin None — the documented
    # default), and connect attaches origin AND pushes, so the share
    # remote immediately holds the bank another machine could clone.
    home = make_home(tmp_path)
    assert bank_origin(home) is None
    share = make_share(tmp_path)
    connect_bank(home, str(share))
    assert bank_origin(home) == str(share)
    assert git_head(share) == git_head(home)


def test_connect_twice_moves_origin(tmp_path):
    # Regression: re-connect must set-url, not add (git errors on a
    # second `remote add origin`).
    home = make_home(tmp_path)
    first = make_share(tmp_path, "first.git")
    second = make_share(tmp_path, "second.git")
    connect_bank(home, str(first))
    connect_bank(home, str(second))
    assert bank_origin(home) == str(second)
    assert git_head(second) == git_head(home)


def test_verify_unreachable_raises_with_fix_guidance(tmp_path):
    # Regression: the preflight fails loud with the git error AND the
    # detach escape hatch — never a silent skip.
    home = make_home(tmp_path)
    subprocess.run(
        ["git", "--git-dir", str(home), "remote", "add", "origin",
         str(tmp_path / "missing.git")],
        check=True,
    )
    with pytest.raises(RuntimeError, match="not reachable"):
        verify_bank_remote(home)


def test_verify_without_origin_names_the_connect_command(tmp_path):
    home = make_home(tmp_path)
    with pytest.raises(ValueError, match="kapso bank connect"):
        verify_bank_remote(home)


def test_create_rejects_malformed_slug():
    # Regression: a bare name (no org/) must fail before any gh call.
    with pytest.raises(ValueError, match="org/name"):
        create_bank_repo("not-a-slug")
