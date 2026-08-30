"""Git-native bank sharing: the bank home's `origin` remote IS the share
target. No remote in config — a bank carries its own, exactly like any
git repo, and `kapso bank connect <url>` is how one gets attached.

All subprocess calls run with GIT_TERMINAL_PROMPT=0 so a missing
credential fails loudly in seconds instead of hanging on a prompt."""

import os
import re
import subprocess
from pathlib import Path
from typing import Optional

GITHUB_SLUG_PATTERN = re.compile(r"^[\w.-]+/[\w.-]+$")


def _git_env() -> dict:
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def bank_origin(home: Path) -> Optional[str]:
    """The bank's `origin` URL, or None when no remote is attached (the
    documented local-only state)."""
    result = subprocess.run(
        ["git", "--git-dir", str(home), "remote", "get-url", "origin"],
        capture_output=True, text=True, env=_git_env(),
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def verify_bank_remote(home: Path) -> str:
    """Preflight the bank's origin: reachable and authenticated, in
    seconds — so a learn() run can never spend hours and then die on the
    final push. Returns the verified URL; raises on any failure."""
    url = bank_origin(home)
    if url is None:
        raise ValueError(
            f"bank {home} has no origin remote — run "
            "`kapso bank connect <url>` first"
        )
    result = subprocess.run(
        ["git", "--git-dir", str(home), "ls-remote", "origin"],
        capture_output=True, text=True, env=_git_env(), timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"bank remote {url} is not reachable with your git "
            f"credentials:\n{result.stderr.strip()}\n"
            "Fix access (ssh key / gh auth / credential helper), or "
            "detach with: git --git-dir "
            f"{home} remote remove origin"
        )
    return url


def connect_bank(home: Path, url: str) -> None:
    """Attach `url` as the bank's origin and push the current bank there.
    Validates reachability before touching anything; the push is plain
    (no force) — a remote with unrelated history is rejected loudly."""
    subcommand = "set-url" if bank_origin(home) else "add"
    subprocess.run(
        ["git", "--git-dir", str(home), "remote", subcommand, "origin", url],
        check=True, env=_git_env(),
    )
    verify_bank_remote(home)
    subprocess.run(
        ["git", "--git-dir", str(home), "push", "origin", "main", "--tags"],
        check=True, env=_git_env(),
    )


def create_bank_repo(slug: str, *, private: bool = True) -> str:
    """Create a GitHub repo for the bank via the `gh` CLI (the caller
    must have checked `gh` exists) and return its clone URL."""
    if not GITHUB_SLUG_PATTERN.match(slug):
        raise ValueError(
            f"expected an org/name slug (e.g. acme/kapso-bank), got {slug!r}"
        )
    visibility = "--private" if private else "--public"
    subprocess.run(
        ["gh", "repo", "create", slug, visibility],
        check=True, env=_git_env(),
    )
    return f"https://github.com/{slug}.git"
