"""Shared pytest wiring.

Live suites — tests that spawn real coding-agent sessions and spend real
subscription quota — are skipped unless --run-live is passed, so a plain
`pytest tests/...` stays safe and free for contributors (onboarding E2E
finding #4: an unmarked live suite hung a gate run for 10 minutes while
silently burning quota)."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-live", action="store_true", default=False,
        help="run live suites that spawn real claude/codex sessions",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: spawns real coding-agent sessions (skipped without --run-live)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-live"):
        return
    skip_live = pytest.mark.skip(reason="live suite — pass --run-live to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
