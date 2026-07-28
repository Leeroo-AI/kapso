"""Deterministic RepoMemory rebuild after BootstrapPin handoff."""

from __future__ import annotations

from kapso.cross_run.canonical import parse_json_bytes
from kapso.cross_run.launch.repository_memory import build_repository_memory
from test_launch_resolver import resolver_case
from test_run_state_publisher import publisher_case


def test_repository_memory_is_deterministic_and_non_mutating(
    publisher_case,
    resolver_case,
):
    active = publisher_case["active"]
    settings = resolver_case["resolver"].settings
    before = active.bootstrap_pin.installation_receipt.workspace_baseline_commit_sha

    first = build_repository_memory(
        active_workspace=active,
        settings=settings,
    )
    second = build_repository_memory(
        active_workspace=active,
        settings=settings,
    )

    assert first == second
    memory = parse_json_bytes(first.payload)
    assert memory["schema_version"] == "kapso.repository_memory.v1"
    assert memory["source_commit_sha"] == before
    assert memory["files"]
    assert tuple(section["section_id"] for section in memory["table_of_contents"]) == (
        "repository.entrypoints",
        "repository.tests",
        "repository.documentation",
        "repository.top_level",
    )
    assert active.bootstrap_pin.installation_receipt.workspace_baseline_commit_sha == (
        before
    )
