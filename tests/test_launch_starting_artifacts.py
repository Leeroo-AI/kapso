"""Filesystem sealing for immutable launch starting artifacts."""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from kapso.cross_run.canonical import content_id
from kapso.cross_run.launch.starting_artifacts import (
    build_launch_starting_artifact_provider,
    LaunchStartingArtifactProviderError,
    LaunchStartingArtifactSetProvider,
)
from test_launch_resolver import resolver_case


def test_verified_artifact_set_materializes_the_exact_task_context(resolver_case):
    fixed = resolver_case["starting_artifacts"].verified
    request = resolver_case["request"]
    scope_contract = resolver_case["evidence"].validation_context.scope_contract
    context = request.task_context_request.bind(
        binding=request.binding,
        scope_contract=scope_contract,
    )
    settings = resolver_case["resolver"].settings.launch
    provider = LaunchStartingArtifactSetProvider(
        fixed.starting_artifacts,
        settings,
    )

    materialized = provider.materialize_exact(
        task_context_binding=context,
        expected_artifact_content_ids=request.starting_artifact_content_ids,
        maximum_entries=settings.starting_artifact_entry_limit,
        maximum_bytes=settings.starting_artifact_byte_limit,
    )

    assert materialized == fixed


def test_verified_artifact_set_rejects_another_requested_closure(resolver_case):
    fixed = resolver_case["starting_artifacts"].verified
    request = resolver_case["request"]
    scope_contract = resolver_case["evidence"].validation_context.scope_contract
    context = request.task_context_request.bind(
        binding=request.binding,
        scope_contract=scope_contract,
    )
    settings = resolver_case["resolver"].settings.launch
    provider = LaunchStartingArtifactSetProvider(
        fixed.starting_artifacts,
        settings,
    )

    with pytest.raises(
        LaunchStartingArtifactProviderError,
        match="differs from its verified closure",
    ):
        provider.materialize_exact(
            task_context_binding=context,
            expected_artifact_content_ids={
                "foreign": content_id("launch-starting-artifact", {"foreign": True})
            },
            maximum_entries=settings.starting_artifact_entry_limit,
            maximum_bytes=settings.starting_artifact_byte_limit,
        )


def test_builder_seals_complete_sorted_source_closures(resolver_case, tmp_path):
    evaluation = (tmp_path / "evaluation").absolute()
    evaluation.mkdir()
    (evaluation / "nested").mkdir()
    (evaluation / "nested" / "score.py").write_text(
        "print('score')\n",
        encoding="utf-8",
    )
    executable = evaluation / "run.sh"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    data = (tmp_path / "data").absolute()
    data.mkdir()
    (data / "input.csv").write_text("x,y\n1,2\n", encoding="utf-8")

    provider = build_launch_starting_artifact_provider(
        sources={
            "task-data": (data, "kapso_datasets"),
            "task-evaluation": (evaluation, "kapso_evaluation"),
        },
        settings=resolver_case["resolver"].settings.launch,
    )

    artifacts = provider._artifacts
    assert tuple(
        item.artifact.starting_artifact_content_id for item in artifacts
    ) == tuple(sorted(item.artifact.starting_artifact_content_id for item in artifacts))
    by_ref = {item.artifact.starting_artifact_ref: item for item in artifacts}
    assert by_ref["task-evaluation"].source_contents == {
        "nested/score.py": b"print('score')\n",
        "run.sh": b"#!/bin/sh\nexit 0\n",
    }
    assert tuple(
        descriptor.mode
        for descriptor in by_ref["task-evaluation"].artifact.source_files
    ) == ("100644", "100755")
    assert by_ref["task-data"].source_contents == {"input.csv": b"x,y\n1,2\n"}


def test_builder_rejects_symlinks_and_configured_byte_overflow(
    resolver_case,
    tmp_path,
):
    source = (tmp_path / "source").absolute()
    source.mkdir()
    target = source / "target.txt"
    target.write_text("complete", encoding="utf-8")
    os.symlink(target, source / "link.txt")

    with pytest.raises(LaunchStartingArtifactProviderError, match="symbolic"):
        build_launch_starting_artifact_provider(
            sources={"task": (source, "task")},
            settings=resolver_case["resolver"].settings.launch,
        )

    (source / "link.txt").unlink()
    settings = replace(
        resolver_case["resolver"].settings.launch,
        starting_artifact_byte_limit=1,
    )
    with pytest.raises(LaunchStartingArtifactProviderError, match="byte bound"):
        build_launch_starting_artifact_provider(
            sources={"task": (source, "task")},
            settings=settings,
        )


def test_builder_accepts_an_exact_empty_closure(resolver_case):
    provider = build_launch_starting_artifact_provider(
        sources={},
        settings=resolver_case["resolver"].settings.launch,
    )

    assert dict(provider.content_ids) == {}
