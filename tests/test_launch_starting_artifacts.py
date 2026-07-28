from __future__ import annotations

import pytest

from kapso.cross_run.canonical import content_id
from kapso.cross_run.launch.starting_artifacts import (
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
