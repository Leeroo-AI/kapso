from __future__ import annotations

from dataclasses import replace

import pytest

from kapso.cross_run.expert.replay_execution import (
    ExpertSourceReplayExecutionError,
    ExpertSourceReplayExecutionProviderKey,
    ExpertSourceReplayExecutionProviderRegistry,
    ResolvedExpertSourceReplayExecutionCase,
    expert_source_replay_execution_provider_key,
)
from test_expert_source_replay_request import _prepared, _request_fixture


class _Provider:
    def __init__(self, dispatch_key):
        self.dispatch_key = dispatch_key

    def execute_leg(self, _invocation):
        raise RuntimeError("execution was not expected")


class _CountingProvider:
    def __init__(self, dispatch_key):
        self._dispatch_key = dispatch_key
        self.dispatch_key_reads = 0

    @property
    def dispatch_key(self):
        self.dispatch_key_reads += 1
        return self._dispatch_key

    def execute_leg(self, _invocation):
        raise RuntimeError("execution was not expected")


@pytest.fixture(scope="module")
def prepared_replay_request(tmp_path_factory):
    return _prepared(
        _request_fixture(tmp_path_factory.mktemp("expert-replay-execution"))
    )


def test_provider_key_comes_from_the_exact_compute_and_adapter_authorities(
    prepared_replay_request,
):
    replay_case = prepared_replay_request.cases[0]
    compute = replay_case.request_case.compute_binding
    manifest = replay_case.task_adapter.manifest

    assert expert_source_replay_execution_provider_key(replay_case) == (
        ExpertSourceReplayExecutionProviderKey(
            paired_execution_protocol_version=(
                compute.paired_execution_protocol_version
            ),
            execution_provider_id=compute.execution_provider_id,
            execution_provider_version=compute.execution_provider_version,
            sandbox_policy_version=compute.sandbox_policy_version,
            task_adapter_runtime_protocol_version=(
                manifest.runtime.runtime_protocol_version
            ),
            task_evaluator_protocol_version=(manifest.task_evaluator.protocol_version),
        )
    )


def test_registry_resolves_the_complete_prepared_request_in_case_order(
    prepared_replay_request,
):
    dispatch_key = expert_source_replay_execution_provider_key(
        prepared_replay_request.cases[0]
    )
    provider = _Provider(dispatch_key)

    resolved = ExpertSourceReplayExecutionProviderRegistry((provider,)).resolve_all(
        prepared_replay_request
    )

    assert tuple(item.materialized_case for item in resolved) == (
        prepared_replay_request.cases
    )
    assert all(not hasattr(item, "provider") for item in resolved)
    assert tuple(item.dispatch_key for item in resolved) == (dispatch_key,)


def test_provider_identity_is_read_once_at_each_explicit_fence(
    prepared_replay_request,
):
    dispatch_key = expert_source_replay_execution_provider_key(
        prepared_replay_request.cases[0]
    )
    provider = _CountingProvider(dispatch_key)
    registry = ExpertSourceReplayExecutionProviderRegistry((provider,))

    assert provider.dispatch_key_reads == 1
    resolved = registry.resolve_all(prepared_replay_request)[0]
    assert provider.dispatch_key_reads == 2
    resolved.require_current_provider_identity()
    assert provider.dispatch_key_reads == 3


def test_mixed_key_aggregate_fails_before_any_provider_revalidation(
    prepared_replay_request,
):
    replay_case = prepared_replay_request.cases[0]
    exact_key = expert_source_replay_execution_provider_key(replay_case)
    missing_key = replace(
        exact_key,
        task_adapter_runtime_protocol_version=(
            f"{exact_key.task_adapter_runtime_protocol_version}.unsupported"
        ),
    )
    provider = _CountingProvider(exact_key)
    registry = ExpertSourceReplayExecutionProviderRegistry((provider,))
    provider.dispatch_key_reads = 0

    with pytest.raises(ExpertSourceReplayExecutionError, match="unsupported"):
        registry._resolve_cases(
            (replay_case, replay_case),
            (exact_key, missing_key),
            prepared_replay_request.candidate,
            prepared_replay_request.parent,
        )

    assert provider.dispatch_key_reads == 0


@pytest.mark.parametrize(
    "field_name",
    (
        "paired_execution_protocol_version",
        "execution_provider_id",
        "execution_provider_version",
        "sandbox_policy_version",
        "task_adapter_runtime_protocol_version",
        "task_evaluator_protocol_version",
    ),
)
def test_registry_has_no_partial_version_or_provider_fallback(
    prepared_replay_request,
    field_name,
):
    exact_key = expert_source_replay_execution_provider_key(
        prepared_replay_request.cases[0]
    )
    changed_key = replace(
        exact_key,
        **{field_name: f"{getattr(exact_key, field_name)}.other"},
    )
    registry = ExpertSourceReplayExecutionProviderRegistry((_Provider(changed_key),))

    with pytest.raises(ExpertSourceReplayExecutionError, match="unsupported"):
        registry.resolve_all(prepared_replay_request)


def test_registry_rejects_empty_duplicate_and_untyped_provider_sets(
    prepared_replay_request,
):
    dispatch_key = expert_source_replay_execution_provider_key(
        prepared_replay_request.cases[0]
    )
    provider = _Provider(dispatch_key)

    with pytest.raises(ExpertSourceReplayExecutionError, match="non-empty"):
        ExpertSourceReplayExecutionProviderRegistry(())
    with pytest.raises(ExpertSourceReplayExecutionError, match="tuple"):
        ExpertSourceReplayExecutionProviderRegistry([provider])
    with pytest.raises(ExpertSourceReplayExecutionError, match="duplicated"):
        ExpertSourceReplayExecutionProviderRegistry((provider, provider))
    with pytest.raises(ExpertSourceReplayExecutionError, match="typed"):
        ExpertSourceReplayExecutionProviderRegistry((_Provider("not-a-key"),))


def test_distinct_full_keys_can_use_the_same_provider_implementation(
    prepared_replay_request,
):
    exact_key = expert_source_replay_execution_provider_key(
        prepared_replay_request.cases[0]
    )
    other_key = replace(
        exact_key,
        task_evaluator_protocol_version="kapso.task_evaluator.v2",
    )
    exact_provider = _Provider(exact_key)
    registry = ExpertSourceReplayExecutionProviderRegistry(
        (exact_provider, _Provider(other_key))
    )

    resolved = registry.resolve_all(prepared_replay_request)

    exact_provider.dispatch_key = replace(
        exact_key,
        execution_provider_version=f"{exact_key.execution_provider_version}.changed",
    )
    with pytest.raises(ExpertSourceReplayExecutionError, match="changed"):
        resolved[0].require_current_provider_identity()


def test_resolved_provider_identity_is_fenced_again_before_execution(
    prepared_replay_request,
):
    replay_case = prepared_replay_request.cases[0]
    exact_key = expert_source_replay_execution_provider_key(replay_case)
    provider = _Provider(exact_key)
    resolved = ExpertSourceReplayExecutionProviderRegistry((provider,)).resolve_all(
        prepared_replay_request
    )[0]
    provider.dispatch_key = replace(
        exact_key,
        execution_provider_version=f"{exact_key.execution_provider_version}.changed",
    )

    with pytest.raises(ExpertSourceReplayExecutionError, match="changed"):
        resolved.require_current_provider_identity()
    with pytest.raises(ExpertSourceReplayExecutionError, match="changed"):
        ResolvedExpertSourceReplayExecutionCase(
            materialized_case=replay_case,
            dispatch_key=exact_key,
            candidate=prepared_replay_request.candidate,
            parent=prepared_replay_request.parent,
            provider=provider,
        )


def test_resolved_provider_requires_exact_typed_expert_sources(
    prepared_replay_request,
):
    replay_case = prepared_replay_request.cases[0]
    exact_key = expert_source_replay_execution_provider_key(replay_case)
    provider = _Provider(exact_key)

    with pytest.raises(ExpertSourceReplayExecutionError, match="expert sources"):
        ResolvedExpertSourceReplayExecutionCase(
            materialized_case=replay_case,
            dispatch_key=exact_key,
            candidate=prepared_replay_request.parent,
            parent=prepared_replay_request.parent,
            provider=provider,
        )


def test_registry_requires_a_typed_prepared_request(prepared_replay_request):
    key = expert_source_replay_execution_provider_key(prepared_replay_request.cases[0])
    registry = ExpertSourceReplayExecutionProviderRegistry((_Provider(key),))

    with pytest.raises(ExpertSourceReplayExecutionError, match="prepared"):
        registry.resolve_all(prepared_replay_request.request)


@pytest.mark.parametrize(
    "invalid_value",
    ("", "contains whitespace", "?unsupported"),
)
def test_provider_key_rejects_malformed_versions(
    prepared_replay_request,
    invalid_value,
):
    exact_key = expert_source_replay_execution_provider_key(
        prepared_replay_request.cases[0]
    )

    with pytest.raises(ValueError):
        replace(
            exact_key,
            paired_execution_protocol_version=invalid_value,
        )
