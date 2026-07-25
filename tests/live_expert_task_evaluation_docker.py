"""Explicit real-Docker task-evaluation production check.

Run directly; the filename intentionally stays outside normal pytest discovery:

    pytest -q tests/live_expert_task_evaluation_docker.py -s

The check pulls one deterministic digest-pinned image from a loopback OCI
registry, then executes parent-comparison and bootstrap matrices through the
production request-bound registry, fresh-authority coordinator, and durable
task-evaluation journal. Adapter, CURRENT, and denylist transports remain at
the synthetic fixture boundary.
"""

from __future__ import annotations

import os
from contextlib import ExitStack
from pathlib import Path

import pytest

import test_expert_release_matrix_reservation as release_matrix_fixture_module
from expert_live_docker_support import (
    assert_no_daemon_resources,
    cleanup_daemon_resources,
    remove_exact_image,
    require_setup_docker_success,
    run_setup_docker,
)
from kapso.cross_run.canonical import source_tree_digest, tree_or_blob_digest
from kapso.cross_run.contracts import (
    SourceFileDescriptor,
    TaskAdapterManifest,
    TaskAdapterReleaseMatrixStartingArtifact,
    TaskAdapterRuntimeContract,
)
from kapso.cross_run.docker.runtime import read_verified_root_executable
from kapso.cross_run.expert.task_evaluation_authority import (
    TaskEvaluationFreshAuthorityCoordinator,
)
from kapso.cross_run.expert.task_evaluation_contracts import TaskEvaluationLegKind
from kapso.cross_run.expert.task_evaluation_docker_bootstrap import (
    build_task_evaluation_docker_provider_registry,
)
from kapso.cross_run.expert.task_evaluation_execution_journal import (
    TaskEvaluationExecutionJournalEventKind,
    task_evaluation_execution_schedule,
)
from kapso.cross_run.expert.task_evaluation_execution_store import (
    CompletedTaskEvaluationExecution,
    ExpertTaskEvaluationExecutionStore,
)
from test_cross_run_contracts import (
    TASK_ADAPTER_RUNTIME_LOCK,
    build_records,
    verified_test_task_adapter,
)
from test_expert_source_replay import _AdapterProvider
from test_expert_task_evaluation_authority_runtime import _bootstrap_prepared
from test_expert_task_evaluation_execution_store import (
    _AdapterAuthority,
    _CurrentAuthority,
    _DenylistAuthority,
)
from test_expert_task_evaluation_reservation import _parent_prepared
from live_expert_replay_docker import _start_local_oci_registry

_MATRIX_ARTIFACT_PAYLOAD = b"task evaluation matrix fixture\n"
_ADAPTER_SOURCE = rb"""#!/bin/busybox sh
set -eu

[ "$(pwd)" = "/kapso/input/adapter" ] || exit 11
[ "$(/bin/busybox hostname)" = "kapso-task-evaluation" ] || exit 12
[ "$(/bin/busybox cat requirements.lock)" = "python==3.11.9" ] || exit 13
[ "$(/bin/busybox cat /kapso/input/task/matrix/live/fixture.bin)" = "task evaluation matrix fixture" ] || exit 14
[ "$(/bin/busybox ls /sys/class/net)" = "lo" ] || exit 15

actual_environment="$(/bin/busybox env | /bin/busybox sort)"
expected_environment='HOME=/kapso/home
HOSTNAME=kapso-task-evaluation
LANG=C
PATH=/bin
PWD=/kapso/input/adapter
SHLVL=1'
if [ "$actual_environment" != "$expected_environment" ]; then
    printf '%s\n' "$actual_environment"
    exit 16
fi

if /bin/busybox sh -c 'printf changed > /kapso/input/expert/EXPERT_REPO.md' >/dev/null 2>&1; then
    exit 21
fi
if /bin/busybox sh -c 'printf changed > /kapso/input/adapter/adapter.py' >/dev/null 2>&1; then
    exit 22
fi
if /bin/busybox sh -c 'printf changed > /kapso/input/task/matrix/live/fixture.bin' >/dev/null 2>&1; then
    exit 23
fi
if /bin/busybox sh -c 'printf changed > /kapso-unexpected-write' >/dev/null 2>&1; then
    exit 24
fi

if [ -f /kapso/input/expert/src/reproducible_execution/__init__.py ]; then
    expert_source="$(/bin/busybox cat /kapso/input/expert/src/reproducible_execution/__init__.py)"
elif [ -f /kapso/input/expert/src/execution.py ]; then
    expert_source="$(/bin/busybox cat /kapso/input/expert/src/execution.py)"
else
    exit 25
fi
case "$expert_source" in
    *run_with_provenance*) score='0.8' ;;
    *'return task.run()'*) score='0.7' ;;
    *) exit 26 ;;
esac

request="$(/bin/busybox cat /kapso/input/request.json)"
opaque_invocation_id="$(printf '%s' "$request" | /bin/busybox sed 's/.*"opaque_invocation_id":"\([^"]*\)".*/\1/')"
evaluation_fingerprint_id="$(printf '%s' "$request" | /bin/busybox sed 's/.*"evaluation_fingerprint_id":"\([^"]*\)".*/\1/')"
replicate_id="$(printf '%s' "$request" | /bin/busybox sed 's/.*"seed_or_replicate_ids":\["\([^"]*\)"\].*/\1/')"

case "$opaque_invocation_id" in task_evaluation_invocation_*) ;; *) exit 27 ;; esac
case "$evaluation_fingerprint_id" in evaluation-fingerprint:sha256:*) ;; *) exit 28 ;; esac
[ -n "$replicate_id" ] || exit 29

printf '{"fingerprint_results":[{"aggregate_value":%s,"evaluation_fingerprint_id":"%s","replicate_values":{"%s":%s}}],"opaque_invocation_id":"%s","protocol_version":"kapso.task_evaluator.v1"}' \
    "$score" "$evaluation_fingerprint_id" "$replicate_id" "$score" "$opaque_invocation_id" \
    > /kapso/writable/result.json
"""


class _RecordingReservationAuthority:
    def __init__(self, validation_store) -> None:
        self.validation_store = validation_store
        self.calls: list[str] = []

    def reopen_task_evaluation_reservation(self, **request):
        self.calls.append(request["reservation_id"])
        return self.validation_store.reopen_task_evaluation_reservation(**request)


def _matrix_starting_artifact() -> TaskAdapterReleaseMatrixStartingArtifact:
    descriptor = SourceFileDescriptor(
        relative_path="fixture.bin",
        digest=tree_or_blob_digest(_MATRIX_ARTIFACT_PAYLOAD),
        mode="100644",
        size=len(_MATRIX_ARTIFACT_PAYLOAD),
    )
    return TaskAdapterReleaseMatrixStartingArtifact.mint(
        starting_artifact_ref="artifact/task-evaluation-live",
        mount_path="matrix/live",
        package_source_root="release_matrix_assets/live",
        materialized_tree_hash=source_tree_digest(
            {
                descriptor.relative_path: (
                    descriptor.digest,
                    descriptor.mode,
                    descriptor.size,
                )
            }
        ),
        source_files=(descriptor,),
    )


def _live_adapter(local_registry):
    runtime_contract = TaskAdapterRuntimeContract(
        runtime_protocol_version="kapso.task_adapter_runtime.v1",
        image_repository=local_registry.repository,
        image_manifest_digest=local_registry.manifest_digest,
        image_config_digest=local_registry.config_digest,
        dependency_lock_path="requirements.lock",
        dependency_lock_digest=tree_or_blob_digest(TASK_ADAPTER_RUNTIME_LOCK),
        operating_system="linux",
        architecture="amd64",
        architecture_variant=None,
        environment={"LANG": "C", "PATH": "/bin"},
    )
    starting_artifact = _matrix_starting_artifact()
    source_contents = {
        "adapter.py": _ADAPTER_SOURCE,
        "release_matrix_assets/live/fixture.bin": _MATRIX_ARTIFACT_PAYLOAD,
        "requirements.lock": TASK_ADAPTER_RUNTIME_LOCK,
    }
    records = build_records(
        task_adapter_runtime=runtime_contract,
        task_adapter_source_contents=source_contents,
        task_adapter_release_matrix_starting_artifacts=(starting_artifact,),
    )
    manifest = next(
        record for record in records if isinstance(record, TaskAdapterManifest)
    )
    return records, verified_test_task_adapter(
        manifest,
        source_contents=source_contents,
    )


def _prepare_parent_and_bootstrap_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    records,
    live_adapter,
):
    original_request_fixture = release_matrix_fixture_module._request_fixture

    def live_request_fixture(fixture_root, *, validation_settings):
        return original_request_fixture(
            fixture_root,
            validation_settings=validation_settings,
            contract_records=records,
            source_adapter=live_adapter,
        )

    def live_adapter_provider(packet, *, rotate_active=False):
        return _AdapterProvider(
            packet,
            source_adapter=live_adapter,
            rotate_active=rotate_active,
        )

    monkeypatch.setattr(
        release_matrix_fixture_module,
        "_request_fixture",
        live_request_fixture,
    )
    monkeypatch.setattr(
        release_matrix_fixture_module,
        "_AdapterProvider",
        live_adapter_provider,
    )
    parent_root = tmp_path / "parent"
    parent_root.mkdir(mode=0o700)
    parent_authority = _parent_prepared(parent_root, monkeypatch)
    bootstrap_root = tmp_path / "bootstrap"
    bootstrap_root.mkdir(mode=0o700)
    bootstrap_authority = _bootstrap_prepared(bootstrap_root, monkeypatch)
    monkeypatch.undo()
    return parent_authority, bootstrap_authority


def _execute_prepared_request(
    *,
    validation_store,
    validation_snapshot,
    prepared_request,
    workspace_root: Path,
    cleanup_handle_ids: list[str],
):
    reservation_snapshot = validation_store.reserve_task_evaluation(
        expected_transition_id=validation_snapshot.transition.transition_id,
        prepared_request=prepared_request,
    ).reservation
    execution_store = ExpertTaskEvaluationExecutionStore(
        ExpertTaskEvaluationExecutionStore.canonical_root(
            validation_store.root
        ).resolve(),
        validation_store.root,
        prepared_request.plan_join.settings.policy,
    )
    provider_registry = build_task_evaluation_docker_provider_registry(
        prepared_request=prepared_request,
        workspace_root=workspace_root,
    )
    reservation_authority = _RecordingReservationAuthority(validation_store)
    current_authority = _CurrentAuthority(prepared_request.current_release_observation)
    adapter_authority = _AdapterAuthority(prepared_request)
    denylist_authority = _DenylistAuthority(prepared_request)
    coordinator = TaskEvaluationFreshAuthorityCoordinator(
        reservation_authority=reservation_authority,
        execution_store=execution_store,
        current_release_authority=current_authority,
        task_adapter_authority=adapter_authority,
        security_denylist_authority=denylist_authority,
    )
    schedule = task_evaluation_execution_schedule(
        reservation_snapshot,
        prepared_request,
    )
    with execution_store.reservation_session(
        reservation_snapshot=reservation_snapshot,
        prepared_request=prepared_request,
    ) as session:
        while len(session.events) < 4 * len(schedule):
            allocation_permit = session.allocate_expected_leg()
            spawn_permit = coordinator.commit_spawn(
                prepared_request=prepared_request,
                reservation_id=reservation_snapshot.reservation.reservation_id,
                invocation_permit=allocation_permit,
                provider_registry=provider_registry,
            )
            cleanup_handle_ids.append(
                session.events[-1].provider_execution_handle.provider_handle_id
            )
            completion = spawn_permit.execute()
            session.record_result_received(completion)
            session.accept_received_result()
        completed = session.completed_execution()

    assert type(completed) is CompletedTaskEvaluationExecution
    assert (
        completed.require_exact(
            execution_store,
            reservation_snapshot,
            prepared_request,
        )
        == completed.events
    )
    assert len(reservation_authority.calls) == 2 * len(schedule)
    assert len(current_authority.calls) == 2 * len(schedule)
    assert len(adapter_authority.calls) == len(prepared_request.adapters) * len(
        schedule
    )
    assert len(denylist_authority.calls) == len(schedule)

    reopened_store = ExpertTaskEvaluationExecutionStore(
        execution_store.root,
        execution_store.trusted_root,
        execution_store.policy_settings,
    )
    with reopened_store.reservation_session(
        reservation_snapshot=reservation_snapshot,
        prepared_request=prepared_request,
    ) as reopened_session:
        reopened_completed = reopened_session.completed_execution()
    assert type(reopened_completed) is CompletedTaskEvaluationExecution
    assert (
        reopened_completed.require_exact(
            reopened_store,
            reservation_snapshot,
            prepared_request,
        )
        == completed.events
    )
    return completed


def _accepted_values_by_leg_kind(
    completed: CompletedTaskEvaluationExecution,
    prepared_request,
) -> dict[TaskEvaluationLegKind, float]:
    leg_kinds = {
        leg.leg_id: leg.kind
        for materialized_case in prepared_request.cases
        for leg in materialized_case.request_case.legs
    }
    values_by_leg_kind = {
        leg_kinds[event.invocation_allocation.evaluation_leg_id]: result.aggregate_value
        for event in completed.events
        if event.event_kind is TaskEvaluationExecutionJournalEventKind.RESULT_ACCEPTED
        for result in event.task_evaluator_result.fingerprint_results
    }
    accepted_event_count = sum(
        event.event_kind is TaskEvaluationExecutionJournalEventKind.RESULT_ACCEPTED
        for event in completed.events
    )
    assert len(values_by_leg_kind) == accepted_event_count
    return values_by_leg_kind


def _provider_handle_ids(
    executions: tuple[CompletedTaskEvaluationExecution, ...],
) -> tuple[str, ...]:
    return tuple(
        event.provider_execution_handle.provider_handle_id
        for completed in executions
        for event in completed.events
        if event.event_kind is TaskEvaluationExecutionJournalEventKind.SPAWN_COMMITTED
    )


def test_real_docker_executes_parent_and_bootstrap_task_evaluations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    provider_settings = (
        release_matrix_fixture_module._quality_only_validation_settings().task_evaluation_provider
    )
    busybox_bytes = read_verified_root_executable(
        Path(provider_settings.runtime.helper_executable_path),
        provider_settings.runtime.helper_executable_digest,
    )

    with ExitStack() as cleanup:
        local_registry = _start_local_oci_registry(cleanup, busybox_bytes)
        records, live_adapter = _live_adapter(local_registry)
        parent_authority, bootstrap_authority = _prepare_parent_and_bootstrap_requests(
            tmp_path,
            monkeypatch,
            records,
            live_adapter,
        )
        docker_config_root = tmp_path / "setup-docker-config"
        docker_config_root.mkdir(mode=0o700)
        docker_config_path = docker_config_root / "config.json"
        docker_config_path.write_bytes(b'{"auths":{}}\n')
        docker_config_path.chmod(0o400)
        image_reference = live_adapter.manifest.runtime.image_reference
        cleanup.callback(
            remove_exact_image,
            provider_settings.runtime,
            docker_config_root,
            image_reference,
        )
        cleanup_handle_ids: list[str] = []
        cleanup.callback(
            cleanup_daemon_resources,
            provider_settings.runtime,
            docker_config_root,
            cleanup_handle_ids,
        )
        pull_result = run_setup_docker(
            provider_settings.runtime,
            docker_config_root,
            (
                "image",
                "pull",
                "--platform",
                "linux/amd64",
                image_reference,
            ),
        )
        require_setup_docker_success(pull_result, "task-evaluation")
        assert local_registry.server.observed_violations == ()
        registry_requests_after_pull = local_registry.server.request_count

        parent_execution = _execute_prepared_request(
            validation_store=parent_authority[0],
            validation_snapshot=parent_authority[1],
            prepared_request=parent_authority[2],
            workspace_root=tmp_path.resolve(),
            cleanup_handle_ids=cleanup_handle_ids,
        )
        bootstrap_execution = _execute_prepared_request(
            validation_store=bootstrap_authority[0],
            validation_snapshot=bootstrap_authority[1],
            prepared_request=bootstrap_authority[2],
            workspace_root=tmp_path.resolve(),
            cleanup_handle_ids=cleanup_handle_ids,
        )
        completed_executions = (parent_execution, bootstrap_execution)
        provider_handle_ids = _provider_handle_ids(completed_executions)

        assert _accepted_values_by_leg_kind(
            parent_execution,
            parent_authority[2],
        ) == {
            TaskEvaluationLegKind.CANDIDATE: 0.8,
            TaskEvaluationLegKind.SOURCE_BASE_CONTROL: 0.7,
        }
        assert _accepted_values_by_leg_kind(
            bootstrap_execution,
            bootstrap_authority[2],
        ) == {TaskEvaluationLegKind.CANDIDATE: 0.7}
        assert len(provider_handle_ids) == 3
        assert len(set(provider_handle_ids)) == 3
        assert local_registry.server.request_count == registry_requests_after_pull
        assert local_registry.server.observed_violations == ()
        assert_no_daemon_resources(
            provider_settings.runtime,
            docker_config_root,
            provider_handle_ids,
            "task evaluation",
        )
        configured_provider_root = (
            tmp_path / provider_settings.workspace_path
        ).resolve()
        assert tuple(configured_provider_root.glob("execution-*")) == ()
        assert os.geteuid() == provider_settings.container_user_id
        assert os.getegid() == provider_settings.container_group_id
