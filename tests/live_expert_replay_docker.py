"""Explicit real-Docker source-replay production check.

Run directly; the filename intentionally stays outside normal pytest discovery:

    pytest -q tests/live_expert_replay_docker.py -s

The check serves a deterministic digest-pinned OCI image from a local read-only
registry, builds the complete replay authority from that runtime, and executes
both journal-owned scientific legs through the concrete Docker provider. Adapter
publisher verification remains at the synthetic test-provider boundary.
"""

from __future__ import annotations

import os
import tarfile
from contextlib import ExitStack
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import urlsplit

from expert_live_docker_support import (
    assert_no_daemon_resources,
    cleanup_daemon_resources,
    remove_exact_image,
    require_setup_docker_success,
    run_setup_docker,
)
from kapso.cross_run.canonical import canonical_json_bytes, tree_or_blob_digest
from kapso.cross_run.contracts import (
    ExpertEvaluatorOutcome,
    ExpertPromotionState,
    ExpertValidationStage,
    TaskAdapterManifest,
    TaskAdapterRuntimeContract,
)
from kapso.cross_run.docker.runtime import read_verified_root_executable
from kapso.cross_run.expert.replay_docker_bootstrap import (
    build_source_replay_docker_provider_registry,
)
from kapso.cross_run.expert.replay_execution_store import (
    ExpertSourceReplayExecutionStore,
    SourceReplayExecutionJournalEventKind,
)
from kapso.cross_run.expert.replay_publication import (
    ExpertSourceReplayDecisionPublicationCoordinator,
)
from kapso.cross_run.expert.replay_stage import (
    ExpertSourceReplayStageOrchestrator,
)
from test_cross_run_contracts import (
    TASK_ADAPTER_RUNTIME_LOCK,
    build_records,
    verified_test_task_adapter,
)
from test_expert_replay_execution_store import _coordinator
from test_expert_source_replay import _validation_policy
from test_expert_source_replay_request import _prepared, _request_fixture

_OCI_MANIFEST_MEDIA_TYPE = "application/vnd.oci.image.manifest.v1+json"
_OCI_CONFIG_MEDIA_TYPE = "application/vnd.oci.image.config.v1+json"
_OCI_LAYER_MEDIA_TYPE = "application/vnd.oci.image.layer.v1.tar"
_REGISTRY_REPOSITORY_PATH = "kapso/source-replay-e2e"
_SENSITIVE_REQUEST_HEADERS = (
    "Authorization",
    "Cookie",
    "Proxy-Authorization",
)
_ADAPTER_SOURCE = rb"""#!/bin/busybox sh
set -eu

[ "$(pwd)" = "/kapso/input/adapter" ] || exit 11
[ "$(/bin/busybox hostname)" = "kapso-task-evaluation" ] || exit 12
[ "$(/bin/busybox cat requirements.lock)" = "python==3.11.9" ] || exit 13
[ "$(/bin/busybox cat /kapso/input/task/inputs/base/artifact.bin)" = "starting artifact:artifact/base" ] || exit 14
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

if /bin/busybox sh -c 'printf changed > /kapso/input/expert/src/expert.py' >/dev/null 2>&1; then
    exit 21
fi
if /bin/busybox sh -c 'printf changed > /kapso/input/adapter/adapter.py' >/dev/null 2>&1; then
    exit 22
fi
if /bin/busybox sh -c 'printf changed > /kapso-unexpected-write' >/dev/null 2>&1; then
    exit 23
fi

expert_source="$(/bin/busybox cat /kapso/input/expert/src/expert.py)"
case "$expert_source" in
    'verified source-base source') score='0.7' ;;
    'verified candidate source') score='0.8' ;;
    *) exit 24 ;;
esac

request="$(/bin/busybox cat /kapso/input/request.json)"
opaque_invocation_id="$(printf '%s' "$request" | /bin/busybox sed 's/.*"opaque_invocation_id":"\([^"]*\)".*/\1/')"
evaluation_fingerprint_id="$(printf '%s' "$request" | /bin/busybox sed 's/.*"evaluation_fingerprint_id":"\([^"]*\)".*/\1/')"
replicate_id="$(printf '%s' "$request" | /bin/busybox sed 's/.*"seed_or_replicate_ids":\["\([^"]*\)"\].*/\1/')"

case "$opaque_invocation_id" in task_evaluation_invocation_*) ;; *) exit 25 ;; esac
case "$evaluation_fingerprint_id" in evaluation-fingerprint:sha256:*) ;; *) exit 26 ;; esac
[ -n "$replicate_id" ] || exit 27

printf '{"fingerprint_results":[{"aggregate_value":%s,"evaluation_fingerprint_id":"%s","replicate_values":{"%s":%s}}],"opaque_invocation_id":"%s","protocol_version":"kapso.task_evaluator.v1"}' \
    "$score" "$evaluation_fingerprint_id" "$replicate_id" "$score" "$opaque_invocation_id" \
    > /kapso/writable/result.json
"""


@dataclass(frozen=True)
class _RegistryResponse:
    payload: bytes
    content_type: str
    content_digest: str | None


class _ReadOnlyRegistryServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, responses: dict[str, _RegistryResponse]) -> None:
        super().__init__(("127.0.0.1", 0), _ReadOnlyRegistryHandler)
        self.responses = responses
        self.observations: list[tuple[str, str]] = []
        self.violations: list[str] = []
        self.observation_lock = Lock()

    @property
    def request_count(self) -> int:
        with self.observation_lock:
            return len(self.observations)

    @property
    def observed_violations(self) -> tuple[str, ...]:
        with self.observation_lock:
            return tuple(self.violations)


class _ReadOnlyRegistryHandler(BaseHTTPRequestHandler):
    server: _ReadOnlyRegistryServer

    def do_HEAD(self) -> None:
        self._serve(include_payload=False)

    def do_GET(self) -> None:
        self._serve(include_payload=True)

    def _serve(self, *, include_payload: bool) -> None:
        request_target = urlsplit(self.path)
        request_path = request_target.path
        with self.server.observation_lock:
            self.server.observations.append((self.command, self.path))
        sensitive_headers = tuple(
            header for header in _SENSITIVE_REQUEST_HEADERS if header in self.headers
        )
        if request_target.query or sensitive_headers:
            with self.server.observation_lock:
                self.server.violations.append(
                    f"unsafe registry request {self.command} {self.path}"
                )
            self.send_error(400)
            return
        response = self.server.responses.get(request_path)
        if response is None:
            with self.server.observation_lock:
                self.server.violations.append(
                    f"unsupported registry request {self.command} {self.path}"
                )
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Docker-Distribution-API-Version", "registry/2.0")
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(len(response.payload)))
        if response.content_digest is not None:
            self.send_header("Docker-Content-Digest", response.content_digest)
        self.end_headers()
        if include_payload:
            self.wfile.write(response.payload)

    def log_message(self, format_string: str, *arguments: object) -> None:
        return


@dataclass(frozen=True)
class _LocalOciRegistry:
    server: _ReadOnlyRegistryServer
    repository: str
    manifest_digest: str
    config_digest: str

    @property
    def image_reference(self) -> str:
        return f"{self.repository}@{self.manifest_digest}"


def _deterministic_busybox_layer(busybox_bytes: bytes) -> bytes:
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w", format=tarfile.USTAR_FORMAT) as layer:
        bin_directory = tarfile.TarInfo("bin")
        bin_directory.type = tarfile.DIRTYPE
        bin_directory.mode = 0o755
        bin_directory.uid = 0
        bin_directory.gid = 0
        bin_directory.mtime = 0
        layer.addfile(bin_directory)

        busybox = tarfile.TarInfo("bin/busybox")
        busybox.mode = 0o755
        busybox.uid = 0
        busybox.gid = 0
        busybox.mtime = 0
        busybox.size = len(busybox_bytes)
        layer.addfile(busybox, BytesIO(busybox_bytes))
    return archive.getvalue()


def _start_local_oci_registry(
    cleanup: ExitStack,
    busybox_bytes: bytes,
) -> _LocalOciRegistry:
    layer = _deterministic_busybox_layer(busybox_bytes)
    layer_digest = tree_or_blob_digest(layer)
    image_config = canonical_json_bytes(
        {
            "architecture": "amd64",
            "config": {"Env": ["LANG=C", "PATH=/bin"]},
            "os": "linux",
            "rootfs": {"diff_ids": [layer_digest], "type": "layers"},
        }
    )
    config_digest = tree_or_blob_digest(image_config)
    manifest = canonical_json_bytes(
        {
            "config": {
                "digest": config_digest,
                "mediaType": _OCI_CONFIG_MEDIA_TYPE,
                "size": len(image_config),
            },
            "layers": [
                {
                    "digest": layer_digest,
                    "mediaType": _OCI_LAYER_MEDIA_TYPE,
                    "size": len(layer),
                }
            ],
            "mediaType": _OCI_MANIFEST_MEDIA_TYPE,
            "schemaVersion": 2,
        }
    )
    manifest_digest = tree_or_blob_digest(manifest)
    repository_api_path = f"/v2/{_REGISTRY_REPOSITORY_PATH}"
    server = _ReadOnlyRegistryServer(
        {
            "/v2/": _RegistryResponse(
                payload=b"{}",
                content_type="application/json",
                content_digest=None,
            ),
            f"{repository_api_path}/manifests/{manifest_digest}": (
                _RegistryResponse(
                    payload=manifest,
                    content_type=_OCI_MANIFEST_MEDIA_TYPE,
                    content_digest=manifest_digest,
                )
            ),
            f"{repository_api_path}/blobs/{config_digest}": _RegistryResponse(
                payload=image_config,
                content_type="application/octet-stream",
                content_digest=config_digest,
            ),
            f"{repository_api_path}/blobs/{layer_digest}": _RegistryResponse(
                payload=layer,
                content_type="application/octet-stream",
                content_digest=layer_digest,
            ),
        }
    )
    server_thread = Thread(
        target=server.serve_forever,
        name="kapso-source-replay-oci-registry",
    )
    server_thread.start()
    cleanup.callback(server.server_close)
    cleanup.callback(server_thread.join)
    cleanup.callback(server.shutdown)
    registry_port = server.server_address[1]
    return _LocalOciRegistry(
        server=server,
        repository=f"127.0.0.1:{registry_port}/{_REGISTRY_REPOSITORY_PATH}",
        manifest_digest=manifest_digest,
        config_digest=config_digest,
    )


def _cleanup_replay_daemon_resources(
    settings,
    docker_config_root: Path,
    execution_store: ExpertSourceReplayExecutionStore,
    reservation,
    prepared_request,
) -> None:
    with execution_store.reservation_session(
        reservation=reservation,
        prepared_request=prepared_request,
    ) as session:
        provider_handle_ids = tuple(
            event.provider_execution_handle.provider_handle_id
            for event in session.events
            if event.event_kind is SourceReplayExecutionJournalEventKind.SPAWN_COMMITTED
        )
    cleanup_daemon_resources(
        settings,
        docker_config_root,
        provider_handle_ids,
    )


def test_real_docker_executes_both_journal_owned_replay_legs(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    validation_settings = _validation_policy()
    provider_settings = validation_settings.task_evaluation_provider
    busybox_bytes = read_verified_root_executable(
        Path(provider_settings.runtime.helper_executable_path),
        provider_settings.runtime.helper_executable_digest,
    )

    with ExitStack() as cleanup:
        local_registry = _start_local_oci_registry(cleanup, busybox_bytes)
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
        source_contents = {
            "adapter.py": _ADAPTER_SOURCE,
            "requirements.lock": TASK_ADAPTER_RUNTIME_LOCK,
        }
        records = build_records(
            task_adapter_runtime=runtime_contract,
            task_adapter_source_contents=source_contents,
        )
        adapter_manifest = next(
            record for record in records if isinstance(record, TaskAdapterManifest)
        )
        source_adapter = verified_test_task_adapter(
            adapter_manifest,
            source_contents=source_contents,
        )
        fixture = _request_fixture(
            tmp_path,
            contract_records=records,
            source_adapter=source_adapter,
        )
        prepared_request = _prepared(fixture)

        docker_config_root = tmp_path / "setup-docker-config"
        docker_config_root.mkdir(mode=0o700)
        docker_config_path = docker_config_root / "config.json"
        docker_config_path.write_bytes(b'{"auths":{}}\n')
        docker_config_path.chmod(0o400)
        cleanup.callback(
            remove_exact_image,
            provider_settings.runtime,
            docker_config_root,
            local_registry.image_reference,
        )
        pull_result = run_setup_docker(
            provider_settings.runtime,
            docker_config_root,
            (
                "image",
                "pull",
                "--platform",
                "linux/amd64",
                local_registry.image_reference,
            ),
        )
        require_setup_docker_success(pull_result, "source-replay")
        assert local_registry.server.observed_violations == ()
        registry_requests_after_pull = local_registry.server.request_count

        initial_snapshot = fixture.validation_store.snapshot(
            prepared_request.request.candidate_id
        )
        assert initial_snapshot is not None
        execution_store = ExpertSourceReplayExecutionStore(
            (fixture.validation_store.root / "source-replay-executions").resolve(),
            fixture.validation_store.root,
            prepared_request.settings.policy,
        )
        committed = fixture.validation_store.reserve_source_replay(
            expected_transition_id=initial_snapshot.transition.transition_id,
            prepared_request=prepared_request,
        )
        cleanup.callback(
            _cleanup_replay_daemon_resources,
            provider_settings.runtime,
            docker_config_root,
            execution_store,
            committed.reservation,
            prepared_request,
        )
        authority_coordinator = _coordinator(
            fixture,
            prepared_request,
            execution_store,
        )
        publication_coordinator = ExpertSourceReplayDecisionPublicationCoordinator(
            validation_store=fixture.validation_store,
            execution_store=execution_store,
            current_release_authority=(authority_coordinator.current_release_authority),
            task_adapter_authority=fixture.adapter_provider,
            security_denylist_authority=(
                authority_coordinator.security_denylist_authority
            ),
        )
        provider_registries = []

        def provider_registry_factory(exact_prepared_request):
            registry = build_source_replay_docker_provider_registry(
                prepared_request=exact_prepared_request,
                workspace_root=tmp_path.resolve(),
            )
            provider_registries.append(registry)
            return registry

        orchestrator = ExpertSourceReplayStageOrchestrator(
            validation_store=fixture.validation_store,
            preflight_coordinator=fixture.coordinator,
            execution_store=execution_store,
            provider_registry_factory=provider_registry_factory,
            spawn_authority_coordinator=authority_coordinator,
            publication_coordinator=publication_coordinator,
        )
        final_snapshot = orchestrator.run(fixture.attempt)
        replayed_snapshot = orchestrator.run(fixture.attempt)
        with execution_store.reservation_session(
            reservation=committed.reservation,
            prepared_request=prepared_request,
        ) as session:
            completed_execution = session.completed_execution()
        stage_result = final_snapshot.accepted_stage_results[-1]
        comparison_receipt = stage_result.paired_comparison_receipt
        fingerprint_comparison = comparison_receipt.case_comparisons[
            0
        ].fingerprint_comparisons[0]
        provider_handle_ids = tuple(
            event.provider_execution_handle.provider_handle_id
            for event in completed_execution.events
            if event.event_kind is SourceReplayExecutionJournalEventKind.SPAWN_COMMITTED
        )

        assert final_snapshot.state.promotion_state is ExpertPromotionState.VALIDATING
        assert final_snapshot.state.next_stage is ExpertValidationStage.AUTOMATED_REVIEW
        assert replayed_snapshot == final_snapshot
        assert len(provider_registries) == 1
        assert fingerprint_comparison.control_result.aggregate_value == 0.7
        assert fingerprint_comparison.candidate_result.aggregate_value == 0.8
        assert fingerprint_comparison.aggregate_raw_delta == 0.8 - 0.7
        assert fingerprint_comparison.aggregate_direction_aligned_delta == 0.8 - 0.7
        assert fingerprint_comparison.aggregate_normalized_effect == 0.8 - 0.7
        assert stage_result.stage_decision.outcome is ExpertEvaluatorOutcome.PASSED
        assert stage_result.stage_decision.hard_regression_comparisons == ()
        assert local_registry.server.request_count == registry_requests_after_pull
        assert local_registry.server.observed_violations == ()
        assert_no_daemon_resources(
            provider_settings.runtime,
            docker_config_root,
            tuple(provider_handle_ids),
            "source replay",
        )
        configured_provider_root = (
            tmp_path / provider_settings.workspace_path
        ).resolve()
        assert tuple(configured_provider_root.glob("replay-*")) == ()
        assert os.geteuid() == provider_settings.container_user_id
        assert os.getegid() == provider_settings.container_group_id
