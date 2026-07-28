"""Thin operational composition over the sealed cross-run services."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from kapso.core.config import load_effective_config
from kapso.cross_run.canonical import canonical_json_bytes, parse_json_bytes
from kapso.cross_run.capture.exporter import RunCaptureRequest
from kapso.cross_run.capture.pipeline import RunCaptureContext, RunCapturePipeline
from kapso.cross_run.catalog.service import CrossRunCatalog
from kapso.cross_run.contracts import (
    ArtifactEnvironment,
    CompletionState,
    EvaluationFingerprint,
    ExpertScopeContract,
    PublicationArtifactKind,
    TaskContextBinding,
)
from kapso.cross_run.github.command import GitHubCommandClient, SubprocessCommandRunner
from kapso.cross_run.github.materializer import GitHubArtifactMaterializer
from kapso.cross_run.github.publisher import AutonomousGitHubPublisher
from kapso.cross_run.github.resolver import GitHubArtifactResolver
from kapso.cross_run.knowledge.publisher import KnowledgeSnapshotPublisher
from kapso.cross_run.settings import CrossRunSettings


class CrossRunOperationError(ValueError):
    """An operational request is incomplete or crosses configured authority."""


_RETRIEVAL_POLICY_VERSION = "kapso.retrieval.v1"


@dataclass(frozen=True)
class GitHubOperationServices:
    """The existing GitHub resolver, materializer, and publication authority."""

    resolver: GitHubArtifactResolver
    materializer: GitHubArtifactMaterializer
    publisher: AutonomousGitHubPublisher


def inspect_cross_run(
    *,
    config_path: str,
    mode: str,
    scope_id: str,
    state_root: Path,
) -> Mapping[str, Any]:
    """Resolve every configured current release without materializing it."""

    settings = _settings(config_path, mode)
    services = _github_services(settings, state_root)
    artifacts = {
        kind.value: _resolved_artifact_summary(
            services.resolver.resolve_current(scope_id, kind)
        )
        for kind in _artifact_kinds()
    }
    repositories = settings.scopes.resolve(scope_id)
    return {
        "operation": "inspect",
        "scope_id": scope_id,
        "repository_binding_fingerprint": repositories.binding_fingerprint,
        "repositories": repositories.to_dict(),
        "artifacts": artifacts,
        "next_action": "verify",
    }


def verify_cross_run(
    *,
    config_path: str,
    mode: str,
    scope_id: str,
    state_root: Path,
) -> Mapping[str, Any]:
    """Resolve and fully materialize each configured current release."""

    settings = _settings(config_path, mode)
    services = _github_services(settings, state_root)
    artifacts = {}
    for kind in _artifact_kinds():
        resolved = services.resolver.resolve_current(scope_id, kind)
        materialized = services.materializer.materialize(resolved)
        artifacts[kind.value] = {
            **_resolved_artifact_summary(resolved),
            "cache_tree_digest": materialized.receipt.cache_tree_digest,
            "materialized_tree_digest": (materialized.receipt.materialized_tree_digest),
            "manifest_digest": materialized.receipt.manifest_digest,
            "asset_digests": dict(materialized.receipt.asset_digests),
            "cache_reused": materialized.reused,
        }
    return {
        "operation": "verify",
        "scope_id": scope_id,
        "artifacts": artifacts,
        "next_action": "resolve-launch",
    }


def capture_cross_run(
    *,
    config_path: str,
    mode: str,
    request_path: Path,
) -> Mapping[str, Any]:
    """Capture one complete stopped frontier from an explicit typed request."""

    settings = _settings(config_path, mode)
    request = _capture_request(request_path)
    stored = RunCapturePipeline(
        RunCaptureContext(request),
        settings,
    ).capture_if_due(CompletionState.STOPPED, force=True)
    if stored is None:
        raise CrossRunOperationError("forced capture returned no bundle")
    return {
        "operation": "capture",
        "scope_id": stored.manifest.scope_id,
        "run_id": stored.manifest.run_id,
        "campaign_id": stored.manifest.campaign_id,
        "bundle_id": stored.manifest.bundle_id,
        "capture_generation": stored.manifest.capture_generation,
        "completion_state": stored.manifest.completion_state.value,
        "artifact_digests": dict(stored.manifest.checksums),
        "next_action": "publish-knowledge",
    }


def publish_knowledge_cross_run(
    *,
    config_path: str,
    mode: str,
    request_path: Path,
    state_root: Path,
) -> Mapping[str, Any]:
    """Build and publish S(n+1) from one explicit catalog and parent."""

    settings = _settings(config_path, mode)
    request = _object_request(
        request_path,
        {
            "catalog_root",
            "scope_contract",
            "expected_parent_sha",
            "expected_current_snapshot_id",
            "committed_at",
            "validation_closure_ids",
        },
    )
    scope_contract = ExpertScopeContract.from_dict(request["scope_contract"])
    catalog_root = _request_path(request_path, request["catalog_root"])
    catalog = CrossRunCatalog(catalog_root, scope_contract, settings.catalog)
    generation = catalog.store.read_current()
    parent_id = request["expected_current_snapshot_id"]
    if parent_id is not None and not isinstance(parent_id, str):
        raise CrossRunOperationError("expected_current_snapshot_id is invalid")
    validation_ids = request["validation_closure_ids"]
    if not isinstance(validation_ids, list) or any(
        not isinstance(item, str) for item in validation_ids
    ):
        raise CrossRunOperationError("validation_closure_ids are invalid")
    services = _github_services(settings, state_root)
    publisher = KnowledgeSnapshotPublisher(
        services.publisher,
        settings.github,
        settings.knowledge,
    )
    parent_ids = () if parent_id is None else (parent_id,)
    built = publisher.build(
        scope_contract,
        generation,
        catalog.store.read_object_bytes,
        parent_snapshot_ids=parent_ids,
        sanitation_policy_version=settings.sanitation.policy_version,
        retrieval_policy_version=_RETRIEVAL_POLICY_VERSION,
        published_at=_required_text(request["committed_at"], "committed_at"),
        publisher_attestation={"issuer": settings.github.publisher_login},
    )
    publication = publisher.publish(
        built.package,
        expected_parent_sha=_required_text(
            request["expected_parent_sha"], "expected_parent_sha"
        ),
        expected_current_snapshot_id=parent_id,
        committed_at=_required_text(request["committed_at"], "committed_at"),
        validation_closure_ids=tuple(sorted(validation_ids)),
    )
    record = publication.telemetry.publication_record
    telemetry = built.embedding_telemetry
    return {
        "operation": "publish-knowledge",
        "scope_id": scope_contract.scope_id,
        "snapshot_id": publication.package.manifest.snapshot_id,
        "catalog_generation_id": generation.catalog_generation_id,
        "catalog_generation": generation.generation_number,
        "commit_sha": record.commit_sha,
        "release_tag": record.tag,
        "asset_digests": {asset.name: asset.sha256 for asset in record.assets},
        "embedding": (
            None
            if telemetry is None
            else {
                "provider": telemetry.provider,
                "model": telemetry.model,
                "call_count": telemetry.call_count,
                "input_tokens": telemetry.input_tokens,
            }
        ),
        "next_action": "propose-expert",
    }


def operation_json(result: Mapping[str, Any]) -> bytes:
    """Render one canonical non-secret operational result."""

    if not isinstance(result, Mapping):
        raise CrossRunOperationError("operation result must be an object")
    return canonical_json_bytes(dict(result)) + b"\n"


def _settings(config_path: str, mode: str) -> CrossRunSettings:
    effective = load_effective_config(config_path, mode)
    if type(effective.cross_run) is not CrossRunSettings:
        raise CrossRunOperationError("selected configuration has no cross-run settings")
    return effective.cross_run


def _github_services(
    settings: CrossRunSettings,
    state_root: Path,
) -> GitHubOperationServices:
    root = _private_state_root(state_root)
    client = GitHubCommandClient(
        SubprocessCommandRunner(),
        working_directory=root,
        timeout_seconds=settings.github.command_timeout_seconds,
        api_version=settings.github.api_version,
        minimum_cli_version=settings.github.minimum_cli_version,
        control_blob_size_bytes=settings.github.control_blob_size_bytes,
    )
    resolver = GitHubArtifactResolver(client, settings.github, settings.scopes)
    materializer = GitHubArtifactMaterializer(client, settings.github, root)
    return GitHubOperationServices(
        resolver=resolver,
        materializer=materializer,
        publisher=AutonomousGitHubPublisher(
            client,
            resolver,
            materializer,
            settings.github,
        ),
    )


def _private_state_root(state_root: Path) -> Path:
    if not isinstance(state_root, Path):
        raise CrossRunOperationError("state_root must be a path")
    root = Path(os.path.abspath(state_root))
    if root in {Path("/"), Path.home()}:
        raise CrossRunOperationError("state_root is unsafe")
    if os.path.lexists(root):
        metadata = root.stat(follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise CrossRunOperationError("state_root must be a real directory")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise CrossRunOperationError("state_root must be owner-private")
    else:
        root.mkdir(parents=True, mode=0o700)
    return root


def _artifact_kinds() -> tuple[PublicationArtifactKind, ...]:
    return (
        PublicationArtifactKind.EXPERT_BASE_RELEASE,
        PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
        PublicationArtifactKind.SECURITY_DENYLIST,
    )


def _resolved_artifact_summary(resolved) -> Mapping[str, Any]:
    record = resolved.pointer.publication_record
    return {
        "artifact_id": record.artifact_id,
        "repository": record.repository_full_name,
        "repository_node_id": record.repository_node_id,
        "pointer_commit_sha": resolved.pointer_commit_sha,
        "source_commit_sha": record.commit_sha,
        "release_tag": record.tag,
        "release_id": record.immutable_release_id,
        "release_attestation_ref": record.release_attestation_ref,
        "asset_digests": {asset.name: asset.sha256 for asset in record.assets},
    }


def _object_request(path: Path, expected_fields: set[str]) -> Mapping[str, Any]:
    normalized = path.expanduser().resolve(strict=True)
    parsed = parse_json_bytes(normalized.read_bytes())
    if not isinstance(parsed, Mapping) or set(parsed) != expected_fields:
        raise CrossRunOperationError("operation request fields are invalid")
    return parsed


def _capture_request(path: Path) -> RunCaptureRequest:
    request = _object_request(
        path,
        {
            "workspace_dir",
            "idea_archive_path",
            "scope_contract_id",
            "scope_id",
            "run_id",
            "campaign_id",
            "configuration_fingerprint",
            "completion_state",
            "started_at",
            "kapso_commit",
            "launch_manifest_id",
            "knowledge_snapshot_id",
            "expert_base_release_id",
            "task_context_binding",
            "artifact_environment",
            "evaluation_fingerprints",
            "run_log_paths",
        },
    )
    fingerprints = request["evaluation_fingerprints"]
    logs = request["run_log_paths"]
    if not isinstance(fingerprints, list) or not isinstance(logs, list):
        raise CrossRunOperationError("capture request arrays are invalid")
    completion = request["completion_state"]
    if completion != CompletionState.STOPPED.value:
        raise CrossRunOperationError("operational capture requires stopped state")
    return RunCaptureRequest(
        workspace_dir=_request_path(path, request["workspace_dir"]),
        idea_archive_path=_request_path(path, request["idea_archive_path"]),
        scope_contract_id=_required_text(
            request["scope_contract_id"], "scope_contract_id"
        ),
        scope_id=_required_text(request["scope_id"], "scope_id"),
        run_id=_required_text(request["run_id"], "run_id"),
        campaign_id=_required_text(request["campaign_id"], "campaign_id"),
        configuration_fingerprint=_required_text(
            request["configuration_fingerprint"], "configuration_fingerprint"
        ),
        completion_state=CompletionState(completion),
        started_at=_required_text(request["started_at"], "started_at"),
        kapso_commit=_required_text(request["kapso_commit"], "kapso_commit"),
        launch_manifest_id=_required_text(
            request["launch_manifest_id"], "launch_manifest_id"
        ),
        knowledge_snapshot_id=_required_text(
            request["knowledge_snapshot_id"], "knowledge_snapshot_id"
        ),
        expert_base_release_id=_required_text(
            request["expert_base_release_id"], "expert_base_release_id"
        ),
        task_context_binding=TaskContextBinding.from_dict(
            request["task_context_binding"]
        ),
        artifact_environment=ArtifactEnvironment.from_dict(
            request["artifact_environment"]
        ),
        evaluation_fingerprints=tuple(
            EvaluationFingerprint.from_dict(item) for item in fingerprints
        ),
        run_log_paths=tuple(_required_text(item, "run_log_path") for item in logs),
    )


def _request_path(request_path: Path, value: object) -> Path:
    text = _required_text(value, "request path")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = request_path.expanduser().resolve(strict=True).parent / candidate
    return candidate.resolve(strict=True)


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CrossRunOperationError(f"{name} must be non-empty text")
    return value
