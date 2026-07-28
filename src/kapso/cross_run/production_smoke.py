"""One staged production driver over the real cross-run trust boundaries."""

from __future__ import annotations

import io
import os
import stat
import tarfile
import tempfile
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from kapso.core.config import load_effective_config
from kapso.core.embedding_contracts import EmbeddingSettings
from kapso.core.embedding_provider import OpenAIEmbeddingProvider
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    normalize_utc_timestamp,
    parse_json_bytes,
    tree_or_blob_digest,
)
from kapso.cross_run.catalog.service import CrossRunCatalog
from kapso.cross_run.catalog.projector import ProjectionResult
from kapso.cross_run.contracts import (
    ArtifactCompleteness,
    ArtifactEnvironment,
    CompletionState,
    ExpertScopeContract,
    PriorIdea,
    PriorIdeaStatus,
    PublicationArtifactKind,
    RunBundle,
    SECURITY_DENYLIST_EVIDENCE_FILENAME,
    SECURITY_DENYLIST_POLICY_VERSION,
    SECURITY_DENYLIST_SCHEMA_VERSION,
    SecurityDenylistEvidenceBundle,
    SecurityDenylistSnapshot,
    TaskContextBinding,
)
from kapso.cross_run.docker.runtime import DockerImageAuthority, PinnedDockerRuntime
from kapso.cross_run.github.publisher import PublicationEnvelope, ReleaseAssetInput
from kapso.cross_run.github.resolver import security_denylist_tag
from kapso.cross_run.knowledge.package import KnowledgeSnapshotPackage
from kapso.cross_run.knowledge.publisher import KnowledgeSnapshotPublisher
from kapso.cross_run.operations import (
    GitHubOperationServices,
    _github_services,
    _private_state_root,
)
from kapso.cross_run.record_contracts import (
    SANITATION_REPORT_SCHEMA,
    SANITATION_SCANNER_VERSION,
    SanitationReport,
)
from kapso.cross_run.security_denylist import (
    AuthenticatedSecurityDenylistAuthority,
    GitHubSecurityDenylistSnapshotProvider,
    SecurityDenylistCheckpointStore,
    SecurityDenylistPublisher,
)
from kapso.cross_run.settings import CrossRunSettings


class ProductionSmokeError(ValueError):
    """The selected production stage or its durable receipt is invalid."""


_STAGE_ORDER = (
    "preflight",
    "bootstrap-authorities",
    "github-read",
    "embeddings",
    "docker-authority",
    "knowledge-publication",
)
_FIXTURE_FILENAME = "transport-smoke.json"
_RECEIPT_FILENAME = "production-smoke-receipt.json"
_RECEIPT_STAGING_FILENAME = ".production-smoke-receipt.next"
_RETRIEVAL_POLICY_VERSION = "kapso.retrieval.v1"
_SECURITY_MANIFEST_FILENAME = "security-denylist.json"
_SECURITY_ASSET_FILENAME = "security-denylist.tar"


def production_smoke_stage_names() -> tuple[str, ...]:
    """Return the canonical selectable production stage order."""

    return _STAGE_ORDER


def run_production_smoke(
    *,
    config_path: str,
    mode: str,
    state_root: Path,
    stages: tuple[str, ...],
) -> Mapping[str, Any]:
    """Run selected stages in canonical order and checkpoint each receipt."""

    effective = load_effective_config(config_path, mode)
    settings = effective.cross_run
    if type(settings) is not CrossRunSettings:
        raise ProductionSmokeError("selected configuration has no cross-run settings")
    selected = _validate_stages(stages)
    fixture, fixture_digest = _load_fixture(settings)
    scope_contract = ExpertScopeContract.from_dict(fixture["scope_contract"])
    settings.scopes.resolve(scope_contract.scope_id)
    root = _private_state_root(state_root)
    smoke_root = _private_state_root(root / settings.production_validation.state_path)
    receipt_path = smoke_root / _RECEIPT_FILENAME
    receipt = _read_receipt(
        receipt_path,
        maximum_bytes=settings.production_validation.receipt_size_bytes,
        configuration_fingerprint=settings.configuration_fingerprint,
        fixture_digest=fixture_digest,
        scope_id=scope_contract.scope_id,
    )
    completed = {item["stage"] for item in receipt["stage_receipts"]}
    for stage in selected:
        if stage in completed:
            continue
        started_at = _timestamp()
        evidence = _run_stage(
            stage=stage,
            settings=settings,
            smoke_root=smoke_root,
            fixture=fixture,
            scope_contract=scope_contract,
        )
        completed_at = _timestamp()
        stage_content = {
            "stage": stage,
            "started_at": started_at,
            "completed_at": completed_at,
            "evidence": dict(evidence),
        }
        stage_receipt = {
            "stage_receipt_id": content_id(
                "production-smoke-stage-receipt",
                stage_content,
            ),
            **stage_content,
        }
        receipt = _append_stage_receipt(receipt, stage_receipt)
        _write_receipt(
            receipt_path,
            receipt,
            maximum_bytes=settings.production_validation.receipt_size_bytes,
        )
        completed.add(stage)
    return receipt


def _run_stage(
    *,
    stage: str,
    settings: CrossRunSettings,
    smoke_root: Path,
    fixture: Mapping[str, Any],
    scope_contract: ExpertScopeContract,
) -> Mapping[str, Any]:
    if stage == "preflight":
        return _preflight(settings, smoke_root, scope_contract.scope_id)
    if stage == "bootstrap-authorities":
        return _bootstrap_authorities(
            settings,
            smoke_root,
            fixture,
            scope_contract,
        )
    if stage == "github-read":
        return _github_read(settings, smoke_root, scope_contract)
    if stage == "embeddings":
        return _embedding_smoke(settings, fixture)
    if stage == "docker-authority":
        return _docker_authority_smoke(settings, smoke_root)
    if stage == "knowledge-publication":
        return _knowledge_publication_smoke(
            settings,
            smoke_root,
            fixture,
            scope_contract,
        )
    raise ProductionSmokeError("unknown production smoke stage")


def _preflight(
    settings: CrossRunSettings,
    smoke_root: Path,
    scope_id: str,
) -> Mapping[str, Any]:
    github = _github_services(settings, smoke_root)
    repositories = {}
    for kind in _artifact_kinds():
        report = github.resolver.diagnose_repository(scope_id, kind)
        repositories[kind.value] = {
            "repository": report.repository_full_name,
            "repository_node_id": report.repository_node_id,
            "private": report.private,
            "write_access": report.write_access,
            "immutable_releases": report.immutable_releases,
            "authenticated_actor": report.authenticated_actor,
        }
    credential_path = Path(settings.launch.coding_agent_codex_auth_source_path)
    metadata = credential_path.stat(follow_symlinks=False)
    if (
        not credential_path.is_absolute()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_size <= 0
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise ProductionSmokeError(
            "configured coding-agent credential source is not a private regular file"
        )
    image = settings.launch.coding_agent_image
    return {
        "repositories": repositories,
        "coding_agent_credential": {
            "present": True,
            "private": True,
        },
        "coding_agent_image": (
            None
            if image is None
            else {
                "image_reference": image.image_reference,
                "image_config_digest": image.image_config_digest,
                "operating_system": image.operating_system,
                "architecture": image.architecture,
                "architecture_variant": image.architecture_variant,
            }
        ),
    }


def _bootstrap_authorities(
    settings: CrossRunSettings,
    smoke_root: Path,
    fixture: Mapping[str, Any],
    scope_contract: ExpertScopeContract,
) -> Mapping[str, Any]:
    if not settings.production_validation.github_write_smoke:
        raise ProductionSmokeError("GitHub write smoke is disabled in configuration")
    github = _github_services(settings, smoke_root)
    security = _bootstrap_security(
        settings,
        github,
        smoke_root,
        scope_contract,
        fixture["committed_at"],
    )
    knowledge = _bootstrap_knowledge(
        settings,
        github,
        smoke_root,
        scope_contract,
        fixture["committed_at"],
    )
    return {"security": security, "knowledge": knowledge}


def _bootstrap_security(
    settings: CrossRunSettings,
    github: GitHubOperationServices,
    smoke_root: Path,
    scope_contract: ExpertScopeContract,
    committed_at: object,
) -> Mapping[str, Any]:
    state = github.resolver.read_current_pointer_state(
        scope_contract.scope_id,
        PublicationArtifactKind.SECURITY_DENYLIST,
        allow_missing=True,
    )
    if state.pointer is not None:
        resolved = github.resolver.resolve_current(
            scope_contract.scope_id,
            PublicationArtifactKind.SECURITY_DENYLIST,
        )
        materialized = github.materializer.materialize(resolved)
        snapshot = GitHubSecurityDenylistSnapshotProvider(
            github.resolver,
            github.materializer,
        ).resolve_current(scope_contract.scope_id)
        if snapshot.snapshot.scope_contract_id != scope_contract.scope_contract_id:
            raise ProductionSmokeError(
                "security CURRENT belongs to another scope contract"
            )
        return {
            "artifact_id": snapshot.snapshot.snapshot_id,
            "publication_id": snapshot.publication_id,
            "commit_sha": snapshot.authority_commit_sha,
            "cache_reused": materialized.reused,
            "created": False,
        }
    evidence = SecurityDenylistEvidenceBundle.mint(evidence=())
    repositories = settings.scopes.resolve(scope_contract.scope_id)
    snapshot = SecurityDenylistSnapshot.mint(
        schema_version=SECURITY_DENYLIST_SCHEMA_VERSION,
        policy_version=SECURITY_DENYLIST_POLICY_VERSION,
        scope_id=scope_contract.scope_id,
        scope_contract_id=scope_contract.scope_contract_id,
        scope_repository_binding_hash=repositories.binding_fingerprint,
        generation=0,
        predecessor_snapshot_id=None,
        evidence_bundle_id=evidence.evidence_bundle_id,
        evidence_source_ids=evidence.source_ids,
        revocations=(),
        exact_dependency_ids=tuple(
            sorted({scope_contract.scope_contract_id, evidence.evidence_bundle_id})
        ),
        checksums={
            SECURITY_DENYLIST_EVIDENCE_FILENAME: tree_or_blob_digest(
                evidence.to_json_bytes()
            )
        },
    )
    normalized_committed_at = _committed_at(committed_at)
    with tempfile.TemporaryDirectory(
        prefix="security-bootstrap-",
        dir=smoke_root,
    ) as temporary:
        temporary_root = Path(temporary)
        source = temporary_root / "source"
        source.mkdir(mode=0o700)
        manifest_path = source / _SECURITY_MANIFEST_FILENAME
        evidence_path = source / SECURITY_DENYLIST_EVIDENCE_FILENAME
        manifest_path.write_bytes(snapshot.to_json_bytes())
        evidence_path.write_bytes(evidence.to_json_bytes())
        asset_path = temporary_root / _SECURITY_ASSET_FILENAME
        _write_security_archive(asset_path, snapshot, evidence)
        asset_payload = asset_path.read_bytes()
        envelope = PublicationEnvelope(
            artifact_kind=PublicationArtifactKind.SECURITY_DENYLIST,
            artifact_id=snapshot.snapshot_id,
            scope_id=scope_contract.scope_id,
            expected_parent_sha=state.head_commit_sha,
            source_tree=source,
            manifest_relative_path=_SECURITY_MANIFEST_FILENAME,
            assets=(
                ReleaseAssetInput(
                    path=asset_path,
                    name=_SECURITY_ASSET_FILENAME,
                    media_type="application/x-tar",
                    size=len(asset_payload),
                    sha256=tree_or_blob_digest(asset_payload),
                ),
            ),
            tag=security_denylist_tag(settings.github, snapshot.generation),
            committed_at=normalized_committed_at,
            validation_closure_ids=tuple(
                sorted({snapshot.snapshot_id, *snapshot.exact_dependency_ids})
            ),
        )
        provider = GitHubSecurityDenylistSnapshotProvider(
            github.resolver,
            github.materializer,
        )
        telemetry = SecurityDenylistPublisher(
            github.publisher,
            github.resolver,
            provider,
            settings.launch,
        ).publish(envelope)
    record = telemetry.publication_record
    return {
        "artifact_id": snapshot.snapshot_id,
        "publication_id": record.publication_id,
        "commit_sha": telemetry.pointer_commit_sha,
        "release_tag": record.tag,
        "asset_digests": {asset.name: asset.sha256 for asset in record.assets},
        "created": True,
    }


def _bootstrap_knowledge(
    settings: CrossRunSettings,
    github: GitHubOperationServices,
    smoke_root: Path,
    scope_contract: ExpertScopeContract,
    committed_at: object,
) -> Mapping[str, Any]:
    state = github.resolver.read_current_pointer_state(
        scope_contract.scope_id,
        PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
        allow_missing=True,
    )
    if state.pointer is not None:
        resolved = github.resolver.resolve_current(
            scope_contract.scope_id,
            PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
        )
        materialized = github.materializer.materialize(resolved)
        package = KnowledgeSnapshotPackage.open(materialized.content)
        if package.manifest.scope_contract_id != scope_contract.scope_contract_id:
            raise ProductionSmokeError(
                "knowledge CURRENT belongs to another scope contract"
            )
        record = resolved.pointer.publication_record
        return {
            "artifact_id": package.manifest.snapshot_id,
            "publication_id": record.publication_id,
            "commit_sha": resolved.pointer_commit_sha,
            "cache_reused": materialized.reused,
            "created": False,
        }
    catalog = CrossRunCatalog(
        smoke_root / "catalog",
        scope_contract,
        settings.catalog,
    )
    generation = catalog.store.read_current()
    normalized_committed_at = _committed_at(committed_at)
    publisher = KnowledgeSnapshotPublisher(
        github.publisher,
        settings.github,
        settings.knowledge,
    )
    built = publisher.build(
        scope_contract,
        generation,
        catalog.store.read_object_bytes,
        parent_snapshot_ids=(),
        sanitation_policy_version=settings.sanitation.policy_version,
        retrieval_policy_version=_RETRIEVAL_POLICY_VERSION,
        published_at=normalized_committed_at,
        publisher_attestation={"issuer": settings.github.publisher_login},
    )
    publication = publisher.publish(
        built.package,
        expected_parent_sha=state.head_commit_sha,
        expected_current_snapshot_id=None,
        committed_at=normalized_committed_at,
        validation_closure_ids=(),
    )
    record = publication.telemetry.publication_record
    return {
        "artifact_id": publication.package.manifest.snapshot_id,
        "publication_id": record.publication_id,
        "commit_sha": publication.telemetry.pointer_commit_sha,
        "release_tag": record.tag,
        "asset_digests": {asset.name: asset.sha256 for asset in record.assets},
        "created": True,
    }


def _github_read(
    settings: CrossRunSettings,
    smoke_root: Path,
    scope_contract: ExpertScopeContract,
) -> Mapping[str, Any]:
    github = _github_services(settings, smoke_root / "clean-read")
    artifacts = {}
    for kind in _artifact_kinds():
        state = github.resolver.read_current_pointer_state(
            scope_contract.scope_id,
            kind,
            allow_missing=True,
        )
        if state.pointer is None:
            artifacts[kind.value] = {"present": False}
            continue
        resolved = github.resolver.resolve_current(scope_contract.scope_id, kind)
        first = github.materializer.materialize(resolved)
        second = github.materializer.materialize(resolved)
        if first.receipt != second.receipt or not second.reused:
            raise ProductionSmokeError(
                "clean GitHub materialization is not stable and reusable"
            )
        record = resolved.pointer.publication_record
        artifacts[kind.value] = {
            "present": True,
            "artifact_id": record.artifact_id,
            "publication_id": record.publication_id,
            "pointer_commit_sha": resolved.pointer_commit_sha,
            "materialized_tree_digest": second.receipt.materialized_tree_digest,
            "cache_reused": second.reused,
        }
    security_state = _private_state_root(smoke_root / "clean-security")
    security_path = security_state / settings.launch.security_denylist_state_path
    trusted_root = _private_state_root(security_path.parent)
    observation = AuthenticatedSecurityDenylistAuthority(
        settings.scopes,
        settings.launch,
        GitHubSecurityDenylistSnapshotProvider(
            github.resolver,
            github.materializer,
        ),
        SecurityDenylistCheckpointStore(
            security_path,
            trusted_root,
            settings.launch.security_denylist_checkpoint_size_bytes,
        ),
    ).observe_exact(
        scope_id=scope_contract.scope_id,
        scope_contract_id=scope_contract.scope_contract_id,
        checked_subject_ids=(scope_contract.scope_contract_id,),
    )
    return {
        "artifacts": artifacts,
        "security_observation_id": observation.observation_id,
        "security_snapshot_id": observation.snapshot_id,
        "security_generation": observation.generation,
    }


def _embedding_smoke(
    settings: CrossRunSettings,
    fixture: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not settings.production_validation.embedding_smoke:
        raise ProductionSmokeError("embedding smoke is disabled in configuration")
    values = fixture["embedding_inputs"]
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value for value in values)
    ):
        raise ProductionSmokeError("production embedding inputs are invalid")
    configured = settings.knowledge.embeddings
    provider_settings = EmbeddingSettings(
        enabled=configured.enabled,
        provider=configured.provider,
        model=configured.model,
        dimensions=configured.dimensions,
        batch_size=configured.batch_size,
        timeout_seconds=configured.timeout_seconds,
        max_retries=configured.max_retries,
        canonicalizer_version=configured.canonicalizer_version,
    )
    provider = OpenAIEmbeddingProvider(provider_settings)
    first = provider.embed(tuple(values))
    second = provider.embed(tuple(values))
    first_identities = tuple(record.input_hash for record in first.records)
    second_identities = tuple(record.input_hash for record in second.records)
    first_vectors = tuple(record.vector for record in first.records)
    second_vectors = tuple(record.vector for record in second.records)
    if first_identities != second_identities or first_vectors != second_vectors:
        raise ProductionSmokeError(
            "OpenAI embedding rebuild changed input identities or vectors"
        )
    return {
        "provider": provider_settings.provider,
        "model": provider_settings.model,
        "dimensions": provider_settings.dimensions,
        "embedding_space_id": provider_settings.embedding_space_id.value,
        "input_hashes": first_identities,
        "vector_digest": tree_or_blob_digest(canonical_json_bytes(first_vectors)),
        "first_call_count": first.telemetry.call_count,
        "second_call_count": second.telemetry.call_count,
        "input_tokens": (first.telemetry.input_tokens + second.telemetry.input_tokens),
    }


def _docker_authority_smoke(
    settings: CrossRunSettings,
    smoke_root: Path,
) -> Mapping[str, Any]:
    runtime_root = _private_state_root(smoke_root / "docker-authority")
    runtime = PinnedDockerRuntime.create(
        trusted_root=runtime_root,
        settings=settings.docker,
    )
    runtime.issue_observation_authority().require_live_authority()
    image = settings.launch.coding_agent_image
    image_evidence = None
    if image is not None:
        image_authority = DockerImageAuthority.mint(
            image_reference=image.image_reference,
            image_config_digest=image.image_config_digest,
            operating_system=image.operating_system,
            architecture=image.architecture,
            architecture_variant=image.architecture_variant,
        )
        inspection = runtime.inspect_exact_image(image_authority)
        image_evidence = {
            "image_reference": image_authority.image_reference,
            "image_config_digest": image_authority.image_config_digest,
            "inspection_digest": tree_or_blob_digest(
                canonical_json_bytes(dict(inspection))
            ),
        }
    socket = os.stat(settings.docker.runtime_socket_path, follow_symlinks=False)
    mutation_lock = os.stat(
        settings.docker.runtime_mutation_lock_path,
        follow_symlinks=False,
    )
    return {
        "runtime_settings_digest": tree_or_blob_digest(settings.docker.to_json_bytes()),
        "socket_device": socket.st_dev,
        "socket_inode": socket.st_ino,
        "mutation_lock_device": mutation_lock.st_dev,
        "mutation_lock_inode": mutation_lock.st_ino,
        "image": image_evidence,
    }


def _knowledge_publication_smoke(
    settings: CrossRunSettings,
    smoke_root: Path,
    fixture: Mapping[str, Any],
    scope_contract: ExpertScopeContract,
) -> Mapping[str, Any]:
    """Publish one admitted synthetic bundle as the first non-empty snapshot."""

    projection = _synthetic_projection(settings, fixture, scope_contract)
    catalog = CrossRunCatalog(
        smoke_root / "catalog",
        scope_contract,
        settings.catalog,
    )
    generation = catalog.store.read_current()
    if generation.generation_number == 0:
        generation = catalog.publish_projection(generation, projection).generation
    elif (
        generation.generation_number != 1
        or projection.source_bundle.bundle_id not in generation.fact_object_ids
    ):
        raise ProductionSmokeError(
            "production catalog is not the expected synthetic generation"
        )

    github = _github_services(settings, smoke_root)
    current = github.resolver.resolve_current(
        scope_contract.scope_id,
        PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
    )
    current_materialized = github.materializer.materialize(current)
    current_package = KnowledgeSnapshotPackage.open(current_materialized.content)
    if (
        projection.source_bundle.bundle_id
        in current_package.manifest.included_bundle_ids
    ):
        if (
            current_package.manifest.catalog_generation != 1
            or current_package.manifest.scope_contract_id
            != scope_contract.scope_contract_id
        ):
            raise ProductionSmokeError(
                "recovered knowledge snapshot has another synthetic generation"
            )
        record = current.pointer.publication_record
        return {
            "snapshot_id": current_package.manifest.snapshot_id,
            "parent_snapshot_id": current_package.manifest.parent_snapshot_ids[0],
            "catalog_generation_id": generation.catalog_generation_id,
            "bundle_id": projection.source_bundle.bundle_id,
            "prior_idea_id": projection.prior_ideas[0].prior_idea_id,
            "publication_id": record.publication_id,
            "commit_sha": current.pointer_commit_sha,
            "release_tag": record.tag,
            "recovered": True,
        }
    if current_package.manifest.catalog_generation != 0:
        raise ProductionSmokeError(
            "knowledge CURRENT advanced beyond the expected EMPTY parent"
        )

    publisher = KnowledgeSnapshotPublisher(
        github.publisher,
        settings.github,
        settings.knowledge,
    )
    committed_at = _committed_at(fixture["committed_at"])
    built = publisher.build(
        scope_contract,
        generation,
        catalog.store.read_object_bytes,
        parent_snapshot_ids=(current_package.manifest.snapshot_id,),
        sanitation_policy_version=settings.sanitation.policy_version,
        retrieval_policy_version=_RETRIEVAL_POLICY_VERSION,
        published_at=committed_at,
        publisher_attestation={"issuer": settings.github.publisher_login},
    )
    publication = publisher.publish(
        built.package,
        expected_parent_sha=current.pointer_commit_sha,
        expected_current_snapshot_id=current_package.manifest.snapshot_id,
        committed_at=committed_at,
        validation_closure_ids=(),
    )
    resolved = github.resolver.resolve_current(
        scope_contract.scope_id,
        PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
    )
    materialized = github.materializer.materialize(resolved)
    published = KnowledgeSnapshotPackage.open(materialized.content)
    if published.manifest != publication.package.manifest:
        raise ProductionSmokeError(
            "published knowledge snapshot differs from clean materialization"
        )
    record = publication.telemetry.publication_record
    telemetry = built.embedding_telemetry
    return {
        "snapshot_id": published.manifest.snapshot_id,
        "parent_snapshot_id": current_package.manifest.snapshot_id,
        "catalog_generation_id": generation.catalog_generation_id,
        "bundle_id": projection.source_bundle.bundle_id,
        "prior_idea_id": projection.prior_ideas[0].prior_idea_id,
        "publication_id": record.publication_id,
        "commit_sha": publication.telemetry.pointer_commit_sha,
        "release_tag": record.tag,
        "embedding_call_count": None if telemetry is None else telemetry.call_count,
        "recovered": False,
    }


def _synthetic_projection(
    settings: CrossRunSettings,
    fixture: Mapping[str, Any],
    scope_contract: ExpertScopeContract,
) -> ProjectionResult:
    """Build the one public, domain-neutral transport-smoke bundle."""

    values = fixture["embedding_inputs"]
    if not isinstance(values, list) or len(values) < 2:
        raise ProductionSmokeError("synthetic projection inputs are incomplete")
    task_context = TaskContextBinding.mint(
        scope_contract_id=scope_contract.scope_contract_id,
        scope_id=scope_contract.scope_id,
        task_family_id="language_model_post_training",
        task_adapter_id="posttrain",
        capability_tags=("language.training",),
        input_contract_fingerprint=tree_or_blob_digest(values[0].encode("utf-8")),
        target_contract_fingerprint=tree_or_blob_digest(values[1].encode("utf-8")),
        starting_artifact_refs=(),
        method_fingerprint=tree_or_blob_digest(b"transport-smoke-method"),
        toolchain_fingerprint=tree_or_blob_digest(b"transport-smoke-toolchain"),
        dependency_runtime_fingerprint=tree_or_blob_digest(b"transport-smoke-runtime"),
        budget_hardware_envelope={"accelerator": "none", "hours": 1},
        transfer_dimensions={
            "dataset_family": "synthetic_public",
            "runtime_family": "python",
        },
    )
    task_context.validate_against(scope_contract)
    expert_release_id = content_id(
        "expert-base-release",
        {"transport_smoke": "source-release"},
    )
    environment = ArtifactEnvironment.mint(
        kapso_commit="0" * 40,
        expert_base_release_id=expert_release_id,
        task_adapter_manifest_id=content_id(
            "task-adapter-manifest",
            {"task_adapter_id": task_context.task_adapter_id},
        ),
        task_adapter_verification_receipt_id=content_id(
            "task-adapter-verification-receipt",
            {"task_adapter_id": task_context.task_adapter_id},
        ),
        starting_artifact_content_ids={},
        dependency_lock_hash=tree_or_blob_digest(b"transport-smoke-lock"),
    )
    checksums = {
        path: tree_or_blob_digest(f"synthetic:{path}".encode("utf-8"))
        for path in (
            "capture_descriptor.json",
            "checkpoint.json",
            "events.jsonl",
            "experiment_history.json",
            "idea_archive.json",
            "sanitation_report.json",
        )
    }
    committed_at = _committed_at(fixture["committed_at"])
    bundle = RunBundle.mint(
        scope_contract_id=scope_contract.scope_contract_id,
        scope_id=scope_contract.scope_id,
        run_id="production_smoke_run",
        campaign_id="production_smoke_campaign",
        completion_state=CompletionState.STOPPED,
        capture_generation=0,
        supersedes_bundle_id=None,
        checkpoint_frontier=1,
        capture_watermarks={"events": 1},
        configuration_fingerprint=settings.configuration_fingerprint,
        artifact_completeness={"checkpoint": ArtifactCompleteness.PRESENT},
        started_at=committed_at,
        captured_at=committed_at,
        kapso_commit=environment.kapso_commit,
        launch_manifest_id=content_id(
            "launch-manifest",
            {"transport_smoke": "first-launch"},
        ),
        knowledge_snapshot_id=content_id(
            "knowledge-snapshot",
            {"transport_smoke": "empty-snapshot"},
        ),
        expert_base_release_id=expert_release_id,
        task_context_binding=task_context,
        artifact_environment=environment,
        capture_descriptor_ref="capture_descriptor.json",
        checkpoint_ref="checkpoint.json",
        execution_event_journal_ref="events.jsonl",
        idea_archive_ref="idea_archive.json",
        experiment_history_ref="experiment_history.json",
        sanitation_report_ref="sanitation_report.json",
        branch_snapshot_refs=(),
        run_log_refs=(),
        checksums=checksums,
    )
    report = SanitationReport.mint(
        schema=SANITATION_REPORT_SCHEMA,
        capture_manifest_id=content_id(
            "capture-manifest",
            {"bundle_id": bundle.bundle_id},
        ),
        scope_id=scope_contract.scope_id,
        task_family_id=task_context.task_family_id,
        policy_version=settings.sanitation.policy_version,
        policy_fingerprint=tree_or_blob_digest(settings.sanitation.to_json_bytes()),
        scanner_version=SANITATION_SCANNER_VERSION,
        status="admitted",
        findings=(),
        excluded_paths=(),
        taint_sources=(),
        admitted_refs=checksums,
    )
    prior_idea = PriorIdea.mint(
        source_bundle_id=bundle.bundle_id,
        supersedes_projection_id=None,
        source={
            "scope_id": scope_contract.scope_id,
            "run_id": bundle.run_id,
            "campaign_id": bundle.campaign_id,
            "batch_id": "production_smoke_batch",
            "idea_id": "production_smoke_idea",
        },
        proposal=values[0],
        descriptor={
            "approach_family": "representation_validation",
            "expected_effect": "reduce_interface_regressions",
            "intervention_target": "input_projection",
            "mechanism": "validate_semantic_parity_before_training",
        },
        assumptions=(values[1],),
        source_status=PriorIdeaStatus.DEFERRED,
        source_rationale="The synthetic run stopped before executing this idea.",
        source_evidence_refs=(),
        task_context_binding=task_context,
        sanitation_report_id=report.report_id,
    )
    return ProjectionResult(
        source_bundle=bundle,
        sanitation_report=report,
        episodes=(),
        prior_ideas=(prior_idea,),
        derivation_objects=(),
    )


def _write_security_archive(
    path: Path,
    snapshot: SecurityDenylistSnapshot,
    evidence: SecurityDenylistEvidenceBundle,
) -> None:
    members = (
        (_SECURITY_MANIFEST_FILENAME, snapshot.to_json_bytes()),
        (SECURITY_DENYLIST_EVIDENCE_FILENAME, evidence.to_json_bytes()),
    )
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as archive:
        for name, payload in members:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            member.mode = 0o444
            member.mtime = 0
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            archive.addfile(member, io.BytesIO(payload))


def _load_fixture(
    settings: CrossRunSettings,
) -> tuple[Mapping[str, Any], str]:
    path = (
        Path(settings.production_validation.fixture_path) / _FIXTURE_FILENAME
    ).resolve(strict=True)
    payload = path.read_bytes()
    parsed = parse_json_bytes(payload)
    if not isinstance(parsed, Mapping) or set(parsed) != {
        "committed_at",
        "embedding_inputs",
        "scope_contract",
    }:
        raise ProductionSmokeError("production smoke fixture fields are invalid")
    _committed_at(parsed["committed_at"])
    ExpertScopeContract.from_dict(parsed["scope_contract"])
    return parsed, tree_or_blob_digest(payload)


def _validate_stages(stages: tuple[str, ...]) -> tuple[str, ...]:
    if (
        type(stages) is not tuple
        or not stages
        or any(stage not in _STAGE_ORDER for stage in stages)
        or len(stages) != len(set(stages))
    ):
        raise ProductionSmokeError("production smoke stages are invalid")
    ordered = tuple(stage for stage in _STAGE_ORDER if stage in stages)
    if ordered != stages:
        raise ProductionSmokeError("production smoke stages are out of order")
    return ordered


def _read_receipt(
    path: Path,
    *,
    maximum_bytes: int,
    configuration_fingerprint: str,
    fixture_digest: str,
    scope_id: str,
) -> Mapping[str, Any]:
    if not path.exists():
        content = {
            "configuration_fingerprint": configuration_fingerprint,
            "fixture_digest": fixture_digest,
            "scope_id": scope_id,
            "stage_receipts": [],
        }
        return {
            "production_smoke_receipt_id": content_id(
                "production-smoke-receipt", content
            ),
            **content,
        }
    metadata = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_size > maximum_bytes
    ):
        raise ProductionSmokeError("production smoke receipt is unsafe or oversized")
    payload = path.read_bytes()
    parsed = parse_json_bytes(payload)
    _validate_receipt(
        parsed,
        configuration_fingerprint=configuration_fingerprint,
        fixture_digest=fixture_digest,
        scope_id=scope_id,
    )
    if canonical_json_bytes(parsed) != payload:
        raise ProductionSmokeError("production smoke receipt is not canonical")
    return parsed


def _append_stage_receipt(
    receipt: Mapping[str, Any],
    stage_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    content = {
        "configuration_fingerprint": receipt["configuration_fingerprint"],
        "fixture_digest": receipt["fixture_digest"],
        "scope_id": receipt["scope_id"],
        "stage_receipts": [*receipt["stage_receipts"], dict(stage_receipt)],
    }
    return {
        "production_smoke_receipt_id": content_id("production-smoke-receipt", content),
        **content,
    }


def _validate_receipt(
    receipt: object,
    *,
    configuration_fingerprint: str,
    fixture_digest: str,
    scope_id: str,
) -> None:
    expected_fields = {
        "production_smoke_receipt_id",
        "configuration_fingerprint",
        "fixture_digest",
        "scope_id",
        "stage_receipts",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected_fields:
        raise ProductionSmokeError("production smoke receipt fields are invalid")
    if (
        receipt["configuration_fingerprint"] != configuration_fingerprint
        or receipt["fixture_digest"] != fixture_digest
        or receipt["scope_id"] != scope_id
        or not isinstance(receipt["stage_receipts"], list)
    ):
        raise ProductionSmokeError("production smoke receipt authority changed")
    stages = []
    for item in receipt["stage_receipts"]:
        if (
            not isinstance(item, Mapping)
            or set(item)
            != {
                "stage_receipt_id",
                "stage",
                "started_at",
                "completed_at",
                "evidence",
            }
            or not isinstance(item["evidence"], Mapping)
        ):
            raise ProductionSmokeError("production stage receipt is invalid")
        content = {
            "stage": item["stage"],
            "started_at": item["started_at"],
            "completed_at": item["completed_at"],
            "evidence": dict(item["evidence"]),
        }
        if item["stage_receipt_id"] != content_id(
            "production-smoke-stage-receipt", content
        ):
            raise ProductionSmokeError("production stage receipt ID is invalid")
        normalize_utc_timestamp(item["started_at"], "smoke stage started_at")
        normalize_utc_timestamp(item["completed_at"], "smoke stage completed_at")
        stages.append(item["stage"])
    if tuple(stages) != tuple(stage for stage in _STAGE_ORDER if stage in stages):
        raise ProductionSmokeError("production stage receipt order is invalid")
    content = dict(receipt)
    actual_id = content.pop("production_smoke_receipt_id")
    if actual_id != content_id("production-smoke-receipt", content):
        raise ProductionSmokeError("production smoke receipt ID is invalid")


def _write_receipt(path: Path, receipt: Mapping[str, Any], maximum_bytes: int) -> None:
    payload = canonical_json_bytes(receipt)
    if len(payload) > maximum_bytes:
        raise ProductionSmokeError("production smoke receipt exceeds its bound")
    staging = path.parent / _RECEIPT_STAGING_FILENAME
    descriptor = os.open(
        staging,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staging, path)
    with ExitStack() as descriptors:
        directory = os.open(
            path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, directory)
        os.fsync(directory)


def _artifact_kinds() -> tuple[PublicationArtifactKind, ...]:
    return (
        PublicationArtifactKind.EXPERT_BASE_RELEASE,
        PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
        PublicationArtifactKind.SECURITY_DENYLIST,
    )


def _timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _committed_at(value: object) -> str:
    if not isinstance(value, str):
        raise ProductionSmokeError("production committed_at must be text")
    return normalize_utc_timestamp(value, "production committed_at")
