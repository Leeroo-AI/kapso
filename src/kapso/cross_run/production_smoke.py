"""One staged production driver over the real cross-run trust boundaries."""

from __future__ import annotations

import io
import os
import stat
import tarfile
import tempfile
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from kapso.core.config import load_effective_config
from kapso.core.embedding_contracts import EmbeddingSettings, cosine_similarity
from kapso.core.embedding_provider import OpenAIEmbeddingProvider
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    normalize_utc_timestamp,
    parse_json_bytes,
    require_content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.capture.bundle import RunBundleStore
from kapso.cross_run.catalog.service import CrossRunCatalog
from kapso.cross_run.catalog.store import CatalogGenerationManifest
from kapso.cross_run.catalog.projector import ProjectionResult
from kapso.cross_run.contracts import (
    CrossRunTaskBindingSettings,
    EMPTY_EXPERT_TREE_DIGEST,
    ExpertEvaluatorResultRecord,
    ExpertScopeContract,
    ExpertValidationStage,
    PriorIdea,
    PublicationArtifactKind,
    RunBundle,
    SECURITY_DENYLIST_EVIDENCE_FILENAME,
    SECURITY_DENYLIST_POLICY_VERSION,
    SECURITY_DENYLIST_SCHEMA_VERSION,
    SecurityDenylistEvidenceBundle,
    SecurityDenylistSnapshot,
    TransferEpisode,
)
from kapso.cross_run.docker.runtime import DockerImageAuthority, PinnedDockerRuntime
from kapso.cross_run.expert.composition_base import ExpertCompositionBaseClosure
from kapso.cross_run.expert.composition_base_provider import (
    GitHubExpertCompositionBaseProvider,
)
from kapso.cross_run.expert.github_evaluator import GitHubExpertEvaluatorExchange
from kapso.cross_run.expert.task_evaluation_materialization import (
    TaskEvaluationMaterializationLimits,
)
from kapso.cross_run.expert.task_evaluation_provider_filesystem import (
    materialize_verified_byte_tree,
)
from kapso.cross_run.expert.triggers import (
    ExpertTriggerEvidencePacketBuilder,
    ExpertTriggerObservation,
    ExpertTriggerObservationKind,
)
from kapso.cross_run.github.publisher import PublicationEnvelope, ReleaseAssetInput
from kapso.cross_run.github.resolver import security_denylist_tag
from kapso.cross_run.knowledge.package import KnowledgeSnapshotPackage
from kapso.cross_run.knowledge.publisher import KnowledgeSnapshotPublisher
from kapso.cross_run.knowledge.index import SnapshotSearchIndex
from kapso.cross_run.knowledge.retrieval import (
    CrossRunRetriever,
    PriorKnowledgeQuery,
)
from kapso.cross_run.launch.contracts import LaunchTaskContextRequest
from kapso.cross_run.operations import (
    GitHubOperationServices,
    _expert_validation_services,
    _github_services,
    _private_state_root,
    publish_expert_cross_run,
    propose_expert_cross_run,
    resolve_launch_cross_run,
    revoke_expert_cross_run,
    validate_expert_cross_run,
)
from kapso.cross_run.production_capture import (
    ProductionCapture,
    build_production_capture,
)
from kapso.cross_run.record_contracts import (
    BundleProjectionManifest,
    ExecutionRevisionEvent,
    SanitationReport,
)
from kapso.cross_run.record_registry import parse_knowledge_record_payload
from kapso.cross_run.production_task_adapters import (
    bootstrap_production_task_adapters,
    production_capture_evaluation_fingerprint,
)
from kapso.cross_run.security_denylist import (
    AuthenticatedSecurityDenylistAuthority,
    GitHubSecurityDenylistSnapshotProvider,
    SecurityDenylistCheckpointStore,
    SecurityDenylistPublisher,
)
from kapso.cross_run.settings import CrossRunSettings
from kapso.execution.coding_agents.operation_receipt import (
    seal_coding_agent_operation,
)
from kapso.execution.coding_agents.structured_call import (
    CodingAgentCallRequest,
    CodingAgentRunnerSettings,
    CodingAgentWorkspacePolicy,
    SubprocessCodingAgentCallRunner,
    coding_agent_mcp_configuration_fingerprint,
)
from kapso.execution.coding_agents.workspace_delta import (
    inspect_coding_agent_workspace,
)


class ProductionSmokeError(ValueError):
    """The selected production stage or its durable receipt is invalid."""


_STAGE_ORDER = (
    "preflight",
    "bootstrap-authorities",
    "github-read",
    "embeddings",
    "docker-authority",
    "task-adapter-bootstrap",
    "expert-proposal",
    "expert-validation-enrollment",
    "expert-bootstrap-validation",
    "expert-bootstrap-publication",
    "knowledge-publication",
    "coding-agent-ideation",
    "expert-successor-proposal",
    "expert-successor-validation",
    "expert-successor-publication",
    "successor-launch",
    "concurrent-publication",
    "clean-machine-launch",
    "live-restart",
    "revocation",
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
            config_path=config_path,
            mode=mode,
            settings=settings,
            smoke_root=smoke_root,
            fixture=fixture,
            scope_contract=scope_contract,
            prior_evidence={
                item["stage"]: item["evidence"] for item in receipt["stage_receipts"]
            },
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
    config_path: str,
    mode: str,
    settings: CrossRunSettings,
    smoke_root: Path,
    fixture: Mapping[str, Any],
    scope_contract: ExpertScopeContract,
    prior_evidence: Mapping[str, Mapping[str, Any]],
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
            prior_evidence,
        )
    if stage == "coding-agent-ideation":
        return _coding_agent_ideation_smoke(
            settings,
            smoke_root,
            fixture,
            scope_contract,
            prior_evidence,
        )
    if stage == "task-adapter-bootstrap":
        return _task_adapter_bootstrap_smoke(
            settings,
            smoke_root,
            scope_contract,
        )
    if stage == "expert-proposal":
        return _expert_proposal_smoke(
            config_path,
            mode,
            settings,
            smoke_root,
            scope_contract,
        )
    if stage == "expert-validation-enrollment":
        return _expert_validation_enrollment_smoke(
            config_path,
            mode,
            smoke_root,
            prior_evidence,
        )
    if stage == "expert-bootstrap-validation":
        return _expert_validation_smoke(
            config_path,
            mode,
            settings,
            smoke_root,
            prior_evidence,
            evidence_stage="expert-validation-enrollment",
        )
    if stage == "expert-bootstrap-publication":
        return _expert_publication_smoke(
            config_path,
            mode,
            smoke_root,
            fixture,
            prior_evidence,
            validation_stage="expert-bootstrap-validation",
            proposal_stage="expert-proposal",
        )
    if stage == "expert-successor-proposal":
        return _expert_successor_proposal_smoke(
            config_path,
            mode,
            settings,
            smoke_root,
            scope_contract,
            prior_evidence,
        )
    if stage == "expert-successor-validation":
        return _expert_validation_smoke(
            config_path,
            mode,
            settings,
            smoke_root,
            prior_evidence,
            evidence_stage="expert-successor-proposal",
        )
    if stage == "expert-successor-publication":
        return _expert_publication_smoke(
            config_path,
            mode,
            smoke_root,
            fixture,
            prior_evidence,
            validation_stage="expert-successor-validation",
            proposal_stage="expert-successor-proposal",
        )
    if stage == "successor-launch":
        return _successor_launch_smoke(
            config_path,
            mode,
            settings,
            smoke_root,
            scope_contract,
            prior_evidence,
            clean_machine=False,
        )
    if stage == "concurrent-publication":
        return _concurrent_publication_smoke(prior_evidence)
    if stage == "clean-machine-launch":
        return _successor_launch_smoke(
            config_path,
            mode,
            settings,
            smoke_root,
            scope_contract,
            prior_evidence,
            clean_machine=True,
        )
    if stage == "live-restart":
        return _live_restart_smoke(prior_evidence)
    if stage == "revocation":
        return _revocation_smoke(
            config_path,
            mode,
            smoke_root,
            fixture,
            prior_evidence,
        )
    raise ProductionSmokeError("unknown production smoke stage")


def _task_adapter_bootstrap_smoke(
    settings: CrossRunSettings,
    smoke_root: Path,
    scope_contract: ExpertScopeContract,
) -> Mapping[str, Any]:
    """Activate public deterministic adapters for the transport-only cascade."""

    image = settings.launch.coding_agent_image
    if image is None:
        raise ProductionSmokeError(
            "task-adapter bootstrap requires a pinned coding-agent image"
        )
    runtime_root = _private_state_root(smoke_root / "task-adapter-docker")
    runtime = PinnedDockerRuntime.create(
        trusted_root=runtime_root,
        settings=settings.docker,
    )
    authority = DockerImageAuthority.mint(
        image_reference=image.image_reference,
        image_config_digest=image.image_config_digest,
        operating_system=image.operating_system,
        architecture=image.architecture,
        architecture_variant=image.architecture_variant,
    )
    inspection = runtime.inspect_exact_image(authority)
    return bootstrap_production_task_adapters(
        settings=settings,
        state_root=smoke_root,
        scope_contract=scope_contract,
        image_inspection=inspection,
    )


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
        "expert_evaluator_authority": _expert_evaluator_authority(settings),
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


def _expert_evaluator_authority(settings: CrossRunSettings) -> Mapping[str, Any]:
    validation = settings.expert.validation
    issuers = {
        evaluator.evaluator_id: validation.evaluator_trust_root_id(
            evaluator.evaluator_id
        )
        for evaluator in validation.policy.evaluators
        if evaluator.stage
        not in {
            ExpertValidationStage.SEALED_CANARY,
            ExpertValidationStage.SOURCE_RUN_REPLAY,
            ExpertValidationStage.RELEASE_MATRIX,
        }
    }
    missing = tuple(
        sorted(issuer_id for issuer_id, root_id in issuers.items() if root_id is None)
    )
    return {
        "configured": not missing,
        "issuer_trust_roots": issuers,
        "missing_issuer_ids": missing,
        "sealed_canary_trust_root": validation.policy.sealed_canary_trust_root,
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
    if first_identities != second_identities:
        raise ProductionSmokeError("OpenAI embedding rebuild changed input identities")
    cosine_distances = tuple(
        max(0.0, 1.0 - cosine_similarity(first_record, second_record))
        for first_record, second_record in zip(first.records, second.records)
    )
    maximum_cosine_distance = max(cosine_distances)
    if (
        maximum_cosine_distance
        > settings.production_validation.embedding_cosine_distance_tolerance
    ):
        raise ProductionSmokeError(
            "OpenAI embedding rebuild exceeded the configured cosine-distance "
            "tolerance"
        )
    first_vectors = tuple(record.vector for record in first.records)
    second_vectors = tuple(record.vector for record in second.records)
    return {
        "provider": provider_settings.provider,
        "model": provider_settings.model,
        "dimensions": provider_settings.dimensions,
        "embedding_space_id": provider_settings.embedding_space_id.value,
        "input_hashes": first_identities,
        "first_vector_digest": tree_or_blob_digest(canonical_json_bytes(first_vectors)),
        "second_vector_digest": tree_or_blob_digest(
            canonical_json_bytes(second_vectors)
        ),
        "maximum_cosine_distance": maximum_cosine_distance,
        "cosine_distance_tolerance": (
            settings.production_validation.embedding_cosine_distance_tolerance
        ),
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
    prior_evidence: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Publish one admitted synthetic bundle as the first non-empty snapshot."""

    bootstrap_publication = prior_evidence.get("expert-bootstrap-publication")
    if not isinstance(bootstrap_publication, Mapping) or not isinstance(
        bootstrap_publication.get("release_id"), str
    ):
        raise ProductionSmokeError(
            "knowledge publication requires the authenticated bootstrap expert release"
        )
    expert_release_id = bootstrap_publication["release_id"]

    github = _github_services(settings, smoke_root)
    current = github.resolver.resolve_current(
        scope_contract.scope_id,
        PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
    )
    current_materialized = github.materializer.materialize(current)
    current_package = KnowledgeSnapshotPackage.open(current_materialized.content)
    task_adapter_manifest_id, verification_receipt_id = _production_task_adapter_pin(
        prior_evidence,
        task_adapter_id="posttrain",
    )
    capture = _synthetic_capture_for_snapshot(
        settings,
        fixture,
        scope_contract,
        current_package,
        expert_release_id,
        task_adapter_manifest_id,
        verification_receipt_id,
    )
    RunBundleStore.initialize(
        smoke_root / settings.capture.state_path,
        settings.capture,
        settings.sanitation,
    ).import_exact(capture.stored_bundle)
    projection = capture.projection
    if (
        projection.source_bundle.bundle_id
        in current_package.manifest.included_bundle_ids
    ):
        if (
            current_package.manifest.catalog_generation <= 0
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
            "catalog_generation_id": (current_package.prepared.catalog_generation_id),
            "bundle_id": projection.source_bundle.bundle_id,
            "prior_idea_id": projection.prior_ideas[0].prior_idea_id,
            "publication_id": record.publication_id,
            "commit_sha": current.pointer_commit_sha,
            "release_tag": record.tag,
            "recovered": True,
        }

    catalog = CrossRunCatalog(
        smoke_root / "catalog",
        scope_contract,
        settings.catalog,
    )
    generation = catalog.store.read_current()
    current_fact_ids = set(current_package.prepared.catalog_generation.fact_object_ids)
    if not current_fact_ids.issubset(generation.fact_object_ids):
        if generation.generation_number != 0:
            raise ProductionSmokeError(
                "local production catalog differs from knowledge CURRENT"
            )
        generation = _seed_catalog_from_snapshot(
            catalog,
            current_package,
        )
    if projection.source_bundle.bundle_id not in generation.fact_object_ids:
        generation = catalog.publish_projection(generation, projection).generation

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


def _coding_agent_ideation_smoke(
    settings: CrossRunSettings,
    smoke_root: Path,
    fixture: Mapping[str, Any],
    scope_contract: ExpertScopeContract,
    prior_evidence: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Prove a real Codex call reads one record from the live S1 snapshot."""

    if not settings.production_validation.coding_agent_smoke:
        raise ProductionSmokeError(
            "coding-agent ideation smoke is disabled in configuration"
        )
    bootstrap_publication = prior_evidence.get("expert-bootstrap-publication")
    if not isinstance(bootstrap_publication, Mapping) or not isinstance(
        bootstrap_publication.get("release_id"), str
    ):
        raise ProductionSmokeError(
            "coding-agent ideation requires the authenticated bootstrap expert release"
        )
    github = _github_services(settings, smoke_root / "coding-agent-github-read")
    resolved = github.resolver.resolve_current(
        scope_contract.scope_id,
        PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
    )
    materialized = github.materializer.materialize(resolved)
    package = KnowledgeSnapshotPackage.open(materialized.content)
    task_adapter_manifest_id, verification_receipt_id = _production_task_adapter_pin(
        prior_evidence,
        task_adapter_id="posttrain",
    )
    projection = _synthetic_capture_for_snapshot(
        settings,
        fixture,
        scope_contract,
        package,
        bootstrap_publication["release_id"],
        task_adapter_manifest_id,
        verification_receipt_id,
    ).projection
    expected_prior_idea = projection.prior_ideas[0]
    index_files = {
        path: payload
        for path, payload in package.files.items()
        if PurePosixPath(path).parts[0] == "index"
    }
    if not index_files:
        raise ProductionSmokeError("live knowledge snapshot lacks a search index")
    retriever = CrossRunRetriever(
        package,
        SnapshotSearchIndex.open(package.prepared, index_files),
        settings.knowledge.retrieval,
    )
    query = PriorKnowledgeQuery(
        task_context_binding=projection.source_bundle.task_context_binding,
        problem=expected_prior_idea.proposal,
        current_gaps=expected_prior_idea.assumptions,
        directive=(
            "Retrieve a deferred representation-validation intervention that can "
            "ground a novel next experiment."
        ),
    )
    embedding_telemetry = None
    if retriever.semantic_embedding_space_ids:
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
        embedded = OpenAIEmbeddingProvider(provider_settings).embed(
            (query.lexical_text,)
        )
        if len(embedded.records) != 1:
            raise ProductionSmokeError(
                "production ideation query embedding returned an invalid batch"
            )
        query = PriorKnowledgeQuery(
            task_context_binding=query.task_context_binding,
            problem=query.problem,
            current_gaps=query.current_gaps,
            directive=query.directive,
            query_embedding=embedded.records[0],
        )
        embedding_telemetry = embedded.telemetry
    retrieval = retriever.retrieve(query)
    if not retrieval.selections:
        raise ProductionSmokeError(
            "production ideation retrieval selected no prior knowledge"
        )
    target_record_id = retrieval.selections[0].record_id

    workspace = _private_state_root(smoke_root / "coding-agent-ideation-workspace")
    artifact_root = _private_state_root(smoke_root / "coding-agent-ideation-artifacts")
    agent = settings.expert.generalizer
    response_schema = _production_ideation_response_schema()
    prompt = (
        "Without changing the workspace, first call "
        "prior_knowledge.list_prior_knowledge. Confirm that the trusted parent's "
        f"primary selected record {target_record_id} is listed, then call "
        "prior_knowledge.get_prior_knowledge_record for that exact "
        "record. Use the complete record to propose one concise, novel next "
        "experiment that preserves its useful mechanism while changing one "
        "scientifically meaningful dimension. Return only the required JSON, "
        "including the exact record ID you read."
    )
    operation_seed = canonical_json_bytes(
        {
            "agent": agent.to_dict(),
            "configuration_fingerprint": settings.configuration_fingerprint,
            "materialization_digest": retrieval.access_materialization.materialization_digest,
            "mcp_configuration_fingerprint": (
                coding_agent_mcp_configuration_fingerprint(
                    retrieval.access_materialization
                )
            ),
            "prompt_digest": tree_or_blob_digest(prompt.encode("utf-8")),
            "response_schema": response_schema,
            "snapshot_id": package.manifest.snapshot_id,
        }
    )
    operation_id = "agent_call_" + tree_or_blob_digest(operation_seed)[7:39]
    request = CodingAgentCallRequest(
        operation_id=operation_id,
        role="production_smoke_ideation",
        cli=agent.cli,
        model=agent.model,
        effort=agent.effort,
        prompt=prompt,
        workspace=str(workspace),
        workspace_policy=CodingAgentWorkspacePolicy.read_only(),
        timeout_seconds=agent.timeout_seconds,
        allowed_tools=agent.allowed_tools,
        prior_knowledge=retrieval.access_materialization,
    )
    runner = SubprocessCodingAgentCallRunner(
        CodingAgentRunnerSettings(
            artifact_root=str(artifact_root),
            termination_grace_seconds=settings.expert.termination_grace_seconds,
            sensitive_file_glob_scan_max_depth=(
                settings.expert.sensitive_file_glob_scan_max_depth
            ),
        )
    )
    result = runner.run(request, response_schema)
    output = _validate_production_ideation_output(
        result.output,
        target_record_id,
    )
    sealed = seal_coding_agent_operation(
        request=request,
        response_schema=response_schema,
        principal_id=settings.expert.generalizer_id,
        agent=agent,
        sensitive_file_glob_scan_max_depth=(
            settings.expert.sensitive_file_glob_scan_max_depth
        ),
        result=result,
    )
    audit_events = tuple(
        parse_json_bytes(line.encode("utf-8"))
        for line in sealed.artifact_bytes["mcp_audit.jsonl"]
        .decode("utf-8")
        .splitlines()
    )
    if (
        len(audit_events) < 2
        or audit_events[0]["tool_name"] != "list_prior_knowledge"
        or not any(
            event["tool_name"] == "get_prior_knowledge_record"
            and event["arguments"] == {"record_id": target_record_id}
            for event in audit_events
        )
    ):
        raise ProductionSmokeError(
            "production ideation did not list and read the selected prior record"
        )
    return {
        "snapshot_id": package.manifest.snapshot_id,
        "prior_record_id": output["prior_record_id"],
        "selection_count": len(retrieval.selections),
        "materialization_digest": retrieval.access_materialization.materialization_digest,
        "operation_id": request.operation_id,
        "operation_receipt_id": sealed.receipt.operation_receipt_id,
        "final_output_digest": result.final_output_digest,
        "mcp_audit_digest": result.mcp_audit_digest,
        "mcp_audit_event_count": result.mcp_audit_event_count,
        "embedding_call_count": (
            None if embedding_telemetry is None else embedding_telemetry.call_count
        ),
    }


def _production_ideation_response_schema() -> Mapping[str, Any]:
    return {
        "type": "object",
        "properties": {
            "idea": {"type": "string", "minLength": 1},
            "mechanism": {"type": "string", "minLength": 1},
            "prior_record_id": {"type": "string", "minLength": 1},
        },
        "required": ["idea", "mechanism", "prior_record_id"],
        "additionalProperties": False,
    }


def _validate_production_ideation_output(
    output: str,
    expected_prior_record_id: str,
) -> Mapping[str, Any]:
    parsed = parse_json_bytes(output.encode("utf-8"))
    if (
        not isinstance(parsed, Mapping)
        or set(parsed) != {"idea", "mechanism", "prior_record_id"}
        or any(
            not isinstance(parsed[field], str) or not parsed[field].strip()
            for field in ("idea", "mechanism", "prior_record_id")
        )
        or parsed["prior_record_id"] != expected_prior_record_id
    ):
        raise ProductionSmokeError(
            "production ideation output does not cite the selected prior record"
        )
    return parsed


def _expert_proposal_smoke(
    config_path: str,
    mode: str,
    settings: CrossRunSettings,
    smoke_root: Path,
    scope_contract: ExpertScopeContract,
) -> Mapping[str, Any]:
    """Create, but never approve, one real empty-scope architecture candidate."""

    github = _github_services(settings, smoke_root / "expert-proposal-github-read")
    expert_state = github.resolver.read_current_pointer_state(
        scope_contract.scope_id,
        PublicationArtifactKind.EXPERT_BASE_RELEASE,
        allow_missing=True,
    )
    if expert_state.pointer is not None:
        resolved = github.resolver.resolve_current(
            scope_contract.scope_id,
            PublicationArtifactKind.EXPERT_BASE_RELEASE,
        )
        return {
            "existing_release_id": resolved.pointer.publication_record.artifact_id,
            "existing_release_commit_sha": resolved.pointer_commit_sha,
            "proposal_skipped": True,
        }
    resolved_knowledge = github.resolver.resolve_current(
        scope_contract.scope_id,
        PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
    )
    materialized = github.materializer.materialize(resolved_knowledge)
    package = KnowledgeSnapshotPackage.open(materialized.content)
    if package.prepared.scope_contract != scope_contract:
        raise ProductionSmokeError(
            "expert proposal knowledge snapshot has another scope contract"
        )
    packet = ExpertTriggerEvidencePacketBuilder(settings.expert.triggers).build(
        knowledge_snapshot=package,
        scope_contract=scope_contract,
        source_base_scope_contract=None,
        source_base_release=None,
        source_base_tree_receipt=None,
        source_base_tree_hash=EMPTY_EXPERT_TREE_DIGEST,
        source_base_repository_map=None,
        source_base_module_contracts=(),
        active_task_bindings=_scope_task_bindings(scope_contract),
    )
    with tempfile.TemporaryDirectory(
        prefix="expert-proposal-request-",
        dir=smoke_root,
    ) as temporary:
        request_path = Path(temporary) / "request.json"
        request_path.write_bytes(
            canonical_json_bytes({"evidence_packet": packet.to_dict()})
        )
        result = propose_expert_cross_run(
            config_path=config_path,
            mode=mode,
            request_path=request_path,
            state_root=smoke_root,
        )
    if (
        result.get("scope_id") != scope_contract.scope_id
        or result.get("source_base_release_id") is not None
        or result.get("change_kind") != "repository_architecture"
    ):
        raise ProductionSmokeError(
            "expert bootstrap proposer returned another candidate class"
        )
    return {
        "candidate_id": result["candidate_id"],
        "candidate_tree_hash": result["candidate_tree_hash"],
        "change_kind": result["change_kind"],
        "knowledge_snapshot_id": package.manifest.snapshot_id,
        "proposal_operation_id": result["proposal_operation_id"],
        "source_base_release_id": result["source_base_release_id"],
        "task_binding_count": len(packet.active_task_bindings),
        "trigger_decision_id": result["trigger_decision_id"],
        "proposal_skipped": False,
    }


def _scope_task_bindings(
    scope_contract: ExpertScopeContract,
) -> tuple[CrossRunTaskBindingSettings, ...]:
    return tuple(
        CrossRunTaskBindingSettings(
            scope_id=scope_contract.scope_id,
            task_family_id=binding.task_family_id,
            task_adapter_id=task_adapter_id,
        )
        for binding in scope_contract.task_adapter_contract
        for task_adapter_id in binding.task_adapter_ids
    )


def _expert_validation_enrollment_smoke(
    config_path: str,
    mode: str,
    smoke_root: Path,
    prior_evidence: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Enroll the proposed candidate and stop at the external evaluator boundary."""

    proposal = prior_evidence.get("expert-proposal")
    if not isinstance(proposal, Mapping) or not isinstance(
        proposal.get("proposal_skipped"), bool
    ):
        raise ProductionSmokeError(
            "expert validation enrollment requires expert proposal evidence"
        )
    if proposal["proposal_skipped"]:
        existing_release_id = proposal.get("existing_release_id")
        if not isinstance(existing_release_id, str) or not existing_release_id:
            raise ProductionSmokeError(
                "skipped expert proposal lacks its authenticated current release"
            )
        return {
            "existing_release_id": existing_release_id,
            "validation_skipped": True,
        }

    candidate_id = proposal.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ProductionSmokeError("expert proposal evidence lacks its candidate")
    with tempfile.TemporaryDirectory(
        prefix="expert-validation-enrollment-request-",
        dir=smoke_root,
    ) as temporary:
        request_path = Path(temporary) / "request.json"
        request_path.write_bytes(
            canonical_json_bytes(
                {
                    "candidate_id": candidate_id,
                    "expected_transition_id": None,
                    "evaluator_result": None,
                }
            )
        )
        result = validate_expert_cross_run(
            config_path=config_path,
            mode=mode,
            request_path=request_path,
            state_root=smoke_root,
        )
    if (
        result.get("operation") != "validate-expert"
        or result.get("candidate_id") != candidate_id
        or result.get("next_stage") != ExpertValidationStage.CONTRACT_SCHEMA.value
    ):
        raise ProductionSmokeError(
            "expert validation enrollment did not reach the evaluator boundary"
        )
    return {
        "candidate_id": candidate_id,
        "validation_attempt_id": result["validation_attempt_id"],
        "transition_id": result["transition_id"],
        "validation_state_id": result["validation_state_id"],
        "next_stage": result["next_stage"],
        "validation_skipped": False,
    }


def _expert_validation_smoke(
    config_path: str,
    mode: str,
    settings: CrossRunSettings,
    smoke_root: Path,
    prior_evidence: Mapping[str, Mapping[str, Any]],
    *,
    evidence_stage: str,
) -> Mapping[str, Any]:
    """Advance autonomous and externally signed validation stages."""

    evidence = prior_evidence.get(evidence_stage)
    if not isinstance(evidence, Mapping):
        raise ProductionSmokeError(
            f"expert validation requires {evidence_stage} evidence"
        )
    existing_release_id = evidence.get("existing_release_id")
    if existing_release_id is not None:
        if not isinstance(existing_release_id, str) or not existing_release_id:
            raise ProductionSmokeError(
                "existing expert validation release identity is invalid"
            )
        return {
            "existing_release_id": existing_release_id,
            "validation_skipped": True,
        }
    candidate_id = evidence.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ProductionSmokeError("expert validation evidence lacks its candidate")
    github = _github_services(settings, smoke_root)
    services = _expert_validation_services(settings, smoke_root, github)
    snapshot = services.validation_store.snapshot(candidate_id)
    if snapshot is None:
        result = _call_expert_validation(
            config_path=config_path,
            mode=mode,
            smoke_root=smoke_root,
            candidate_id=candidate_id,
            expected_transition_id=None,
        )
        snapshot = services.validation_store.snapshot(candidate_id)
        if snapshot is None or result.get("candidate_id") != candidate_id:
            raise ProductionSmokeError(
                "expert validation enrollment did not persist its candidate"
            )
    autonomous_stages = {
        ExpertValidationStage.AUTOMATED_REVIEW,
        ExpertValidationStage.SOURCE_RUN_REPLAY,
        ExpertValidationStage.RELEASE_MATRIX,
        ExpertValidationStage.PUBLICATION_ELIGIBILITY,
    }
    while snapshot.state.promotion_state.value == "validating":
        next_stage = snapshot.state.next_stage
        if next_stage is None:
            raise ProductionSmokeError(
                "validating expert candidate has no next validation stage"
            )
        if next_stage not in autonomous_stages:
            evaluator_ids = tuple(
                evaluator.evaluator_id
                for evaluator in settings.expert.validation.policy.evaluators
                if evaluator.stage is next_stage
            )
            missing_roots = tuple(
                evaluator_id
                for evaluator_id in evaluator_ids
                if settings.expert.validation.evaluator_trust_root_id(evaluator_id)
                is None
            )
            if missing_roots:
                raise ProductionSmokeError(
                    "expert validation requires an externally signed evaluator "
                    f"result for {next_stage.value}; evaluator_ids={evaluator_ids}; "
                    f"missing_trust_roots={missing_roots}"
                )
        predecessor_transition_id = snapshot.transition.transition_id
        evaluator_result = None
        if next_stage not in autonomous_stages:
            attempt = snapshot.latest_attempt
            if attempt is None:
                raise ProductionSmokeError(
                    "external expert validation has no active attempt"
                )
            stored_candidate = services.candidate_store.read(candidate_id)
            scope_id = (
                stored_candidate.closure.validation_context.scope_contract.scope_id
            )
            evaluator_result = GitHubExpertEvaluatorExchange(
                client=github.resolver.client,
                github_settings=settings.github,
                validation_settings=settings.expert.validation,
                sanitation_settings=settings.sanitation,
                security_repository=(
                    settings.scopes.resolve(scope_id).security_repository
                ),
            ).evaluate(
                stored_candidate=stored_candidate,
                attempt=attempt,
                stage=next_stage,
                expected_transition_id=predecessor_transition_id,
            )
        _call_expert_validation(
            config_path=config_path,
            mode=mode,
            smoke_root=smoke_root,
            candidate_id=candidate_id,
            expected_transition_id=predecessor_transition_id,
            evaluator_result=evaluator_result,
        )
        snapshot = services.validation_store.snapshot(candidate_id)
        if (
            snapshot is None
            or snapshot.transition.transition_id == predecessor_transition_id
        ):
            raise ProductionSmokeError(
                "autonomous expert validation stage did not advance"
            )
    if snapshot.state.promotion_state.value != "approved":
        raise ProductionSmokeError(
            "expert validation reached a non-approved terminal state"
        )
    return {
        "candidate_id": candidate_id,
        "validation_attempt_id": snapshot.state.validation_attempt_id,
        "transition_id": snapshot.transition.transition_id,
        "validation_state_id": snapshot.state.validation_state_id,
        "accepted_stage_result_ids": tuple(
            reference.stage_result_record_id
            for reference in snapshot.state.accepted_stage_results
        ),
        "promotion_state": snapshot.state.promotion_state.value,
        "validation_skipped": False,
    }


def _call_expert_validation(
    *,
    config_path: str,
    mode: str,
    smoke_root: Path,
    candidate_id: str,
    expected_transition_id: str | None,
    evaluator_result: ExpertEvaluatorResultRecord | None = None,
) -> Mapping[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="expert-validation-request-",
        dir=smoke_root,
    ) as temporary:
        request_path = Path(temporary) / "request.json"
        request_path.write_bytes(
            canonical_json_bytes(
                {
                    "candidate_id": candidate_id,
                    "expected_transition_id": expected_transition_id,
                    "evaluator_result": (
                        None if evaluator_result is None else evaluator_result.to_dict()
                    ),
                }
            )
        )
        return validate_expert_cross_run(
            config_path=config_path,
            mode=mode,
            request_path=request_path,
            state_root=smoke_root,
        )


def _expert_publication_smoke(
    config_path: str,
    mode: str,
    smoke_root: Path,
    fixture: Mapping[str, Any],
    prior_evidence: Mapping[str, Mapping[str, Any]],
    *,
    validation_stage: str,
    proposal_stage: str,
) -> Mapping[str, Any]:
    """Publish one externally approved candidate through the existing service."""

    evidence = prior_evidence.get(validation_stage)
    if not isinstance(evidence, Mapping):
        evidence = prior_evidence.get(proposal_stage)
    if not isinstance(evidence, Mapping):
        raise ProductionSmokeError(
            f"expert publication requires {proposal_stage} evidence"
        )
    existing_release_id = evidence.get("existing_release_id")
    if existing_release_id is not None:
        if not isinstance(existing_release_id, str) or not existing_release_id:
            raise ProductionSmokeError("existing expert release identity is invalid")
        return {
            "release_id": existing_release_id,
            "publication_skipped": True,
        }
    candidate_id = evidence.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ProductionSmokeError("expert publication evidence lacks its candidate")
    with tempfile.TemporaryDirectory(
        prefix="expert-publication-request-",
        dir=smoke_root,
    ) as temporary:
        request_path = Path(temporary) / "request.json"
        request_path.write_bytes(
            canonical_json_bytes(
                {
                    "candidate_id": candidate_id,
                    "committed_at": _committed_at(fixture["committed_at"]),
                }
            )
        )
        result = publish_expert_cross_run(
            config_path=config_path,
            mode=mode,
            request_path=request_path,
            state_root=smoke_root,
        )
    if result.get("candidate_id") != candidate_id or not isinstance(
        result.get("release_id"), str
    ):
        raise ProductionSmokeError(
            "expert publication returned another candidate or release"
        )
    return {
        "candidate_id": candidate_id,
        "release_id": result["release_id"],
        "activation_receipt_id": result["activation_receipt_id"],
        "publication_id": result["publication_id"],
        "commit_sha": result["commit_sha"],
        "release_tag": result["release_tag"],
        "asset_digests": result["asset_digests"],
        "replayed": result["replayed"],
        "publication_skipped": False,
    }


def _expert_successor_proposal_smoke(
    config_path: str,
    mode: str,
    settings: CrossRunSettings,
    smoke_root: Path,
    scope_contract: ExpertScopeContract,
    prior_evidence: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Propose E1 from the authenticated current expert and knowledge releases."""

    publication = prior_evidence.get("expert-bootstrap-publication")
    if not isinstance(publication, Mapping) or not isinstance(
        publication.get("release_id"), str
    ):
        raise ProductionSmokeError(
            "expert successor proposal requires bootstrap publication evidence"
        )
    github = _github_services(settings, smoke_root / "expert-successor-github-read")
    resolved_knowledge = github.resolver.resolve_current(
        scope_contract.scope_id,
        PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
    )
    knowledge = KnowledgeSnapshotPackage.open(
        github.materializer.materialize(resolved_knowledge).content
    )
    if len(knowledge.prepared.admitted_episode_ids) != 1:
        raise ProductionSmokeError(
            "expert successor proposal requires exactly one admitted production "
            "TransferEpisode before trigger inspection"
        )
    base_provider = GitHubExpertCompositionBaseProvider(
        github.resolver,
        github.materializer,
        settings.expert,
    )
    current_base = base_provider.resolve_current(scope_contract)
    base = current_base.closure
    if base.release_manifest.release_id != publication["release_id"]:
        raise ProductionSmokeError(
            "expert successor source differs from bootstrap publication"
        )
    trigger_observation = _inspect_expert_successor_trigger(
        settings=settings,
        smoke_root=smoke_root,
        knowledge=knowledge,
        base_provider=base_provider,
        base=base,
    )
    packet = ExpertTriggerEvidencePacketBuilder(settings.expert.triggers).build(
        knowledge_snapshot=knowledge,
        scope_contract=scope_contract,
        source_base_scope_contract=base.scope_contract,
        source_base_release=base.release_manifest,
        source_base_tree_receipt=base.source_base_tree_receipt,
        source_base_tree_hash=base.release_manifest.candidate_tree_hash,
        source_base_repository_map=base.repository_map,
        source_base_module_contracts=base.module_contracts,
        active_task_bindings=_scope_task_bindings(scope_contract),
        trigger_observations=(trigger_observation,),
    )
    with tempfile.TemporaryDirectory(
        prefix="expert-successor-proposal-request-",
        dir=smoke_root,
    ) as temporary:
        request_path = Path(temporary) / "request.json"
        request_path.write_bytes(
            canonical_json_bytes({"evidence_packet": packet.to_dict()})
        )
        result = propose_expert_cross_run(
            config_path=config_path,
            mode=mode,
            request_path=request_path,
            state_root=smoke_root,
        )
    if result.get("source_base_release_id") != publication["release_id"]:
        raise ProductionSmokeError(
            "expert successor proposal returned another source release"
        )
    return {
        "candidate_id": result["candidate_id"],
        "candidate_tree_hash": result["candidate_tree_hash"],
        "change_kind": result["change_kind"],
        "knowledge_snapshot_id": knowledge.manifest.snapshot_id,
        "proposal_operation_id": result["proposal_operation_id"],
        "source_base_release_id": result["source_base_release_id"],
        "source_episode_id": trigger_observation.exact_evidence_ids[0],
        "trigger_inspection_operation_id": (
            trigger_observation.inspection_operation.operation_receipt_id
        ),
        "trigger_observation_id": trigger_observation.observation_id,
        "trigger_decision_id": result["trigger_decision_id"],
    }


def _inspect_expert_successor_trigger(
    *,
    settings: CrossRunSettings,
    smoke_root: Path,
    knowledge: KnowledgeSnapshotPackage,
    base_provider: GitHubExpertCompositionBaseProvider,
    base: ExpertCompositionBaseClosure,
) -> ExpertTriggerObservation:
    """Run one real read-only inspection over E0 and an admitted S1 episode."""

    episode_id = knowledge.prepared.admitted_episode_ids[0]
    episode_envelope = knowledge.record_by_id(episode_id)
    episode = parse_knowledge_record_payload(
        episode_envelope["record_kind"],
        episode_envelope["payload"],
    )
    if type(episode) is not TransferEpisode:
        raise ProductionSmokeError(
            "expert trigger inspection episode parsed incorrectly"
        )
    policy = settings.expert.validation.policy
    source_base = base_provider.materialize_exact(
        base.release_manifest,
        base.source_base_tree_receipt,
        TaskEvaluationMaterializationLimits(
            maximum_entries=policy.task_evaluation_materialization_entry_limit,
            maximum_bytes=policy.task_evaluation_materialization_byte_limit,
            timeout_seconds=policy.task_evaluation_materialization_timeout_seconds,
        ),
    )
    modules_by_contract = {
        module.module_contract_id: module for module in base.module_contracts
    }
    selectable = tuple(
        sorted(
            (
                node.capability_id,
                modules_by_contract[node.module_contract_ref],
                entrypoint,
            )
            for node in base.repository_map.capability_nodes
            if node.module_contract_ref in modules_by_contract
            for entrypoint in modules_by_contract[
                node.module_contract_ref
            ].entrypoint_refs
            if entrypoint in source_base.source_contents
        )
    )
    if not selectable:
        raise ProductionSmokeError(
            "current expert release has no inspectable capability entrypoint"
        )
    capability_id, module, affected_path = selectable[0]
    trigger_settings = settings.expert.triggers
    trigger_configuration_fingerprint = tree_or_blob_digest(
        canonical_json_bytes(trigger_settings.to_dict())
    )
    task_context_binding_id = episode.task_context_binding.task_context_binding_id
    fixed_payload = {
        "affected_capability_ids": [capability_id],
        "affected_paths": [affected_path],
        "configuration_fingerprint": trigger_configuration_fingerprint,
        "difficulty_evidence_signatures": {},
        "difficulty_signature": None,
        "exact_evidence_ids": [episode.episode_id],
        "independent_lineage_ids": [],
        "inspection_policy_version": trigger_settings.inspection_policy_version,
        "kind": ExpertTriggerObservationKind.MECHANICALLY_GENERAL_FIX.value,
        "occurrence_count": 1,
        "source_base_tree_hash": base.release_manifest.candidate_tree_hash,
        "task_context_binding_ids": [task_context_binding_id],
    }
    response_schema = _expert_trigger_inspection_response_schema(fixed_payload)
    prompt = (
        "Inspect the named expert capability entrypoint in the read-only workspace. "
        "Use the complete admitted transfer episode and module contract below to "
        "identify one concrete, reusable refinement that addresses the observed "
        "technical difficulty without encoding a task-specific decision tree. "
        "The refinement must preserve the current repository topology and be "
        "implementable within the named capability and path. Return only the JSON "
        "required by the response schema; describe the causal refinement precisely.\n\n"
        "TRANSFER_EPISODE:\n"
        + canonical_json_bytes(episode.to_dict()).decode("utf-8")
        + "\n\nMODULE_CONTRACT:\n"
        + canonical_json_bytes(module.to_dict()).decode("utf-8")
        + "\n\nFIXED_OBSERVATION_FIELDS:\n"
        + canonical_json_bytes(fixed_payload).decode("utf-8")
    )
    workspace_root = _private_state_root(
        smoke_root / "expert-trigger-inspection-workspaces"
    )
    workspace = workspace_root / base.release_manifest.candidate_tree_hash[7:]
    descriptors = (
        base.source_base_tree_receipt.source_extraction_receipt.source_tree_files
    )
    if not workspace.exists():
        materialize_verified_byte_tree(
            trusted_root=workspace_root,
            destination_root=workspace,
            descriptors=descriptors,
            source_contents=source_base.source_contents,
        )
    observed_workspace = inspect_coding_agent_workspace(
        workspace,
        maximum_entries=settings.expert.candidate_entry_limit,
        maximum_bytes=settings.expert.candidate_byte_limit,
    )
    if observed_workspace.tree_hash != base.release_manifest.candidate_tree_hash:
        raise ProductionSmokeError(
            "expert trigger inspection workspace differs from the released tree"
        )
    agent = settings.expert.generalizer
    operation_seed = canonical_json_bytes(
        {
            "agent": agent.to_dict(),
            "episode_id": episode.episode_id,
            "prompt_digest": tree_or_blob_digest(prompt.encode("utf-8")),
            "response_schema": response_schema,
            "source_base_tree_hash": base.release_manifest.candidate_tree_hash,
            "trigger_configuration_fingerprint": trigger_configuration_fingerprint,
            "workspace": str(workspace),
        }
    )
    request = CodingAgentCallRequest(
        operation_id=("agent_call_" + tree_or_blob_digest(operation_seed)[7:39]),
        role=trigger_settings.inspector_role,
        cli=agent.cli,
        model=agent.model,
        effort=agent.effort,
        prompt=prompt,
        workspace=str(workspace),
        workspace_policy=CodingAgentWorkspacePolicy.read_only(),
        timeout_seconds=agent.timeout_seconds,
        allowed_tools=agent.allowed_tools,
        prior_knowledge=None,
    )
    runner = SubprocessCodingAgentCallRunner(
        CodingAgentRunnerSettings(
            artifact_root=str(
                _private_state_root(smoke_root / "expert-trigger-inspection-artifacts")
            ),
            termination_grace_seconds=settings.expert.termination_grace_seconds,
            sensitive_file_glob_scan_max_depth=(
                settings.expert.sensitive_file_glob_scan_max_depth
            ),
        )
    )
    result = runner.run(request, response_schema)
    sealed = seal_coding_agent_operation(
        request=request,
        response_schema=response_schema,
        principal_id=trigger_settings.inspector_id,
        agent=agent,
        sensitive_file_glob_scan_max_depth=(
            settings.expert.sensitive_file_glob_scan_max_depth
        ),
        result=result,
    )
    inspected = parse_json_bytes(sealed.final_output.encode("utf-8"))
    if not isinstance(inspected, Mapping) or set(inspected) != {
        *fixed_payload,
        "description",
    }:
        raise ProductionSmokeError(
            "expert trigger inspection returned an invalid observation"
        )
    return ExpertTriggerObservation.mint(
        kind=ExpertTriggerObservationKind.MECHANICALLY_GENERAL_FIX,
        source_base_tree_hash=base.release_manifest.candidate_tree_hash,
        inspection_policy_version=trigger_settings.inspection_policy_version,
        configuration_fingerprint=trigger_configuration_fingerprint,
        inspection_operation=sealed.receipt,
        inspection_final_output=sealed.final_output,
        difficulty_signature=None,
        difficulty_evidence_signatures={},
        description=inspected["description"],
        affected_capability_ids=(capability_id,),
        affected_paths=(affected_path,),
        exact_evidence_ids=(episode.episode_id,),
        independent_lineage_ids=(),
        task_context_binding_ids=(task_context_binding_id,),
        occurrence_count=1,
    )


def _expert_trigger_inspection_response_schema(
    fixed_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Constrain a live inspection to one authenticated evidence boundary."""

    properties = {
        name: _trigger_value_schema(value)
        for name, value in fixed_payload.items()
    }
    properties["description"] = {"type": "string", "minLength": 1}
    return {
        "type": "object",
        "properties": properties,
        "required": sorted(properties),
        "additionalProperties": False,
    }


def _trigger_value_schema(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {"type": "null"}
    if type(value) is str:
        return {"type": "string"}
    if type(value) is int:
        return {"type": "integer"}
    if type(value) is list and all(type(item) is str for item in value):
        return {
            "type": "array",
            "items": {"type": "string"},
        }
    if type(value) is dict and not value:
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
    raise ProductionSmokeError("trigger fixed response value has no exact schema")


def _successor_launch_smoke(
    config_path: str,
    mode: str,
    settings: CrossRunSettings,
    smoke_root: Path,
    scope_contract: ExpertScopeContract,
    prior_evidence: Mapping[str, Mapping[str, Any]],
    *,
    clean_machine: bool,
) -> Mapping[str, Any]:
    """Launch from the exact E1/S1 receipts, optionally under a fresh state root."""

    publication = prior_evidence.get("expert-successor-publication")
    knowledge = prior_evidence.get("knowledge-publication")
    if (
        not isinstance(publication, Mapping)
        or not isinstance(publication.get("release_id"), str)
        or not isinstance(knowledge, Mapping)
        or not isinstance(knowledge.get("snapshot_id"), str)
    ):
        raise ProductionSmokeError(
            "successor launch requires exact expert and knowledge publication evidence"
        )
    launch_state_root = (
        _private_state_root(smoke_root / "clean-machine")
        if clean_machine
        else smoke_root
    )
    if clean_machine:
        _task_adapter_bootstrap_smoke(settings, launch_state_root, scope_contract)
    github = _github_services(settings, launch_state_root)
    services = _expert_validation_services(settings, launch_state_root, github)
    binding = _scope_task_bindings(scope_contract)[0]
    adapter = services.task_adapter_store.resolve_active(
        scope_contract_id=scope_contract.scope_contract_id,
        task_family_id=binding.task_family_id,
        task_adapter_id=binding.task_adapter_id,
    )
    context = adapter.manifest.release_matrix_cases[0].task_context_binding
    task_context_request = LaunchTaskContextRequest.mint(
        capability_tags=context.capability_tags,
        input_contract_fingerprint=context.input_contract_fingerprint,
        target_contract_fingerprint=context.target_contract_fingerprint,
        starting_artifact_refs=context.starting_artifact_refs,
        method_fingerprint=context.method_fingerprint,
        toolchain_fingerprint=context.toolchain_fingerprint,
        dependency_runtime_fingerprint=context.dependency_runtime_fingerprint,
        budget_hardware_envelope=context.budget_hardware_envelope,
        transfer_dimensions=context.transfer_dimensions,
    )
    launch_root = launch_state_root / "successor-launch"
    with tempfile.TemporaryDirectory(
        prefix="successor-launch-request-",
        dir=launch_state_root,
    ) as temporary:
        request_path = Path(temporary) / "request.json"
        request_path.write_bytes(
            canonical_json_bytes(
                {
                    "goal": "Run the public transport successor fixture.",
                    "additional_context": "",
                    "task_context_request": task_context_request.to_dict(),
                    "starting_artifacts": {},
                    "dependency_runtime_contract": adapter.manifest.runtime.to_dict(),
                    "budget_fidelity_envelope": {"transport_fixture": "full"},
                    "scope_id": binding.scope_id,
                    "task_family_id": binding.task_family_id,
                    "task_adapter_id": binding.task_adapter_id,
                    "requested_coding_agent": settings.launch.coding_agent.cli,
                    "objective_direction": "maximize",
                    "empty_scope_bootstrap_authorization_id": None,
                }
            )
        )
        result = resolve_launch_cross_run(
            config_path=config_path,
            mode=mode,
            request_path=request_path,
            state_root=launch_state_root,
            run_root=launch_root,
        )
    if (
        result.get("expert_release_id") != publication["release_id"]
        or result.get("knowledge_snapshot_id") != knowledge["snapshot_id"]
    ):
        raise ProductionSmokeError("successor launch did not pin exact E1 and S1")
    return {
        "run_id": result["run_id"],
        "campaign_id": result["campaign_id"],
        "launch_manifest_id": result["launch_manifest_id"],
        "bootstrap_pin_id": result["bootstrap_pin_id"],
        "expert_release_id": result["expert_release_id"],
        "knowledge_snapshot_id": result["knowledge_snapshot_id"],
        "task_adapter_manifest_id": result["task_adapter_manifest_id"],
        "workspace_baseline_commit_sha": result["workspace_baseline_commit_sha"],
        "clean_machine": clean_machine,
    }


def _concurrent_publication_smoke(
    prior_evidence: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not isinstance(prior_evidence.get("expert-successor-publication"), Mapping):
        raise ProductionSmokeError(
            "concurrent publication requires the first eligible successor release"
        )
    raise ProductionSmokeError(
        "concurrent publication requires a second independently eligible knowledge "
        "and expert child from the same authenticated parents"
    )


def _live_restart_smoke(
    prior_evidence: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    launch = prior_evidence.get("clean-machine-launch")
    if not isinstance(launch, Mapping) or not isinstance(launch.get("run_id"), str):
        raise ProductionSmokeError(
            "live restart requires a completed clean-machine launch receipt"
        )
    raise ProductionSmokeError(
        "live restart requires external daemon and host restart control"
    )


def _revocation_smoke(
    config_path: str,
    mode: str,
    smoke_root: Path,
    fixture: Mapping[str, Any],
    prior_evidence: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    publication = prior_evidence.get("expert-successor-publication")
    if not isinstance(publication, Mapping) or not isinstance(
        publication.get("candidate_id"), str
    ):
        raise ProductionSmokeError(
            "revocation requires successor publication candidate evidence"
        )
    with tempfile.TemporaryDirectory(
        prefix="expert-revocation-request-",
        dir=smoke_root,
    ) as temporary:
        request_path = Path(temporary) / "request.json"
        request_path.write_bytes(
            canonical_json_bytes(
                {
                    "candidate_id": publication["candidate_id"],
                    "revoked_at": _committed_at(fixture["committed_at"]),
                }
            )
        )
        result = revoke_expert_cross_run(
            config_path=config_path,
            mode=mode,
            request_path=request_path,
            state_root=smoke_root,
        )
    if result.get("candidate_id") != publication["candidate_id"]:
        raise ProductionSmokeError("revocation returned another candidate")
    return {
        "candidate_id": result["candidate_id"],
        "release_id": result["release_id"],
        "revocation_receipt_id": result["revocation_receipt_id"],
        "security_snapshot_id": result["security_snapshot_id"],
        "security_publication_id": result["security_publication_id"],
        "matched_revocation_ids": result["matched_revocation_ids"],
        "replayed": result["replayed"],
    }


def _synthetic_projection(
    settings: CrossRunSettings,
    fixture: Mapping[str, Any],
    scope_contract: ExpertScopeContract,
    expert_base_release_id: str,
    task_adapter_manifest_id: str,
    task_adapter_verification_receipt_id: str,
    *,
    previous: ProjectionResult | None = None,
) -> ProjectionResult:
    """Build and project the one replayable transport-smoke capture."""

    return _synthetic_capture(
        settings,
        fixture,
        scope_contract,
        expert_base_release_id,
        task_adapter_manifest_id,
        task_adapter_verification_receipt_id,
        previous=previous,
    ).projection


def _synthetic_capture(
    settings: CrossRunSettings,
    fixture: Mapping[str, Any],
    scope_contract: ExpertScopeContract,
    expert_base_release_id: str,
    task_adapter_manifest_id: str,
    task_adapter_verification_receipt_id: str,
    *,
    previous: ProjectionResult | None = None,
) -> ProductionCapture:
    values = fixture["embedding_inputs"]
    if not isinstance(values, list):
        raise ProductionSmokeError("synthetic projection inputs are incomplete")
    require_content_id(expert_base_release_id, "transport smoke expert release")
    if expert_base_release_id.split(":sha256:", 1)[0] != "expert-base-release":
        raise ProductionSmokeError("transport smoke expert release has wrong namespace")
    require_content_id(task_adapter_manifest_id, "transport smoke adapter manifest")
    require_content_id(
        task_adapter_verification_receipt_id,
        "transport smoke adapter verification receipt",
    )
    return build_production_capture(
        settings=settings,
        scope_contract=scope_contract,
        expert_base_release_id=expert_base_release_id,
        task_adapter_manifest_id=task_adapter_manifest_id,
        task_adapter_verification_receipt_id=(
            task_adapter_verification_receipt_id
        ),
        embedding_inputs=values,
        committed_at=_committed_at(fixture["committed_at"]),
        run_id=_synthetic_run_id(
            scope_contract=scope_contract,
            expert_base_release_id=expert_base_release_id,
            task_adapter_manifest_id=task_adapter_manifest_id,
            task_adapter_verification_receipt_id=(
                task_adapter_verification_receipt_id
            ),
        ),
        evaluation_fingerprint=production_capture_evaluation_fingerprint(
            settings,
            "posttrain",
        ),
        previous=previous,
    )


def _synthetic_capture_for_snapshot(
    settings: CrossRunSettings,
    fixture: Mapping[str, Any],
    scope_contract: ExpertScopeContract,
    package: KnowledgeSnapshotPackage,
    expert_base_release_id: str,
    task_adapter_manifest_id: str,
    task_adapter_verification_receipt_id: str,
) -> ProductionCapture:
    """Rebuild this config's exact raw capture or mint its direct successor."""

    run_id = _synthetic_run_id(
        scope_contract=scope_contract,
        expert_base_release_id=expert_base_release_id,
        task_adapter_manifest_id=task_adapter_manifest_id,
        task_adapter_verification_receipt_id=(
            task_adapter_verification_receipt_id
        ),
    )
    records = tuple(
        parse_knowledge_record_payload(
            envelope["record_kind"],
            envelope["payload"],
        )
        for envelope in package.record_envelopes
    )
    bundles = tuple(
        sorted(
            (
                record
                for record in records
                if isinstance(record, RunBundle)
                and record.run_id == run_id
                and record.campaign_id == "production_smoke_campaign"
            ),
            key=lambda bundle: bundle.capture_generation,
        )
    )
    if not bundles:
        return _synthetic_capture(
            settings,
            fixture,
            scope_contract,
            expert_base_release_id,
            task_adapter_manifest_id,
            task_adapter_verification_receipt_id,
        )
    if tuple(bundle.capture_generation for bundle in bundles) != tuple(
        range(len(bundles))
    ) or any(
        bundle.supersedes_bundle_id
        != (None if position == 0 else bundles[position - 1].bundle_id)
        for position, bundle in enumerate(bundles)
    ):
        raise ProductionSmokeError(
            "knowledge snapshot has an invalid synthetic run lineage"
        )
    projections = tuple(
        _projection_from_snapshot_records(bundle, records) for bundle in bundles
    )
    current_projection = projections[-1]
    current_bundle = current_projection.source_bundle
    if (
        current_bundle.configuration_fingerprint == settings.configuration_fingerprint
        and current_bundle.artifact_environment.task_adapter_manifest_id
        == task_adapter_manifest_id
        and current_bundle.artifact_environment.task_adapter_verification_receipt_id
        == task_adapter_verification_receipt_id
    ):
        if current_bundle.expert_base_release_id != expert_base_release_id:
            raise ProductionSmokeError(
                "current synthetic bundle names another bootstrap expert release"
            )
        rebuilt = _synthetic_capture(
            settings,
            fixture,
            scope_contract,
            expert_base_release_id,
            task_adapter_manifest_id,
            task_adapter_verification_receipt_id,
            previous=None if len(projections) == 1 else projections[-2],
        )
        if rebuilt.projection != current_projection:
            raise ProductionSmokeError(
                "current synthetic projection cannot reproduce its raw capture"
            )
        return rebuilt
    return _synthetic_capture(
        settings,
        fixture,
        scope_contract,
        expert_base_release_id,
        task_adapter_manifest_id,
        task_adapter_verification_receipt_id,
        previous=current_projection,
    )


def _projection_from_snapshot_records(
    bundle: RunBundle,
    records: tuple[Any, ...],
) -> ProjectionResult:
    manifests = tuple(
        record
        for record in records
        if isinstance(record, BundleProjectionManifest)
        and record.source_bundle_id == bundle.bundle_id
    )
    if len(manifests) != 1:
        raise ProductionSmokeError(
            "current synthetic bundle lacks one projection manifest"
        )
    manifest = manifests[0]
    reports = tuple(
        record
        for record in records
        if isinstance(record, SanitationReport)
        and record.report_id == manifest.sanitation_report_id
    )
    priors = tuple(
        record
        for record in records
        if isinstance(record, PriorIdea)
        and record.prior_idea_id in manifest.prior_idea_ids
    )
    episodes = tuple(
        record
        for record in records
        if isinstance(record, TransferEpisode)
        and record.episode_id in manifest.episode_ids
    )
    derivations = tuple(
        record
        for record in records
        if isinstance(record, ExecutionRevisionEvent)
        and record.event_id in manifest.derivation_object_ids
    )
    if (
        len(reports) != 1
        or tuple(sorted(item.prior_idea_id for item in priors))
        != manifest.prior_idea_ids
        or tuple(sorted(item.episode_id for item in episodes)) != manifest.episode_ids
        or tuple(sorted(item.event_id for item in derivations))
        != manifest.derivation_object_ids
    ):
        raise ProductionSmokeError("current synthetic projection closure is invalid")
    return ProjectionResult(
        source_bundle=bundle,
        sanitation_report=reports[0],
        episodes=tuple(sorted(episodes, key=lambda item: item.episode_id)),
        prior_ideas=tuple(sorted(priors, key=lambda item: item.prior_idea_id)),
        derivation_objects=tuple(
            sorted(derivations, key=lambda item: item.event_id)
        ),
    )


def _synthetic_run_id(
    *,
    scope_contract: ExpertScopeContract,
    expert_base_release_id: str,
    task_adapter_manifest_id: str,
    task_adapter_verification_receipt_id: str,
) -> str:
    identity = content_id(
        "production-smoke-run",
        {
            "scope_contract_id": scope_contract.scope_contract_id,
            "expert_base_release_id": expert_base_release_id,
            "task_adapter_manifest_id": task_adapter_manifest_id,
            "task_adapter_verification_receipt_id": (
                task_adapter_verification_receipt_id
            ),
        },
    )
    return "production_smoke_run_" + identity.split(":sha256:", 1)[1]


def _production_task_adapter_pin(
    prior_evidence: Mapping[str, Mapping[str, Any]],
    *,
    task_adapter_id: str,
) -> tuple[str, str]:
    bootstrap = prior_evidence.get("task-adapter-bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise ProductionSmokeError(
            "synthetic projection requires verified task-adapter bootstrap evidence"
        )
    adapters = bootstrap.get("adapters")
    if not isinstance(adapters, (list, tuple)):
        raise ProductionSmokeError(
            "synthetic projection requires verified task-adapter bootstrap evidence"
        )
    matches = tuple(
        adapter
        for adapter in adapters
        if isinstance(adapter, Mapping)
        and adapter.get("task_adapter_id") == task_adapter_id
    )
    if len(matches) != 1:
        raise ProductionSmokeError(
            "synthetic projection requires one exact task-adapter pin"
        )
    manifest_id = matches[0].get("task_adapter_manifest_id")
    receipt_id = matches[0].get("verification_receipt_id")
    if not isinstance(manifest_id, str) or not isinstance(receipt_id, str):
        raise ProductionSmokeError(
            "synthetic projection task-adapter pin is incomplete"
        )
    require_content_id(manifest_id, "transport smoke adapter manifest")
    require_content_id(receipt_id, "transport smoke adapter verification receipt")
    return manifest_id, receipt_id


def _seed_catalog_from_snapshot(
    catalog: CrossRunCatalog,
    package: KnowledgeSnapshotPackage,
) -> CatalogGenerationManifest:
    """Rebuild one local catalog base from authenticated snapshot source facts."""

    fact_ids = set(package.prepared.catalog_generation.fact_object_ids)
    if not fact_ids:
        return catalog.store.read_current()
    records = tuple(
        parse_knowledge_record_payload(
            envelope["record_kind"],
            envelope["payload"],
        )
        for envelope in package.record_envelopes
        if envelope["record_id"] in fact_ids
    )
    return catalog.publish(
        expected_generation=catalog.store.read_current(),
        operation_id=content_id(
            "production-smoke-catalog-import",
            {"snapshot_id": package.manifest.snapshot_id},
        ),
        objects=records,
        dependency_closure_ids=tuple(sorted(fact_ids)),
    ).generation


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
