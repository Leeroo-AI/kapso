"""Concrete production composition for GitHub-backed launch and resume."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from kapso.cross_run.canonical import canonical_json_bytes, tree_or_blob_digest
from kapso.cross_run.contracts import CrossRunTaskBindingSettings
from kapso.cross_run.embedding_space import EmbeddingSpace
from kapso.cross_run.expert.release_authority import (
    GitHubExpertReleaseActivationProvider,
)
from kapso.cross_run.expert.release_use_policy import (
    GitHubExpertReleaseUsePolicyAuthority,
)
from kapso.cross_run.github.command import (
    GitHubCommandClient,
    SubprocessCommandRunner,
)
from kapso.cross_run.github.materializer import GitHubArtifactMaterializer
from kapso.cross_run.github.resolver import GitHubArtifactResolver
from kapso.cross_run.launch.bootstrap import LaunchBootstrapCoordinator
from kapso.cross_run.launch.contracts import LaunchRequest, LaunchTaskContextRequest
from kapso.cross_run.launch.resolver import LaunchResolver
from kapso.cross_run.launch.resume import RunResumeCoordinator
from kapso.cross_run.launch.starting_artifacts import (
    build_launch_starting_artifact_provider,
    LaunchStartingArtifactSetProvider,
)
from kapso.cross_run.security_denylist import (
    AuthenticatedSecurityDenylistAuthority,
    GitHubSecurityDenylistSnapshotProvider,
    SecurityDenylistCheckpointStore,
)
from kapso.cross_run.settings import CrossRunSettings, EffectiveConfig
from kapso.cross_run.task_adapter_authority import CanonicalTaskAdapterAuthority
from kapso.cross_run.task_adapter_store import (
    TaskAdapterAuthorityRegistry,
    TaskAdapterPackageStore,
)


class ProductionLaunchCompositionError(RuntimeError):
    """Production launch services cannot share one configured authority root."""


@dataclass(frozen=True)
class ProductionLaunchPreparation:
    """Complete repository-free launch inputs derived before any provider call."""

    settings: CrossRunSettings
    binding: CrossRunTaskBindingSettings
    experiment_embedding_space: EmbeddingSpace
    starting_artifacts: LaunchStartingArtifactSetProvider
    request: LaunchRequest

    def __post_init__(self) -> None:
        if (
            type(self.settings) is not CrossRunSettings
            or type(self.binding) is not CrossRunTaskBindingSettings
            or type(self.experiment_embedding_space) is not EmbeddingSpace
            or type(self.starting_artifacts) is not LaunchStartingArtifactSetProvider
            or type(self.request) is not LaunchRequest
            or self.request.binding != self.binding
            or dict(self.request.starting_artifact_content_ids)
            != dict(self.starting_artifacts.content_ids)
        ):
            raise ProductionLaunchCompositionError(
                "production launch preparation contains mixed authority"
            )
        self.settings.scopes.resolve(self.binding.scope_id)


def build_production_launch_preparation(
    *,
    effective_config: EffectiveConfig,
    goal: str,
    task_context_request: LaunchTaskContextRequest,
    starting_artifact_sources: Mapping[str, tuple[Path, str]],
    dependency_runtime_contract: Mapping[str, Any],
    budget_fidelity_envelope: Mapping[str, Any],
    scope_id: str | None,
    task_family_id: str | None,
    task_adapter_id: str | None,
    requested_coding_agent: str | None,
    empty_scope_bootstrap_authorization_id: str | None = None,
) -> ProductionLaunchPreparation:
    """Derive one exact request from config, task semantics, and sealed inputs."""

    if (
        type(effective_config) is not EffectiveConfig
        or not isinstance(goal, str)
        or not goal.strip()
        or type(task_context_request) is not LaunchTaskContextRequest
        or not isinstance(starting_artifact_sources, Mapping)
        or not isinstance(dependency_runtime_contract, Mapping)
        or not isinstance(budget_fidelity_envelope, Mapping)
    ):
        raise ProductionLaunchCompositionError(
            "production launch preparation requires exact public inputs"
        )
    settings = effective_config.cross_run
    if type(settings) is not CrossRunSettings:
        raise ProductionLaunchCompositionError(
            "selected configuration has no cross-run settings"
        )
    binding = resolve_production_binding(
        effective_config=effective_config,
        scope_id=scope_id,
        task_family_id=task_family_id,
        task_adapter_id=task_adapter_id,
    )
    settings.scopes.resolve(binding.scope_id)
    experiment_embedding_space = production_experiment_embedding_space(settings)
    starting_artifacts = build_launch_starting_artifact_provider(
        sources=starting_artifact_sources,
        settings=settings.launch,
    )
    configured_coding_agent = effective_config.mode.get("coding_agent")
    if not isinstance(configured_coding_agent, Mapping):
        raise ProductionLaunchCompositionError(
            "selected mode has no coding-agent configuration"
        )
    coding_agent = (
        configured_coding_agent.get("type")
        if requested_coding_agent is None
        else requested_coding_agent
    )
    search_strategy = effective_config.mode.get("search_strategy")
    if not isinstance(search_strategy, Mapping):
        raise ProductionLaunchCompositionError(
            "selected mode has no search-strategy configuration"
        )
    search_mode = search_strategy.get("type")
    if not isinstance(coding_agent, str) or not isinstance(search_mode, str):
        raise ProductionLaunchCompositionError(
            "selected mode has invalid coding-agent or search-strategy identity"
        )
    request = LaunchRequest.mint(
        binding=binding,
        task_context_request=task_context_request,
        goal_digest=tree_or_blob_digest(goal.encode("utf-8")),
        starting_artifact_content_ids=dict(starting_artifacts.content_ids),
        requested_coding_agent=coding_agent,
        search_mode=search_mode,
        dependency_runtime_contract=dict(dependency_runtime_contract),
        budget_fidelity_envelope=dict(budget_fidelity_envelope),
        configuration_fingerprint=tree_or_blob_digest(
            canonical_json_bytes(
                {
                    "mode_name": effective_config.mode_name,
                    "mode": dict(effective_config.mode),
                    "cross_run_configuration_fingerprint": (
                        settings.configuration_fingerprint
                    ),
                }
            )
        ),
        empty_scope_bootstrap_authorization_id=(empty_scope_bootstrap_authorization_id),
    )
    return ProductionLaunchPreparation(
        settings=settings,
        binding=binding,
        experiment_embedding_space=experiment_embedding_space,
        starting_artifacts=starting_artifacts,
        request=request,
    )


def resolve_production_binding(
    *,
    effective_config: EffectiveConfig,
    scope_id: str | None,
    task_family_id: str | None,
    task_adapter_id: str | None,
) -> CrossRunTaskBindingSettings:
    explicit_values = (scope_id, task_family_id, task_adapter_id)
    if all(value is None for value in explicit_values):
        binding = effective_config.cross_run_binding
        if type(binding) is not CrossRunTaskBindingSettings:
            raise ProductionLaunchCompositionError(
                "scope, task family, and task adapter are required"
            )
        return binding
    if any(value is None for value in explicit_values):
        raise ProductionLaunchCompositionError(
            "scope, task family, and task adapter must be supplied together"
        )
    binding = CrossRunTaskBindingSettings(
        scope_id=scope_id,
        task_family_id=task_family_id,
        task_adapter_id=task_adapter_id,
    )
    configured = effective_config.cross_run_binding
    if configured is not None and binding != configured:
        raise ProductionLaunchCompositionError(
            "explicit task binding differs from the selected mode"
        )
    return binding


def production_experiment_embedding_space(
    settings: CrossRunSettings,
) -> EmbeddingSpace:
    embedding_settings = settings.launch.experiment_embeddings
    return EmbeddingSpace.mint(
        provider=embedding_settings.provider,
        model=embedding_settings.model,
        dimensions=embedding_settings.dimensions,
        canonicalizer_version=embedding_settings.canonicalizer_version,
    )


@dataclass(frozen=True)
class ProductionLaunchServices:
    """Concrete services needed to install adapters, launch, and resume."""

    coordinator: LaunchBootstrapCoordinator
    task_adapter_store: TaskAdapterPackageStore
    github_resolver: GitHubArtifactResolver
    security_authority: AuthenticatedSecurityDenylistAuthority
    release_use_authority: GitHubExpertReleaseUsePolicyAuthority

    def __post_init__(self) -> None:
        if (
            type(self.coordinator) is not LaunchBootstrapCoordinator
            or type(self.task_adapter_store) is not TaskAdapterPackageStore
            or type(self.github_resolver) is not GitHubArtifactResolver
            or type(self.security_authority)
            is not AuthenticatedSecurityDenylistAuthority
            or type(self.release_use_authority)
            is not GitHubExpertReleaseUsePolicyAuthority
        ):
            raise ProductionLaunchCompositionError(
                "production launch services contain substituted components"
            )


def build_production_launch_services(
    *,
    settings: CrossRunSettings,
    binding: CrossRunTaskBindingSettings,
    experiment_embedding_space: EmbeddingSpace,
    starting_artifacts: LaunchStartingArtifactSetProvider,
    state_root: Path,
) -> ProductionLaunchServices:
    """Compose existing M1-M9 services under one caller-owned state root."""

    if (
        type(settings) is not CrossRunSettings
        or type(binding) is not CrossRunTaskBindingSettings
        or type(experiment_embedding_space) is not EmbeddingSpace
        or type(starting_artifacts) is not LaunchStartingArtifactSetProvider
        or not isinstance(state_root, Path)
        or not state_root.is_absolute()
        or state_root != Path(os.path.abspath(state_root))
        or state_root in {Path("/"), Path.home()}
    ):
        raise ProductionLaunchCompositionError(
            "production launch requires exact configured inputs"
        )
    settings.scopes.resolve(binding.scope_id)
    _require_state_root(state_root)
    github_settings = settings.github
    client = GitHubCommandClient(
        SubprocessCommandRunner(),
        working_directory=state_root,
        timeout_seconds=github_settings.command_timeout_seconds,
        api_version=github_settings.api_version,
        minimum_cli_version=github_settings.minimum_cli_version,
        control_blob_size_bytes=github_settings.control_blob_size_bytes,
    )
    github_resolver = GitHubArtifactResolver(
        client,
        github_settings,
        settings.scopes,
    )
    materializer = GitHubArtifactMaterializer(
        client,
        github_settings,
        state_root,
    )
    task_adapter_settings = settings.expert.task_adapters
    task_adapter_authorities = tuple(
        CanonicalTaskAdapterAuthority(authority)
        for authority in task_adapter_settings.trusted_authorities
    )
    task_adapter_store = TaskAdapterPackageStore(
        state_root / task_adapter_settings.state_path,
        state_root,
        task_adapter_settings,
        TaskAdapterAuthorityRegistry(
            task_adapter_settings,
            task_adapter_authorities,
        ),
    )
    security_state_path = state_root / settings.launch.security_denylist_state_path
    security_trusted_root = security_state_path.parent
    _require_private_directory(security_trusted_root, state_root)
    security_authority = AuthenticatedSecurityDenylistAuthority(
        settings.scopes,
        settings.launch,
        GitHubSecurityDenylistSnapshotProvider(github_resolver, materializer),
        SecurityDenylistCheckpointStore(
            security_state_path,
            security_trusted_root,
            settings.launch.security_denylist_checkpoint_size_bytes,
        ),
    )
    activation_provider = GitHubExpertReleaseActivationProvider(
        github_resolver,
        materializer,
    )
    release_use_authority = GitHubExpertReleaseUsePolicyAuthority(
        github_resolver,
        materializer,
        activation_provider,
    )
    resolver = LaunchResolver(
        settings=settings,
        experiment_embedding_space=experiment_embedding_space,
        github_resolver=github_resolver,
        materializer=materializer,
        task_adapters=task_adapter_store,
        starting_artifacts=starting_artifacts,
        release_use_authority=release_use_authority,
        security_authority=security_authority,
    )
    resume = RunResumeCoordinator(
        settings=settings,
        binding=binding,
        security_authority=security_authority,
        release_use_authority=release_use_authority,
    )
    return ProductionLaunchServices(
        coordinator=LaunchBootstrapCoordinator(
            settings=settings,
            binding=binding,
            resolver=resolver,
            resume_coordinator=resume,
        ),
        task_adapter_store=task_adapter_store,
        github_resolver=github_resolver,
        security_authority=security_authority,
        release_use_authority=release_use_authority,
    )


def _require_state_root(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = os.stat(path, follow_symlinks=False)
    if (
        path.resolve() != path
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
    ):
        raise ProductionLaunchCompositionError(
            "production launch state root is not an owned real directory"
        )


def _require_private_directory(path: Path, state_root: Path) -> None:
    if state_root not in path.parents:
        raise ProductionLaunchCompositionError(
            "production launch private state escapes its root"
        )
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
    metadata = os.stat(path, follow_symlinks=False)
    if (
        path.resolve() != path
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
    ):
        raise ProductionLaunchCompositionError(
            "production launch private state root is unsafe"
        )


__all__ = [
    "build_production_launch_preparation",
    "build_production_launch_services",
    "production_experiment_embedding_space",
    "ProductionLaunchCompositionError",
    "ProductionLaunchPreparation",
    "ProductionLaunchServices",
    "resolve_production_binding",
]
