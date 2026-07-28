"""Concrete production composition for GitHub-backed launch and resume."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

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
from kapso.cross_run.launch.resume import RunResumeCoordinator
from kapso.cross_run.launch.starting_artifacts import (
    LaunchStartingArtifactSetProvider,
)
from kapso.cross_run.security_denylist import (
    AuthenticatedSecurityDenylistAuthority,
    GitHubSecurityDenylistSnapshotProvider,
    SecurityDenylistCheckpointStore,
)
from kapso.cross_run.settings import CrossRunSettings
from kapso.cross_run.task_adapter_authority import CanonicalTaskAdapterAuthority
from kapso.cross_run.task_adapter_store import (
    TaskAdapterAuthorityRegistry,
    TaskAdapterPackageStore,
)
from kapso.cross_run.launch.resolver import LaunchResolver


class ProductionLaunchCompositionError(RuntimeError):
    """Production launch services cannot share one configured authority root."""


@dataclass(frozen=True)
class ProductionLaunchServices:
    """Concrete services needed to install adapters, launch, and resume."""

    coordinator: LaunchBootstrapCoordinator
    task_adapter_store: TaskAdapterPackageStore
    github_resolver: GitHubArtifactResolver
    security_authority: AuthenticatedSecurityDenylistAuthority

    def __post_init__(self) -> None:
        if (
            type(self.coordinator) is not LaunchBootstrapCoordinator
            or type(self.task_adapter_store) is not TaskAdapterPackageStore
            or type(self.github_resolver) is not GitHubArtifactResolver
            or type(self.security_authority)
            is not AuthenticatedSecurityDenylistAuthority
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
    "build_production_launch_services",
    "ProductionLaunchCompositionError",
    "ProductionLaunchServices",
]
