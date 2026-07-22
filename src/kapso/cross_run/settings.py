"""Strict configuration and scope routing for cross-run behavior."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    require_identifier,
    to_json_value,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ContractValidationError,
    CrossRunTaskBindingSettings,
    ExpertScopeContract,
    ExpertValidationStage,
    ExpertValidationTrack,
    IdentityConflictError,
    ObjectiveDirection,
    ScopeRepositorySettings,
    StrictContract,
)
from kapso.cross_run.git_refs import require_git_ref_name

_SECRET_KEY_PATTERN = re.compile(
    r"^(?:.*_)?(?:access_token|api_key|auth_token|credential|credentials|oauth_token|password|private_key|secret|secrets|token)$",
    re.IGNORECASE,
)
_CLI_NAMES = ("claude_code", "codex")
_MINIMUM_ZSTD_WINDOW_SIZE_BYTES = 1024
_PUBLICATION_NORMAL_FIXED_CONTENT_WRITES = 18
_PUBLICATION_RECOVERY_FIXED_CONTENT_WRITES = 10
_PUBLICATION_AND_RESOLUTION_FIXED_READS = 64
_CONTENT_WRITE_REQUEST_POINTS = 5


class CrossRunConfigurationError(ValueError):
    """Cross-run configuration is missing, malformed, or contradictory."""


def _require_path(value: str, name: str) -> None:
    if not value:
        raise CrossRunConfigurationError(f"{name} must not be empty")
    path = PurePosixPath(value)
    if ".." in path.parts or value != path.as_posix():
        raise CrossRunConfigurationError(f"{name} must be a normalized path")


def _require_relative_path(value: str, name: str) -> PurePosixPath:
    _require_path(value, name)
    path = PurePosixPath(value)
    if path.is_absolute() or path == PurePosixPath("."):
        raise CrossRunConfigurationError(f"{name} must be workspace relative")
    return path


def _require_positive(value: int | float, name: str) -> None:
    if value <= 0:
        raise CrossRunConfigurationError(f"{name} must be positive")


def _require_non_negative(value: int | float, name: str) -> None:
    if value < 0:
        raise CrossRunConfigurationError(f"{name} must be non-negative")


def _require_ratio(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise CrossRunConfigurationError(f"{name} must be in [0, 1]")


def _require_cli(value: str, name: str) -> None:
    if value not in _CLI_NAMES:
        raise CrossRunConfigurationError(f"{name} must be one of {_CLI_NAMES}")


def _reject_secret_keys(value: Any, path: str = "cross_run") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _SECRET_KEY_PATTERN.search(str(key)):
                raise CrossRunConfigurationError(
                    f"secret-bearing configuration key is forbidden: {path}.{key}"
                )
            _reject_secret_keys(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for position, child in enumerate(value):
            _reject_secret_keys(child, f"{path}[{position}]")


@dataclass(frozen=True)
class ScopeRegistrySettings(StrictContract):
    scopes: tuple[ScopeRepositorySettings, ...]

    def _validate(self) -> None:
        if not self.scopes:
            raise CrossRunConfigurationError("scope registry must not be empty")
        scope_ids = tuple(scope.scope_id for scope in self.scopes)
        if scope_ids != tuple(sorted(set(scope_ids))):
            raise IdentityConflictError("scope registry must be sorted and unique")
        repositories: set[str] = set()
        for scope in self.scopes:
            for repository in (
                scope.expert_repository,
                scope.knowledge_repository,
                scope.security_repository,
            ):
                if repository in repositories:
                    raise IdentityConflictError(
                        f"repository {repository} belongs to more than one scope"
                    )
                repositories.add(repository)

    @classmethod
    def from_config(cls, payload: Mapping[str, Any]) -> ScopeRegistrySettings:
        if not isinstance(payload, Mapping):
            raise CrossRunConfigurationError("cross_run.scopes must be an object")
        scopes: list[ScopeRepositorySettings] = []
        for scope_id in sorted(payload):
            value = payload[scope_id]
            if not isinstance(value, Mapping) or set(value) != {"repositories"}:
                raise CrossRunConfigurationError(
                    f"scope {scope_id} must contain only repositories"
                )
            repositories = value["repositories"]
            if not isinstance(repositories, Mapping) or set(repositories) != {
                "expert",
                "knowledge",
                "security",
            }:
                raise CrossRunConfigurationError(
                    f"scope {scope_id} repositories must contain expert, knowledge, and security"
                )
            scopes.append(
                ScopeRepositorySettings(
                    scope_id=scope_id,
                    expert_repository=repositories["expert"],
                    knowledge_repository=repositories["knowledge"],
                    security_repository=repositories["security"],
                )
            )
        return cls(scopes=tuple(scopes))

    def resolve(self, scope_id: str) -> ScopeRepositorySettings:
        matches = tuple(scope for scope in self.scopes if scope.scope_id == scope_id)
        if not matches:
            raise CrossRunConfigurationError(f"unknown cross-run scope: {scope_id}")
        return matches[0]

    @property
    def fingerprint(self) -> str:
        return tree_or_blob_digest(canonical_json_bytes(self.to_config()))

    def to_config(self) -> dict[str, Any]:
        return {
            scope.scope_id: {
                "repositories": {
                    "expert": scope.expert_repository,
                    "knowledge": scope.knowledge_repository,
                    "security": scope.security_repository,
                }
            }
            for scope in self.scopes
        }


@dataclass(frozen=True)
class GitHubSettings(StrictContract):
    api_version: str
    minimum_cli_version: str
    default_branch: str
    publisher_login: str
    commit_author_name: str
    commit_author_email: str
    expert_tag_prefix: str
    knowledge_tag_prefix: str
    security_denylist_tag_prefix: str
    cache_path: str
    command_timeout_seconds: int
    release_asset_size_bytes: int
    release_asset_count_limit: int
    materialized_asset_size_bytes: int
    archive_entry_limit: int
    zstd_window_size_bytes: int
    source_tree_size_bytes: int
    source_entry_limit: int
    git_tree_request_size_bytes: int
    content_write_budget_per_minute: int
    request_point_budget_per_minute: int
    cache_entry_limit: int
    git_tree_metadata_size_bytes: int
    control_blob_size_bytes: int
    cache_retention_releases: int

    def _validate(self) -> None:
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", self.api_version):
            raise CrossRunConfigurationError("invalid GitHub API version")
        if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", self.minimum_cli_version):
            raise CrossRunConfigurationError("invalid minimum GitHub CLI version")
        require_git_ref_name(
            f"refs/heads/{self.default_branch}",
            "GitHub default branch",
            qualified=True,
            error_type=CrossRunConfigurationError,
        )
        if not re.fullmatch(r"[A-Za-z0-9-]+", self.publisher_login):
            raise CrossRunConfigurationError("invalid GitHub publisher login")
        if not self.commit_author_name.strip():
            raise CrossRunConfigurationError("GitHub commit author name is required")
        if not re.fullmatch(r"[^@\s]+@[^@\s]+", self.commit_author_email):
            raise CrossRunConfigurationError("invalid GitHub commit author email")
        for name in (
            "expert_tag_prefix",
            "knowledge_tag_prefix",
            "security_denylist_tag_prefix",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.endswith("/"):
                raise CrossRunConfigurationError(f"invalid {name}")
            require_git_ref_name(
                f"refs/tags/{value}artifact",
                name,
                qualified=True,
                error_type=CrossRunConfigurationError,
            )
        tag_prefixes = {
            self.expert_tag_prefix,
            self.knowledge_tag_prefix,
            self.security_denylist_tag_prefix,
        }
        if len(tag_prefixes) != 3:
            raise CrossRunConfigurationError("publication tag prefixes must differ")
        _require_path(self.cache_path, "github.cache_path")
        _require_positive(
            self.command_timeout_seconds, "github.command_timeout_seconds"
        )
        _require_positive(
            self.release_asset_size_bytes, "github.release_asset_size_bytes"
        )
        _require_positive(
            self.release_asset_count_limit, "github.release_asset_count_limit"
        )
        _require_positive(
            self.materialized_asset_size_bytes,
            "github.materialized_asset_size_bytes",
        )
        _require_positive(self.archive_entry_limit, "github.archive_entry_limit")
        _require_positive(
            self.zstd_window_size_bytes,
            "github.zstd_window_size_bytes",
        )
        if self.zstd_window_size_bytes < _MINIMUM_ZSTD_WINDOW_SIZE_BYTES:
            raise CrossRunConfigurationError(
                "github.zstd_window_size_bytes is below the decoder minimum"
            )
        _require_positive(self.source_tree_size_bytes, "github.source_tree_size_bytes")
        _require_positive(self.source_entry_limit, "github.source_entry_limit")
        _require_positive(
            self.git_tree_request_size_bytes,
            "github.git_tree_request_size_bytes",
        )
        _require_positive(
            self.content_write_budget_per_minute,
            "github.content_write_budget_per_minute",
        )
        _require_positive(
            self.request_point_budget_per_minute,
            "github.request_point_budget_per_minute",
        )
        maximum_content_writes = max(
            _PUBLICATION_NORMAL_FIXED_CONTENT_WRITES + self.release_asset_count_limit,
            _PUBLICATION_RECOVERY_FIXED_CONTENT_WRITES
            + 2 * self.release_asset_count_limit,
        )
        if maximum_content_writes > self.content_write_budget_per_minute:
            raise CrossRunConfigurationError(
                "GitHub publication exceeds configured content-write budget"
            )
        maximum_request_points = (
            self.source_entry_limit
            + 1
            + _PUBLICATION_AND_RESOLUTION_FIXED_READS
            + maximum_content_writes * _CONTENT_WRITE_REQUEST_POINTS
        )
        if maximum_request_points > self.request_point_budget_per_minute:
            raise CrossRunConfigurationError(
                "GitHub publication/resolution exceeds configured request-point budget"
            )
        _require_positive(self.cache_entry_limit, "github.cache_entry_limit")
        _require_positive(
            self.git_tree_metadata_size_bytes,
            "github.git_tree_metadata_size_bytes",
        )
        _require_positive(
            self.control_blob_size_bytes, "github.control_blob_size_bytes"
        )
        _require_positive(
            self.cache_retention_releases, "github.cache_retention_releases"
        )


@dataclass(frozen=True)
class CaptureSettings(StrictContract):
    state_path: str
    quarantine_path: str
    checkpoint_path: str
    experiment_history_path: str
    journal_filename: str
    bundle_asset_size_bytes: int
    bundle_entry_limit: int
    bundle_lineage_limit: int
    source_entry_limit: int
    git_command_timeout_seconds: int
    git_command_output_bytes: int
    score_comparison_tolerance: float
    capture_interval_seconds: int
    quarantine_retention_generations: int

    def _validate(self) -> None:
        state_path = _require_relative_path(self.state_path, "capture.state_path")
        quarantine_path = _require_relative_path(
            self.quarantine_path, "capture.quarantine_path"
        )
        checkpoint_path = _require_relative_path(
            self.checkpoint_path, "capture.checkpoint_path"
        )
        experiment_history_path = _require_relative_path(
            self.experiment_history_path,
            "capture.experiment_history_path",
        )
        journal_filename = _require_relative_path(
            self.journal_filename, "capture.journal_filename"
        )
        if len(journal_filename.parts) != 1:
            raise CrossRunConfigurationError(
                "capture.journal_filename must be one filename"
            )
        if (
            state_path == quarantine_path
            or state_path in quarantine_path.parents
            or (quarantine_path in state_path.parents)
        ):
            raise CrossRunConfigurationError(
                "capture state and quarantine paths must be disjoint"
            )
        if checkpoint_path == experiment_history_path:
            raise CrossRunConfigurationError("capture authority paths must be distinct")
        if any(
            path == quarantine_path or quarantine_path in path.parents
            for path in (checkpoint_path, experiment_history_path)
        ):
            raise CrossRunConfigurationError(
                "capture authority paths must be outside quarantine"
            )
        _require_positive(
            self.bundle_asset_size_bytes, "capture.bundle_asset_size_bytes"
        )
        _require_positive(self.bundle_entry_limit, "capture.bundle_entry_limit")
        _require_positive(self.bundle_lineage_limit, "capture.bundle_lineage_limit")
        _require_positive(self.source_entry_limit, "capture.source_entry_limit")
        _require_positive(
            self.git_command_timeout_seconds,
            "capture.git_command_timeout_seconds",
        )
        _require_positive(
            self.git_command_output_bytes,
            "capture.git_command_output_bytes",
        )
        _require_positive(
            self.score_comparison_tolerance,
            "capture.score_comparison_tolerance",
        )
        _require_positive(
            self.capture_interval_seconds, "capture.capture_interval_seconds"
        )
        _require_positive(
            self.quarantine_retention_generations,
            "capture.quarantine_retention_generations",
        )


@dataclass(frozen=True)
class SanitationSettings(StrictContract):
    policy_version: str
    max_file_bytes: int
    allowed_suffixes: tuple[str, ...]
    allowed_filenames: tuple[str, ...]
    allowed_spdx_licenses: tuple[str, ...]
    denied_path_patterns: tuple[str, ...]

    def _validate(self) -> None:
        if not self.policy_version:
            raise CrossRunConfigurationError("sanitation.policy_version is required")
        _require_positive(self.max_file_bytes, "sanitation.max_file_bytes")
        if not self.allowed_suffixes or self.allowed_suffixes != tuple(
            sorted(set(self.allowed_suffixes))
        ):
            raise CrossRunConfigurationError(
                "sanitation.allowed_suffixes must be sorted and unique"
            )
        for values, name in (
            (self.allowed_filenames, "sanitation.allowed_filenames"),
            (self.allowed_spdx_licenses, "sanitation.allowed_spdx_licenses"),
        ):
            if not values or values != tuple(sorted(set(values))):
                raise CrossRunConfigurationError(f"{name} must be sorted and unique")
            if any(not value for value in values):
                raise CrossRunConfigurationError(f"{name} must not contain empty text")
        if not self.denied_path_patterns or self.denied_path_patterns != tuple(
            sorted(set(self.denied_path_patterns))
        ):
            raise CrossRunConfigurationError(
                "sanitation.denied_path_patterns must be sorted and unique"
            )
        if any(not value for value in self.denied_path_patterns):
            raise CrossRunConfigurationError(
                "sanitation.denied_path_patterns must not contain empty text"
            )
        if any(
            value.startswith("token:")
            and re.fullmatch(r"token:[a-z0-9]+", value) is None
            for value in self.denied_path_patterns
        ):
            raise CrossRunConfigurationError(
                "sanitation token path patterns are invalid"
            )


@dataclass(frozen=True)
class CodingAgentSettings(StrictContract):
    cli: str
    model: str
    timeout_seconds: int
    effort: str
    allowed_tools: tuple[str, ...]

    def _validate(self) -> None:
        _require_cli(self.cli, "coding agent cli")
        if not self.model:
            raise CrossRunConfigurationError("coding agent model must not be empty")
        _require_positive(self.timeout_seconds, "coding agent timeout_seconds")
        valid_efforts = {
            "codex": {"minimal", "low", "medium", "high", "xhigh"},
            "claude_code": {"low", "medium", "high", "xhigh", "max"},
        }
        if self.effort not in valid_efforts[self.cli]:
            raise CrossRunConfigurationError(
                "coding agent effort is incompatible with its CLI"
            )
        supported_tools = {
            "codex": {"Read", "WebSearch"},
            "claude_code": {
                "Edit",
                "Glob",
                "Grep",
                "Read",
                "WebSearch",
                "Write",
            },
        }
        if self.allowed_tools != tuple(sorted(set(self.allowed_tools))) or not set(
            self.allowed_tools
        ).issubset(supported_tools[self.cli]):
            raise CrossRunConfigurationError(
                "coding agent tools must be supported, sorted, and unique"
            )


@dataclass(frozen=True)
class CatalogReviewerSettings(StrictContract):
    reviewer_id: str
    reviewer_role: str
    rubric_version: str
    agent: CodingAgentSettings

    def _validate(self) -> None:
        for value, name in (
            (self.reviewer_id, "catalog reviewer_id"),
            (self.reviewer_role, "catalog reviewer_role"),
            (self.rubric_version, "catalog reviewer rubric_version"),
        ):
            require_identifier(value, name)


@dataclass(frozen=True)
class CatalogAdmissionSettings(StrictContract):
    policy_version: str
    approval_judgment: str
    rejection_judgment: str
    required_approvals: int
    required_rejections: int
    minimum_supporting_runs: int
    minimum_supporting_task_contexts: int
    require_comparable_support: bool
    require_isolated_support: bool

    def _validate(self) -> None:
        for value, name in (
            (self.policy_version, "catalog admission policy_version"),
            (self.approval_judgment, "catalog admission approval_judgment"),
            (self.rejection_judgment, "catalog admission rejection_judgment"),
        ):
            require_identifier(value, name)
        if self.approval_judgment == self.rejection_judgment:
            raise CrossRunConfigurationError(
                "catalog approval and rejection judgments must differ"
            )
        for value, name in (
            (self.required_approvals, "catalog admission required_approvals"),
            (self.required_rejections, "catalog admission required_rejections"),
            (
                self.minimum_supporting_runs,
                "catalog admission minimum_supporting_runs",
            ),
            (
                self.minimum_supporting_task_contexts,
                "catalog admission minimum_supporting_task_contexts",
            ),
        ):
            _require_positive(value, name)


@dataclass(frozen=True)
class CatalogSettings(StrictContract):
    state_path: str
    agent_artifact_path: str
    termination_grace_seconds: int
    sensitive_file_glob_scan_max_depth: int
    claim_packet_record_limit: int
    review_packet_record_limit: int
    claim_proposer_id: str
    claim_proposer_role: str
    claim_proposer: CodingAgentSettings
    reviewers: tuple[CatalogReviewerSettings, ...]
    admission: CatalogAdmissionSettings
    publication_interval_runs: int

    def _validate(self) -> None:
        state_path = _require_relative_path(self.state_path, "catalog.state_path")
        artifact_path = _require_relative_path(
            self.agent_artifact_path,
            "catalog.agent_artifact_path",
        )
        if (
            state_path == artifact_path
            or state_path in artifact_path.parents
            or artifact_path in state_path.parents
        ):
            raise CrossRunConfigurationError(
                "catalog state and agent artifacts must be disjoint"
            )
        _require_positive(
            self.termination_grace_seconds,
            "catalog.termination_grace_seconds",
        )
        _require_positive(
            self.sensitive_file_glob_scan_max_depth,
            "catalog.sensitive_file_glob_scan_max_depth",
        )
        _require_positive(
            self.claim_packet_record_limit,
            "catalog.claim_packet_record_limit",
        )
        _require_positive(
            self.review_packet_record_limit,
            "catalog.review_packet_record_limit",
        )
        require_identifier(self.claim_proposer_id, "catalog.claim_proposer_id")
        require_identifier(self.claim_proposer_role, "catalog.claim_proposer_role")
        if not self.reviewers:
            raise CrossRunConfigurationError("catalog reviewers must not be empty")
        reviewer_ids = tuple(reviewer.reviewer_id for reviewer in self.reviewers)
        if reviewer_ids != tuple(sorted(set(reviewer_ids))):
            raise CrossRunConfigurationError(
                "catalog reviewers must be sorted and uniquely identified"
            )
        if self.claim_proposer_id in reviewer_ids:
            raise CrossRunConfigurationError(
                "catalog claim proposer cannot be a reviewer"
            )
        if self.admission.required_approvals > len(self.reviewers):
            raise CrossRunConfigurationError(
                "catalog approval quorum exceeds configured reviewers"
            )
        if self.admission.required_rejections > len(self.reviewers):
            raise CrossRunConfigurationError(
                "catalog rejection quorum exceeds configured reviewers"
            )
        _require_positive(
            self.publication_interval_runs, "catalog.publication_interval_runs"
        )

    @property
    def configuration_fingerprint(self) -> str:
        return tree_or_blob_digest(canonical_json_bytes(self.to_dict()))


@dataclass(frozen=True)
class EmbeddingSettings(StrictContract):
    enabled: bool
    provider: str
    model: str
    dimensions: int
    batch_size: int
    timeout_seconds: int
    max_retries: int
    canonicalizer_version: str

    def _validate(self) -> None:
        if self.provider != "openai":
            raise CrossRunConfigurationError(
                "only the configured OpenAI provider is supported"
            )
        if not self.model:
            raise CrossRunConfigurationError("embedding model is required")
        if not self.canonicalizer_version:
            raise CrossRunConfigurationError(
                "embedding canonicalizer version is required"
            )
        _require_positive(self.dimensions, "knowledge.embeddings.dimensions")
        _require_positive(self.batch_size, "knowledge.embeddings.batch_size")
        _require_positive(self.timeout_seconds, "knowledge.embeddings.timeout_seconds")
        _require_non_negative(self.max_retries, "knowledge.embeddings.max_retries")


@dataclass(frozen=True)
class RetrievalSettings(StrictContract):
    lexical_weight: float
    max_records: int
    max_records_per_run: int
    max_records_per_family: int
    max_records_per_lineage: int
    max_records_per_outcome: int
    max_records_per_type: int
    prompt_byte_budget: int
    materialization_byte_budget: int

    def _validate(self) -> None:
        _require_ratio(self.lexical_weight, "knowledge.retrieval.lexical_weight")
        for name in (
            "max_records",
            "max_records_per_run",
            "max_records_per_family",
            "max_records_per_lineage",
            "max_records_per_outcome",
            "max_records_per_type",
            "prompt_byte_budget",
            "materialization_byte_budget",
        ):
            _require_positive(getattr(self, name), f"knowledge.retrieval.{name}")
        for name in (
            "max_records_per_run",
            "max_records_per_family",
            "max_records_per_lineage",
            "max_records_per_outcome",
            "max_records_per_type",
        ):
            if getattr(self, name) > self.max_records:
                raise CrossRunConfigurationError(
                    f"knowledge.retrieval.{name} exceeds total record cap"
                )
        if self.materialization_byte_budget < self.prompt_byte_budget:
            raise CrossRunConfigurationError(
                "knowledge retrieval materialization budget is below prompt budget"
            )

    @property
    def semantic_weight(self) -> float:
        return 1.0 - self.lexical_weight


@dataclass(frozen=True)
class KnowledgeSettings(StrictContract):
    snapshot_path: str
    index_path: str
    archive_compression_level: int
    embeddings: EmbeddingSettings
    retrieval: RetrievalSettings

    def _validate(self) -> None:
        _require_path(self.snapshot_path, "knowledge.snapshot_path")
        _require_path(self.index_path, "knowledge.index_path")
        _require_positive(
            self.archive_compression_level,
            "knowledge.archive_compression_level",
        )
        if self.archive_compression_level > 22:
            raise CrossRunConfigurationError(
                "knowledge.archive_compression_level exceeds the zstd maximum"
            )


@dataclass(frozen=True)
class ExpertTriggerSettings(StrictContract):
    policy_version: str
    inspection_policy_version: str
    inspector_id: str
    inspector_role: str
    minimum_failure_lineages: int
    minimum_failure_contexts: int
    minimum_success_lineages: int
    minimum_success_contexts: int
    minimum_duplicate_files: int
    maximum_ancestor_candidates: int

    def _validate(self) -> None:
        require_identifier(self.policy_version, "expert trigger policy_version")
        require_identifier(
            self.inspection_policy_version,
            "expert trigger inspection_policy_version",
        )
        require_identifier(self.inspector_id, "expert trigger inspector_id")
        require_identifier(self.inspector_role, "expert trigger inspector_role")
        for value, name in (
            (self.minimum_failure_lineages, "minimum_failure_lineages"),
            (self.minimum_failure_contexts, "minimum_failure_contexts"),
            (self.minimum_success_lineages, "minimum_success_lineages"),
            (self.minimum_success_contexts, "minimum_success_contexts"),
            (self.minimum_duplicate_files, "minimum_duplicate_files"),
            (self.maximum_ancestor_candidates, "maximum_ancestor_candidates"),
        ):
            _require_positive(value, f"expert.triggers.{name}")


@dataclass(frozen=True)
class ExpertEvaluatorSettings(StrictContract):
    stage: ExpertValidationStage
    evaluator_id: str
    evaluator_role: str
    evaluator_version: str
    timeout_seconds: int

    def _validate(self) -> None:
        for value, name in (
            (self.evaluator_id, "expert evaluator_id"),
            (self.evaluator_role, "expert evaluator_role"),
            (self.evaluator_version, "expert evaluator_version"),
        ):
            require_identifier(value, name)
        _require_positive(
            self.timeout_seconds,
            f"expert evaluator {self.stage.value} timeout_seconds",
        )


@dataclass(frozen=True)
class ExpertReviewerSettings(StrictContract):
    reviewer_id: str
    reviewer_role: str
    rubric_version: str
    agent: CodingAgentSettings

    def _validate(self) -> None:
        for value, name in (
            (self.reviewer_id, "expert reviewer_id"),
            (self.reviewer_role, "expert reviewer_role"),
            (self.rubric_version, "expert reviewer rubric_version"),
        ):
            require_identifier(value, name)
        if self.agent.allowed_tools:
            raise CrossRunConfigurationError("expert reviewers must not receive tools")


@dataclass(frozen=True)
class ExpertParetoDimensionSettings(StrictContract):
    dimension_id: str
    direction: ObjectiveDirection
    hard_regression_ratio: float
    noise_floor_ratio: float

    def _validate(self) -> None:
        require_identifier(self.dimension_id, "expert Pareto dimension_id")
        _require_ratio(
            self.hard_regression_ratio,
            f"expert Pareto {self.dimension_id} hard_regression_ratio",
        )
        _require_ratio(
            self.noise_floor_ratio,
            f"expert Pareto {self.dimension_id} noise_floor_ratio",
        )


@dataclass(frozen=True)
class ExpertPromotionPolicySettings(StrictContract):
    policy_version: str
    approval_judgment: str
    rejection_judgment: str
    required_approvals: int
    required_rejections: int
    minimum_independent_contexts: int
    minimum_repeat_evaluations: int
    pareto_dimensions: tuple[ExpertParetoDimensionSettings, ...]

    def _validate(self) -> None:
        for value, name in (
            (self.policy_version, "expert promotion policy_version"),
            (self.approval_judgment, "expert promotion approval_judgment"),
            (self.rejection_judgment, "expert promotion rejection_judgment"),
        ):
            require_identifier(value, name)
        if self.approval_judgment == self.rejection_judgment:
            raise CrossRunConfigurationError(
                "expert approval and rejection judgments must differ"
            )
        for value, name in (
            (self.required_approvals, "required_approvals"),
            (self.required_rejections, "required_rejections"),
            (self.minimum_independent_contexts, "minimum_independent_contexts"),
            (self.minimum_repeat_evaluations, "minimum_repeat_evaluations"),
        ):
            _require_positive(value, f"expert.validation.promotion.{name}")
        if not self.pareto_dimensions:
            raise CrossRunConfigurationError(
                "expert Pareto dimensions must not be empty"
            )
        dimension_ids = tuple(
            dimension.dimension_id for dimension in self.pareto_dimensions
        )
        if dimension_ids != tuple(sorted(set(dimension_ids))):
            raise CrossRunConfigurationError(
                "expert Pareto dimensions must be sorted and unique"
            )


@dataclass(frozen=True)
class ExpertValidationPolicySettings(StrictContract):
    source_replay_selection_policy_version: str
    source_replay_request_policy_version: str
    source_replay_paired_execution_protocol_version: str
    source_replay_execution_provider_id: str
    source_replay_execution_provider_version: str
    source_replay_sandbox_policy_version: str
    source_replay_termination_grace_seconds: int
    source_replay_cpu_millicore_limit: int
    source_replay_memory_byte_limit: int
    source_replay_shared_memory_byte_limit: int
    source_replay_process_limit: int
    source_replay_open_file_limit: int
    source_replay_writable_entry_limit: int
    source_replay_writable_byte_limit: int
    source_replay_stdout_byte_limit: int
    source_replay_stderr_byte_limit: int
    source_replay_task_request_byte_limit: int
    source_replay_journal_event_byte_limit: int
    source_replay_result_byte_limit: int
    source_replay_staging_entry_limit: int
    source_replay_accelerator_class_id: str | None
    source_replay_accelerator_count: int
    source_replay_episode_limit: int
    source_replay_bundle_limit: int
    source_replay_context_materializer_id: str
    source_replay_context_materializer_version: str
    source_replay_materialization_entry_limit: int
    source_replay_materialization_byte_limit: int
    source_replay_materialization_timeout_seconds: int
    source_replay_score_comparison_tolerance: float
    sealed_canary_trust_root: str | None
    architecture_requires_sealed_canary: bool
    artifact_entry_limit: int
    artifact_byte_limit: int
    evaluators: tuple[ExpertEvaluatorSettings, ...]
    reviewers: tuple[ExpertReviewerSettings, ...]
    promotion: ExpertPromotionPolicySettings

    def _validate(self) -> None:
        require_identifier(
            self.source_replay_selection_policy_version,
            "expert.validation.policy.source_replay_selection_policy_version",
        )
        require_identifier(
            self.source_replay_request_policy_version,
            "expert.validation.policy.source_replay_request_policy_version",
        )
        for value, name in (
            (
                self.source_replay_paired_execution_protocol_version,
                "source_replay_paired_execution_protocol_version",
            ),
            (
                self.source_replay_execution_provider_id,
                "source_replay_execution_provider_id",
            ),
            (
                self.source_replay_execution_provider_version,
                "source_replay_execution_provider_version",
            ),
            (
                self.source_replay_sandbox_policy_version,
                "source_replay_sandbox_policy_version",
            ),
        ):
            require_identifier(value, f"expert.validation.policy.{name}")
        for value, name in (
            (
                self.source_replay_termination_grace_seconds,
                "source_replay_termination_grace_seconds",
            ),
            (
                self.source_replay_cpu_millicore_limit,
                "source_replay_cpu_millicore_limit",
            ),
            (
                self.source_replay_memory_byte_limit,
                "source_replay_memory_byte_limit",
            ),
            (
                self.source_replay_shared_memory_byte_limit,
                "source_replay_shared_memory_byte_limit",
            ),
            (self.source_replay_process_limit, "source_replay_process_limit"),
            (
                self.source_replay_open_file_limit,
                "source_replay_open_file_limit",
            ),
            (
                self.source_replay_writable_entry_limit,
                "source_replay_writable_entry_limit",
            ),
            (
                self.source_replay_writable_byte_limit,
                "source_replay_writable_byte_limit",
            ),
            (
                self.source_replay_stdout_byte_limit,
                "source_replay_stdout_byte_limit",
            ),
            (
                self.source_replay_stderr_byte_limit,
                "source_replay_stderr_byte_limit",
            ),
            (
                self.source_replay_task_request_byte_limit,
                "source_replay_task_request_byte_limit",
            ),
            (
                self.source_replay_journal_event_byte_limit,
                "source_replay_journal_event_byte_limit",
            ),
            (
                self.source_replay_result_byte_limit,
                "source_replay_result_byte_limit",
            ),
            (
                self.source_replay_staging_entry_limit,
                "source_replay_staging_entry_limit",
            ),
        ):
            _require_positive(value, f"expert.validation.policy.{name}")
        _require_non_negative(
            self.source_replay_accelerator_count,
            "expert.validation.policy.source_replay_accelerator_count",
        )
        if (self.source_replay_accelerator_class_id is None) != (
            self.source_replay_accelerator_count == 0
        ):
            raise CrossRunConfigurationError(
                "source replay accelerator class and count must be present together"
            )
        if self.source_replay_accelerator_class_id is not None:
            require_identifier(
                self.source_replay_accelerator_class_id,
                "expert.validation.policy.source_replay_accelerator_class_id",
            )
        _require_positive(
            self.source_replay_episode_limit,
            "expert.validation.policy.source_replay_episode_limit",
        )
        _require_positive(
            self.source_replay_bundle_limit,
            "expert.validation.policy.source_replay_bundle_limit",
        )
        for value, name in (
            (
                self.source_replay_context_materializer_id,
                "source_replay_context_materializer_id",
            ),
            (
                self.source_replay_context_materializer_version,
                "source_replay_context_materializer_version",
            ),
        ):
            require_identifier(value, f"expert.validation.policy.{name}")
        _require_positive(
            self.source_replay_materialization_entry_limit,
            "expert.validation.policy.source_replay_materialization_entry_limit",
        )
        _require_positive(
            self.source_replay_materialization_byte_limit,
            "expert.validation.policy.source_replay_materialization_byte_limit",
        )
        _require_positive(
            self.source_replay_materialization_timeout_seconds,
            "expert.validation.policy.source_replay_materialization_timeout_seconds",
        )
        _require_positive(
            self.source_replay_score_comparison_tolerance,
            "expert.validation.policy.source_replay_score_comparison_tolerance",
        )
        _require_positive(
            self.artifact_entry_limit,
            "expert.validation.policy.artifact_entry_limit",
        )
        _require_positive(
            self.artifact_byte_limit,
            "expert.validation.policy.artifact_byte_limit",
        )
        if (
            self.source_replay_shared_memory_byte_limit
            > self.source_replay_memory_byte_limit
            or self.artifact_entry_limit > self.source_replay_writable_entry_limit
            or self.artifact_byte_limit > self.source_replay_writable_byte_limit
            or self.source_replay_stdout_byte_limit
            > self.source_replay_writable_byte_limit
            or self.source_replay_stderr_byte_limit
            > self.source_replay_writable_byte_limit
            or self.source_replay_result_byte_limit
            > self.source_replay_writable_byte_limit
        ):
            raise CrossRunConfigurationError(
                "source replay compute limits are internally inconsistent"
            )
        if self.sealed_canary_trust_root is not None:
            require_identifier(
                self.sealed_canary_trust_root,
                "expert.validation.sealed_canary_trust_root",
            )
        configurable_stages = tuple(
            stage
            for stage in ExpertValidationStage
            if stage
            not in {
                ExpertValidationStage.AUTOMATED_REVIEW,
                ExpertValidationStage.PUBLICATION_ELIGIBILITY,
            }
        )
        evaluator_stages = tuple(evaluator.stage for evaluator in self.evaluators)
        if evaluator_stages != configurable_stages:
            raise CrossRunConfigurationError(
                "expert evaluators must cover every executable stage in order"
            )
        source_replay_evaluator = self.evaluators[
            configurable_stages.index(ExpertValidationStage.SOURCE_RUN_REPLAY)
        ]
        if (
            self.source_replay_termination_grace_seconds
            > source_replay_evaluator.timeout_seconds
        ):
            raise CrossRunConfigurationError(
                "source replay termination grace exceeds its leg timeout"
            )
        evaluator_ids = tuple(evaluator.evaluator_id for evaluator in self.evaluators)
        if len(evaluator_ids) != len(set(evaluator_ids)):
            raise CrossRunConfigurationError(
                "expert evaluator identities must be unique"
            )
        if not self.reviewers:
            raise CrossRunConfigurationError("expert reviewers must not be empty")
        reviewer_ids = tuple(reviewer.reviewer_id for reviewer in self.reviewers)
        if reviewer_ids != tuple(sorted(set(reviewer_ids))):
            raise CrossRunConfigurationError(
                "expert reviewers must be sorted and uniquely identified"
            )
        if set(reviewer_ids) & set(evaluator_ids):
            raise CrossRunConfigurationError(
                "expert evaluators and reviewers must have distinct identities"
            )
        evaluator_roles = {evaluator.evaluator_role for evaluator in self.evaluators}
        reviewer_roles = {reviewer.reviewer_role for reviewer in self.reviewers}
        if evaluator_roles & reviewer_roles:
            raise CrossRunConfigurationError(
                "expert evaluator and reviewer roles must be disjoint"
            )
        if self.promotion.required_approvals > len(self.reviewers):
            raise CrossRunConfigurationError(
                "expert approval quorum exceeds configured reviewers"
            )
        if self.promotion.required_rejections > len(self.reviewers):
            raise CrossRunConfigurationError(
                "expert rejection quorum exceeds configured reviewers"
            )

    def required_stages(
        self,
        validation_track: ExpertValidationTrack,
        configured_task_family_ids: tuple[str, ...],
        *,
        has_parent_release: bool,
    ) -> tuple[ExpertValidationStage, ...]:
        if not configured_task_family_ids or configured_task_family_ids != tuple(
            sorted(set(configured_task_family_ids))
        ):
            raise CrossRunConfigurationError(
                "configured task families must be non-empty, sorted, and unique"
            )
        if (
            not has_parent_release
            and validation_track is not ExpertValidationTrack.REPOSITORY_ARCHITECTURE
        ):
            raise CrossRunConfigurationError(
                "only repository architecture may validate without a parent release"
            )
        mechanical_stages = {
            ExpertValidationStage.CONTRACT_SCHEMA,
            ExpertValidationStage.IDENTITY_SECRETS_LICENSE_DEPENDENCY,
            ExpertValidationStage.STATIC_UNIT_SECURITY_RESOURCE,
            ExpertValidationStage.SYNTHETIC_FRESH_TASK,
            ExpertValidationStage.AUTOMATED_REVIEW,
            ExpertValidationStage.RELEASE_MATRIX,
            ExpertValidationStage.PUBLICATION_ELIGIBILITY,
        }
        if has_parent_release:
            mechanical_stages.add(ExpertValidationStage.SOURCE_RUN_REPLAY)
        if validation_track is ExpertValidationTrack.MECHANICAL_GENERAL_FIX:
            selected = mechanical_stages
        elif not has_parent_release:
            selected = mechanical_stages
        else:
            selected = set(mechanical_stages)
            selected.add(ExpertValidationStage.DEVELOPMENT_ANCHORS)
            if (
                validation_track is ExpertValidationTrack.BEHAVIORAL_CAPABILITY
                or self.architecture_requires_sealed_canary
            ):
                selected.add(ExpertValidationStage.SEALED_CANARY)
            if len(configured_task_family_ids) > 1:
                selected.add(ExpertValidationStage.CROSS_FAMILY_TRANSFER)
        return tuple(stage for stage in ExpertValidationStage if stage in selected)

    def can_validate(
        self,
        validation_track: ExpertValidationTrack,
        configured_task_family_ids: tuple[str, ...],
        *,
        has_parent_release: bool,
    ) -> bool:
        required_stages = self.required_stages(
            validation_track,
            configured_task_family_ids,
            has_parent_release=has_parent_release,
        )
        return (
            ExpertValidationStage.SEALED_CANARY not in required_stages
            or self.sealed_canary_trust_root is not None
        )

    @property
    def policy_fingerprint(self) -> str:
        return tree_or_blob_digest(canonical_json_bytes(self.to_dict()))

    def validation_policy(self) -> ExpertValidationPolicy:
        return ExpertValidationPolicy.mint(
            policy=self,
        )


@dataclass(frozen=True)
class ExpertValidationPolicy(StrictContract):
    validation_policy_id: str
    policy: ExpertValidationPolicySettings

    CONTENT_NAMESPACE = "expert-validation-policy"
    IDENTITY_FIELD = "validation_policy_id"


@dataclass(frozen=True)
class SourceReplayDockerProviderSettings(StrictContract):
    workspace_path: str
    runtime_executable_path: str
    runtime_executable_digest: str
    runtime_socket_path: str
    helper_executable_path: str
    helper_executable_digest: str
    runtime_server_version: str
    runtime_api_version: str
    runtime_host_operating_system: str
    runtime_host_architecture: str
    runtime_storage_driver: str
    runtime_cgroup_version: str
    runtime_default_runtime: str
    required_security_options: tuple[str, ...]
    container_user_id: int
    container_group_id: int
    cpu_period_microseconds: int
    command_timeout_seconds: int
    cleanup_timeout_seconds: int
    command_output_byte_limit: int
    result_archive_overhead_byte_limit: int

    def _validate(self) -> None:
        _require_relative_path(
            self.workspace_path,
            "expert.validation.source_replay_provider.workspace_path",
        )
        for value, name in (
            (self.runtime_executable_path, "runtime_executable_path"),
            (self.runtime_socket_path, "runtime_socket_path"),
            (self.helper_executable_path, "helper_executable_path"),
        ):
            _require_path(
                value,
                f"expert.validation.source_replay_provider.{name}",
            )
            if not PurePosixPath(value).is_absolute():
                raise CrossRunConfigurationError(
                    f"expert.validation.source_replay_provider.{name} must be absolute"
                )
        for value, name in (
            (self.runtime_executable_digest, "runtime_executable_digest"),
            (self.helper_executable_digest, "helper_executable_digest"),
        ):
            if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
                raise CrossRunConfigurationError(
                    f"expert.validation.source_replay_provider.{name} must be a sha256 digest"
                )
        for value, name in (
            (self.runtime_server_version, "runtime_server_version"),
            (self.runtime_api_version, "runtime_api_version"),
            (
                self.runtime_host_operating_system,
                "runtime_host_operating_system",
            ),
            (self.runtime_host_architecture, "runtime_host_architecture"),
            (self.runtime_storage_driver, "runtime_storage_driver"),
            (self.runtime_cgroup_version, "runtime_cgroup_version"),
            (self.runtime_default_runtime, "runtime_default_runtime"),
        ):
            require_identifier(
                value,
                f"expert.validation.source_replay_provider.{name}",
            )
        if (
            not self.required_security_options
            or self.required_security_options
            != tuple(sorted(set(self.required_security_options)))
            or any(
                not isinstance(option, str)
                or re.fullmatch(r"[a-z0-9=,._-]+", option) is None
                for option in self.required_security_options
            )
        ):
            raise CrossRunConfigurationError(
                "source replay provider security options must be sorted and unique"
            )
        for value, name in (
            (self.container_user_id, "container_user_id"),
            (self.container_group_id, "container_group_id"),
        ):
            if type(value) is not int or value < 0:
                raise CrossRunConfigurationError(
                    f"expert.validation.source_replay_provider.{name} must be a non-negative integer"
                )
        for value, name in (
            (self.cpu_period_microseconds, "cpu_period_microseconds"),
            (self.command_timeout_seconds, "command_timeout_seconds"),
            (self.cleanup_timeout_seconds, "cleanup_timeout_seconds"),
            (self.command_output_byte_limit, "command_output_byte_limit"),
            (
                self.result_archive_overhead_byte_limit,
                "result_archive_overhead_byte_limit",
            ),
        ):
            if type(value) is not int or value <= 0:
                raise CrossRunConfigurationError(
                    f"expert.validation.source_replay_provider.{name} must be a positive integer"
                )
        if self.cleanup_timeout_seconds > self.command_timeout_seconds:
            raise CrossRunConfigurationError(
                "source replay provider cleanup timeout exceeds its command timeout"
            )


@dataclass(frozen=True)
class ExpertValidationSettings(StrictContract):
    state_path: str
    source_replay_provider: SourceReplayDockerProviderSettings
    policy: ExpertValidationPolicySettings

    def _validate(self) -> None:
        _require_relative_path(self.state_path, "expert.validation.state_path")
        if (
            self.policy.source_replay_cpu_millicore_limit
            * self.source_replay_provider.cpu_period_microseconds
            % 1000
            != 0
        ):
            raise CrossRunConfigurationError(
                "source replay millicore limit has no exact runtime quota"
            )

    @property
    def configuration_fingerprint(self) -> str:
        return tree_or_blob_digest(canonical_json_bytes(self.to_dict()))


@dataclass(frozen=True)
class TaskAdapterAuthorityTrustSettings(StrictContract):
    authority_id: str
    authority_version: str

    def _validate(self) -> None:
        require_identifier(self.authority_id, "task adapter authority_id")
        require_identifier(self.authority_version, "task adapter authority_version")

    @property
    def identity(self) -> tuple[str, str]:
        return self.authority_id, self.authority_version


@dataclass(frozen=True)
class TaskAdapterStoreSettings(StrictContract):
    state_path: str
    active_authority: TaskAdapterAuthorityTrustSettings
    trusted_authorities: tuple[TaskAdapterAuthorityTrustSettings, ...]
    package_entry_limit: int
    package_byte_limit: int
    source_byte_limit: int
    zstd_window_size_bytes: int

    def _validate(self) -> None:
        _require_relative_path(self.state_path, "expert.task_adapters.state_path")
        authority_identities = tuple(
            authority.identity for authority in self.trusted_authorities
        )
        if not authority_identities or authority_identities != tuple(
            sorted(set(authority_identities))
        ):
            raise CrossRunConfigurationError(
                "trusted task adapter authorities must be non-empty, sorted, and unique"
            )
        if self.active_authority.identity not in authority_identities:
            raise CrossRunConfigurationError(
                "active task adapter authority must be trusted"
            )
        for value, name in (
            (self.package_entry_limit, "package_entry_limit"),
            (self.package_byte_limit, "package_byte_limit"),
            (self.source_byte_limit, "source_byte_limit"),
            (self.zstd_window_size_bytes, "zstd_window_size_bytes"),
        ):
            _require_positive(value, f"expert.task_adapters.{name}")


@dataclass(frozen=True)
class ExpertSettings(StrictContract):
    workspace_path: str
    candidate_path: str
    agent_artifact_path: str
    candidate_entry_limit: int
    candidate_byte_limit: int
    agent_artifact_byte_limit: int
    termination_grace_seconds: int
    sensitive_file_glob_scan_max_depth: int
    architect_id: str
    architect_role: str
    architect: CodingAgentSettings
    generalizer_id: str
    generalizer_role: str
    generalizer: CodingAgentSettings
    triggers: ExpertTriggerSettings
    task_adapters: TaskAdapterStoreSettings
    validation: ExpertValidationSettings

    def _validate(self) -> None:
        workspace_path = _require_relative_path(
            self.workspace_path,
            "expert.workspace_path",
        )
        candidate_path = _require_relative_path(
            self.candidate_path,
            "expert.candidate_path",
        )
        artifact_path = _require_relative_path(
            self.agent_artifact_path,
            "expert.agent_artifact_path",
        )
        validation_path = _require_relative_path(
            self.validation.state_path,
            "expert.validation.state_path",
        )
        source_replay_provider_path = _require_relative_path(
            self.validation.source_replay_provider.workspace_path,
            "expert.validation.source_replay_provider.workspace_path",
        )
        task_adapter_path = _require_relative_path(
            self.task_adapters.state_path,
            "expert.task_adapters.state_path",
        )
        paths = {
            "workspaces": workspace_path,
            "candidates": candidate_path,
            "agent artifacts": artifact_path,
            "validation": validation_path,
            "source replay provider": source_replay_provider_path,
            "task adapters": task_adapter_path,
        }
        for name, path in paths.items():
            for other_name, other_path in paths.items():
                if name < other_name and (
                    path == other_path
                    or path in other_path.parents
                    or other_path in path.parents
                ):
                    raise CrossRunConfigurationError(
                        f"expert {name} and {other_name} must be disjoint"
                    )
        _require_positive(
            self.candidate_entry_limit,
            "expert.candidate_entry_limit",
        )
        _require_positive(
            self.candidate_byte_limit,
            "expert.candidate_byte_limit",
        )
        _require_positive(
            self.agent_artifact_byte_limit,
            "expert.agent_artifact_byte_limit",
        )
        _require_positive(
            self.termination_grace_seconds,
            "expert.termination_grace_seconds",
        )
        _require_positive(
            self.sensitive_file_glob_scan_max_depth,
            "expert.sensitive_file_glob_scan_max_depth",
        )
        for value, name in (
            (self.architect_id, "expert.architect_id"),
            (self.architect_role, "expert.architect_role"),
            (self.generalizer_id, "expert.generalizer_id"),
            (self.generalizer_role, "expert.generalizer_role"),
        ):
            require_identifier(value, name)
        if self.architect_id == self.generalizer_id:
            raise CrossRunConfigurationError(
                "expert architect and generalizer identities must differ"
            )
        proposal_authorities = {
            self.architect_id,
            self.generalizer_id,
            self.triggers.inspector_id,
        }
        validation_authorities = {
            *(
                evaluator.evaluator_id
                for evaluator in self.validation.policy.evaluators
            ),
            *(reviewer.reviewer_id for reviewer in self.validation.policy.reviewers),
        }
        if proposal_authorities & validation_authorities:
            raise CrossRunConfigurationError(
                "expert proposal and validation authorities must be disjoint"
            )
        proposal_roles = {
            self.architect_role,
            self.generalizer_role,
            self.triggers.inspector_role,
        }
        validation_roles = {
            *(
                evaluator.evaluator_role
                for evaluator in self.validation.policy.evaluators
            ),
            *(reviewer.reviewer_role for reviewer in self.validation.policy.reviewers),
        }
        if proposal_roles & validation_roles:
            raise CrossRunConfigurationError(
                "expert proposal and validation roles must be disjoint"
            )
        self._validate_candidate_editor(self.architect, "architect")
        self._validate_candidate_editor(self.generalizer, "generalizer")

    @staticmethod
    def _validate_candidate_editor(
        agent: CodingAgentSettings,
        name: str,
    ) -> None:
        if "WebSearch" in agent.allowed_tools:
            raise CrossRunConfigurationError(
                f"expert {name} cannot search outside its persisted packet"
            )
        required = {"Read"} if agent.cli == "codex" else {"Edit", "Read", "Write"}
        if not required.issubset(agent.allowed_tools):
            raise CrossRunConfigurationError(
                f"expert {name} lacks required candidate-workspace tools"
            )


@dataclass(frozen=True)
class LaunchSettings(StrictContract):
    cache_path: str
    bootstrap_pin_path: str
    security_denylist_state_path: str
    security_denylist_checkpoint_size_bytes: int
    security_denylist_checked_subject_limit: int
    security_denylist_checked_subject_size_bytes: int
    artifact_ttl_seconds: int
    denylist_refresh_seconds: int
    security_denylist_revocation_limit: int
    security_denylist_lineage_limit: int

    def _validate(self) -> None:
        _require_path(self.cache_path, "launch.cache_path")
        _require_path(self.bootstrap_pin_path, "launch.bootstrap_pin_path")
        _require_path(
            self.security_denylist_state_path,
            "launch.security_denylist_state_path",
        )
        _require_positive(
            self.security_denylist_checkpoint_size_bytes,
            "launch.security_denylist_checkpoint_size_bytes",
        )
        _require_positive(
            self.security_denylist_checked_subject_limit,
            "launch.security_denylist_checked_subject_limit",
        )
        _require_positive(
            self.security_denylist_checked_subject_size_bytes,
            "launch.security_denylist_checked_subject_size_bytes",
        )
        _require_positive(self.artifact_ttl_seconds, "launch.artifact_ttl_seconds")
        _require_positive(
            self.denylist_refresh_seconds, "launch.denylist_refresh_seconds"
        )
        _require_positive(
            self.security_denylist_revocation_limit,
            "launch.security_denylist_revocation_limit",
        )
        _require_positive(
            self.security_denylist_lineage_limit,
            "launch.security_denylist_lineage_limit",
        )


@dataclass(frozen=True)
class ProductionValidationSettings(StrictContract):
    fixture_path: str
    github_write_smoke: bool
    embedding_smoke: bool
    coding_agent_smoke: bool
    task_smoke_timeout_seconds: int

    def _validate(self) -> None:
        _require_path(self.fixture_path, "production_validation.fixture_path")
        _require_positive(
            self.task_smoke_timeout_seconds,
            "production_validation.task_smoke_timeout_seconds",
        )


@dataclass(frozen=True)
class CrossRunSettings(StrictContract):
    scopes: ScopeRegistrySettings
    github: GitHubSettings
    capture: CaptureSettings
    sanitation: SanitationSettings
    catalog: CatalogSettings
    knowledge: KnowledgeSettings
    expert: ExpertSettings
    launch: LaunchSettings
    production_validation: ProductionValidationSettings

    def _validate(self) -> None:
        if self.capture.git_command_output_bytes < self.sanitation.max_file_bytes:
            raise CrossRunConfigurationError(
                "capture Git output limit must admit one allowlisted source file"
            )
        if (
            self.capture.score_comparison_tolerance
            != self.expert.validation.policy.source_replay_score_comparison_tolerance
        ):
            raise CrossRunConfigurationError(
                "capture and source replay projection tolerances must match"
            )
        if self.expert.validation.policy.source_replay_journal_event_byte_limit < 2 * (
            self.launch.security_denylist_checked_subject_size_bytes
            + self.expert.validation.policy.source_replay_task_request_byte_limit
        ):
            raise CrossRunConfigurationError(
                "source replay journal event bound cannot contain its denylist and "
                "task-request authorities"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CrossRunSettings:
        if not isinstance(payload, Mapping):
            raise CrossRunConfigurationError("cross_run must be an object")
        _reject_secret_keys(payload)
        expected = {
            "scopes",
            "github",
            "capture",
            "sanitation",
            "catalog",
            "knowledge",
            "expert",
            "launch",
            "production_validation",
        }
        missing = tuple(sorted(expected - set(payload)))
        unknown = tuple(sorted(set(payload) - expected))
        if missing or unknown:
            raise CrossRunConfigurationError(
                f"cross_run fields mismatch; missing={missing}, unknown={unknown}"
            )
        return cls(
            scopes=ScopeRegistrySettings.from_config(payload["scopes"]),
            github=payload["github"],
            capture=payload["capture"],
            sanitation=payload["sanitation"],
            catalog=payload["catalog"],
            knowledge=payload["knowledge"],
            expert=payload["expert"],
            launch=payload["launch"],
            production_validation=payload["production_validation"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scopes": self.scopes.to_config(),
            "github": to_json_value(self.github),
            "capture": to_json_value(self.capture),
            "sanitation": to_json_value(self.sanitation),
            "catalog": to_json_value(self.catalog),
            "knowledge": to_json_value(self.knowledge),
            "expert": to_json_value(self.expert),
            "launch": to_json_value(self.launch),
            "production_validation": to_json_value(self.production_validation),
        }

    @property
    def configuration_fingerprint(self) -> str:
        return tree_or_blob_digest(canonical_json_bytes(self.to_dict()))

    def resolve_binding(
        self,
        binding: CrossRunTaskBindingSettings,
        scope_contract: ExpertScopeContract,
    ) -> ScopeRepositorySettings:
        scope_contract.validate_binding(binding)
        return self.scopes.resolve(binding.scope_id)


@dataclass(frozen=True)
class EffectiveConfig:
    mode_name: str
    mode: Mapping[str, Any]
    cross_run: CrossRunSettings | None
    registry_source_fingerprint: str | None

    def __post_init__(self) -> None:
        if not self.mode_name:
            raise CrossRunConfigurationError("mode_name must not be empty")
        object.__setattr__(self, "mode", MappingProxyType(dict(self.mode)))
        forbidden_mode_fields = {
            "cross_run",
            "cross_run_registry_fingerprint",
        } & set(self.mode)
        if forbidden_mode_fields:
            raise CrossRunConfigurationError(
                "mode cannot override global cross-run configuration: "
                f"{tuple(sorted(forbidden_mode_fields))}"
            )
        if self.cross_run is None and self.registry_source_fingerprint is not None:
            raise CrossRunConfigurationError(
                "registry fingerprint cannot exist without cross_run settings"
            )
        if self.cross_run is not None:
            if self.registry_source_fingerprint != self.cross_run.scopes.fingerprint:
                raise CrossRunConfigurationError(
                    "runtime scope registry fingerprint mismatch"
                )


def compose_runtime_config(
    canonical_config: Mapping[str, Any],
    workload_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose a self-contained runtime config with one pinned scope registry."""
    if "cross_run" not in canonical_config:
        raise CrossRunConfigurationError("canonical config has no cross_run tree")
    if (
        "cross_run" in workload_config
        or "cross_run_registry_fingerprint" in workload_config
    ):
        raise CrossRunConfigurationError(
            "workload config cannot override the canonical cross_run tree"
        )
    settings = CrossRunSettings.from_dict(canonical_config["cross_run"])
    repository_coordinates = {
        repository
        for scope in settings.scopes.scopes
        for repository in (
            scope.expert_repository,
            scope.knowledge_repository,
            scope.security_repository,
        )
    }

    def reject_repository_copy(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key == "cross_run_binding":
                    binding = CrossRunTaskBindingSettings.from_dict(child)
                    settings.scopes.resolve(binding.scope_id)
                if key in {
                    "cross_run",
                    "cross_run_registry_fingerprint",
                    "expert_repository",
                    "knowledge_repository",
                    "security_repository",
                    "repositories",
                }:
                    raise CrossRunConfigurationError(
                        f"workload config cannot declare repository routing at {path}.{key}"
                    )
                reject_repository_copy(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for position, child in enumerate(value):
                reject_repository_copy(child, f"{path}[{position}]")
        elif value in repository_coordinates:
            raise CrossRunConfigurationError(
                f"workload config duplicates a repository coordinate at {path}"
            )

    reject_repository_copy(workload_config, "workload")
    runtime = to_json_value(workload_config)
    runtime["cross_run"] = settings.to_dict()
    runtime["cross_run_registry_fingerprint"] = settings.scopes.fingerprint
    return runtime


def validate_runtime_registry(
    runtime_config: Mapping[str, Any], canonical_settings: CrossRunSettings
) -> None:
    required = {"cross_run", "cross_run_registry_fingerprint"}
    missing = tuple(sorted(required - set(runtime_config)))
    if missing:
        raise CrossRunConfigurationError(
            f"runtime config is missing copied scope registry fields: {missing}"
        )
    runtime_settings = CrossRunSettings.from_dict(runtime_config["cross_run"])
    declared = runtime_config["cross_run_registry_fingerprint"]
    if declared != canonical_settings.scopes.fingerprint:
        raise CrossRunConfigurationError("runtime registry source fingerprint is stale")
    if runtime_settings.scopes.to_config() != canonical_settings.scopes.to_config():
        raise CrossRunConfigurationError(
            "runtime registry differs from canonical source"
        )
