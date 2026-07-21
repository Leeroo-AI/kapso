"""Shared semantic validation for immutable RunBundle lineage edges."""

from __future__ import annotations

from kapso.cross_run.contracts import RunBundle

RUN_BUNDLE_STABLE_IDENTITY_FIELDS = (
    "scope_contract_id",
    "scope_id",
    "run_id",
    "campaign_id",
    "started_at",
    "kapso_commit",
    "launch_manifest_id",
    "knowledge_snapshot_id",
    "expert_base_release_id",
    "task_context_binding",
    "artifact_environment",
)


def validate_run_bundle_root(
    bundle: RunBundle,
    error_type: type[ValueError],
) -> None:
    """Require the exporter-defined origin of one complete run lineage."""

    if bundle.supersedes_bundle_id is not None or bundle.capture_generation != 0:
        raise error_type("bundle lineage root is not generation zero")


def validate_run_bundle_successor(
    parent: RunBundle,
    child: RunBundle,
    error_type: type[ValueError],
) -> None:
    """Require one exact, monotonic adjacent supersession edge."""

    if child.supersedes_bundle_id != parent.bundle_id:
        raise error_type("bundle predecessor identity changed")
    if child.capture_generation != parent.capture_generation + 1:
        raise error_type("bundle capture generations are not contiguous")
    if child.checkpoint_frontier < parent.checkpoint_frontier:
        raise error_type("bundle checkpoint frontier moved backwards")
    if set(child.capture_watermarks) != set(parent.capture_watermarks) or any(
        child.capture_watermarks[name] < parent.capture_watermarks[name]
        for name in parent.capture_watermarks
    ):
        raise error_type("bundle capture watermarks moved backwards")
    if any(
        getattr(child, name) != getattr(parent, name)
        for name in RUN_BUNDLE_STABLE_IDENTITY_FIELDS
    ):
        raise error_type("bundle supersession changed stable run identity")
