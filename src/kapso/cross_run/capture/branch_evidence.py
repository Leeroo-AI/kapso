"""Pure verification of one captured Git branch evidence closure."""

from __future__ import annotations

import re
from collections.abc import Callable

from kapso.cross_run.canonical import source_tree_digest, tree_or_blob_digest
from kapso.cross_run.capture.exporter import (
    BRANCH_SNAPSHOT_SCHEMA,
    BranchSnapshot,
    CaptureDescriptor,
)
from kapso.cross_run.capture.git_evidence import (
    has_ancestry_path,
    parse_commit_object,
    reconstruct_root_tree_sha,
)
from kapso.cross_run.git_refs import git_object_sha
from kapso.cross_run.record_contracts import ExecutionRevisionEvent
from kapso.execution.memories.experiment_memory.record import ExperimentRecord


def validate_branch_evidence(
    *,
    read_ref: Callable[[str], bytes],
    descriptor: CaptureDescriptor,
    record: ExperimentRecord,
    event: ExecutionRevisionEvent,
    branch: BranchSnapshot,
    error_type: type[ValueError],
) -> set[str]:
    """Verify identity, ancestry, source bytes, and exact Git tree closure."""

    expected_revision_ref = (
        f"refs/kapso/execution-revisions/{event.run_id}/"
        f"node-{event.node_id}/revision-{event.execution_revision}"
    )
    if (
        branch.schema != BRANCH_SNAPSHOT_SCHEMA
        or branch.branch_name != record.branch_name
        or branch.parent_branch_name != event.artifact_refs.get("parent_branch", "")
        or branch.revision_ref != expected_revision_ref
        or branch.revision_ref != event.artifact_refs.get("candidate_ref")
        or branch.commit_sha != event.artifact_refs.get("candidate_commit")
        or branch.implementation_base_ref
        != event.artifact_refs.get("implementation_base", "")
        or branch.diff_base_ref != event.artifact_refs.get("diff_base", "")
        or branch.feedback_base_ref != event.artifact_refs.get("feedback_base", "")
        or dict(branch.base_commit_shas)
        != {
            name: event.artifact_refs[f"{name}_base_commit"]
            for name in ("implementation", "diff", "feedback")
            if f"{name}_base_commit" in event.artifact_refs
        }
    ):
        raise error_type("branch snapshot identity changed")
    expected_event_refs = {
        name: value
        for name, value in {
            "branch": record.branch_name,
            "parent_branch": branch.parent_branch_name,
            "implementation_base": branch.implementation_base_ref,
            "diff_base": branch.diff_base_ref,
            "feedback_base": branch.feedback_base_ref,
            "candidate_commit": branch.commit_sha,
            "candidate_ref": branch.revision_ref,
        }.items()
        if value
    }
    for position, attempt in enumerate(record.evaluation_attempts):
        expected_event_refs[f"evaluation_commit_{position}"] = attempt.commit_sha
    for name, commit in branch.base_commit_shas.items():
        expected_event_refs[f"{name}_base_commit"] = commit
    if dict(event.artifact_refs) != expected_event_refs:
        raise error_type("journal branch provenance closure changed")
    evaluated = tuple(sorted({item.commit_sha for item in record.evaluation_attempts}))
    if branch.evaluated_commit_shas != evaluated:
        raise error_type("branch evaluation commit closure changed")
    commit_graph: dict[str, tuple[str, ...]] = {}
    commit_tree_shas: dict[str, str] = {}
    evidence_refs: set[str] = set()
    for item in branch.commit_objects:
        payload_ref = item["payload_ref"]
        if (
            payload_ref in evidence_refs
            or payload_ref not in descriptor.artifact_refs.values()
        ):
            raise error_type("Git commit payload ref is invalid")
        payload = read_ref(payload_ref)
        commit_sha = item["commit_sha"]
        if git_object_sha("commit", payload) != commit_sha:
            raise error_type("Git commit payload identity changed")
        parsed = parse_commit_object(payload)
        commit_graph[commit_sha] = parsed.parent_shas
        commit_tree_shas[commit_sha] = parsed.tree_sha
        evidence_refs.add(payload_ref)
    if (
        branch.commit_sha not in commit_graph
        or commit_tree_shas[branch.commit_sha] != branch.root_tree_sha
    ):
        raise error_type("candidate commit/tree proof changed")
    for base_sha in branch.base_commit_shas.values():
        if base_sha not in commit_graph or not has_ancestry_path(
            commit_graph, branch.commit_sha, base_sha
        ):
            raise error_type("branch base ancestry is not proven")
    for proof_sha in commit_graph:
        belongs_to_proof = proof_sha == branch.commit_sha or any(
            has_ancestry_path(commit_graph, branch.commit_sha, proof_sha)
            and has_ancestry_path(commit_graph, proof_sha, base_sha)
            for base_sha in branch.base_commit_shas.values()
        )
        if not belongs_to_proof:
            raise error_type("Git ancestry proof has unrelated commits")
    tree: dict[str, tuple[str, str, int]] = {}
    git_tree_entries: dict[str, tuple[str, str]] = {}
    seen_refs: set[str] = set()
    for item in branch.source_files:
        required = {
            "git_blob_sha",
            "mode",
            "payload_ref",
            "sha256",
            "size",
            "source_path",
        }
        if set(item) != required:
            raise error_type("branch source descriptor is invalid")
        payload_ref = item["payload_ref"]
        if item["mode"] not in {"100644", "100755"}:
            raise error_type("branch source mode is not a regular file")
        if (
            payload_ref in seen_refs
            or payload_ref not in descriptor.artifact_refs.values()
        ):
            raise error_type("branch source payload ref is invalid")
        payload = read_ref(payload_ref)
        if (
            len(payload) != item["size"]
            or tree_or_blob_digest(payload) != item["sha256"]
            or git_object_sha("blob", payload) != item["git_blob_sha"]
        ):
            raise error_type("branch source payload identity changed")
        source_path = item["source_path"]
        if source_path in tree:
            raise error_type("branch source path is duplicated")
        tree[source_path] = (item["sha256"], item["mode"], item["size"])
        git_tree_entries[source_path] = (item["mode"], item["git_blob_sha"])
        seen_refs.add(payload_ref)
        evidence_refs.add(payload_ref)
    exclusion_fields = {
        "git_object_sha",
        "mode",
        "object_type",
        "path",
        "reason",
        "size",
    }
    valid_exclusion_reasons = {
        "artifact_class",
        "denied_path",
        "file_too_large",
        "non_regular_file",
        "unknown_size",
    }
    for item in branch.excluded_files:
        if set(item) != exclusion_fields:
            raise error_type("branch exclusion descriptor is invalid")
        mode = item["mode"]
        object_type = item["object_type"]
        size = item["size"]
        if (
            item["reason"] not in valid_exclusion_reasons
            or re.fullmatch(r"[0-9a-f]{40}", item["git_object_sha"]) is None
            or (mode, object_type)
            not in {
                ("100644", "blob"),
                ("100755", "blob"),
                ("120000", "blob"),
                ("160000", "commit"),
            }
            or (
                (object_type == "blob" and (type(size) is not int or size < 0))
                or (object_type == "commit" and size is not None)
            )
        ):
            raise error_type("branch exclusion identity is invalid")
        git_tree_entries[item["path"]] = (mode, item["git_object_sha"])
    if len(git_tree_entries) != len(branch.source_files) + len(branch.excluded_files):
        raise error_type("branch Git tree paths are not unique")
    if reconstruct_root_tree_sha(git_tree_entries) != branch.root_tree_sha:
        raise error_type("branch Git root tree proof changed")
    expected_tree_digest = (
        source_tree_digest(tree) if tree else tree_or_blob_digest(b"[]")
    )
    if expected_tree_digest != branch.source_tree_digest:
        raise error_type("branch source tree digest changed")
    return evidence_refs
