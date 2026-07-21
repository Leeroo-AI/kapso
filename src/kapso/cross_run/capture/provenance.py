"""Cross-authority provenance invariants for captured execution revisions."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Sequence

from kapso.cross_run.canonical import to_json_value
from kapso.cross_run.capture.journal import ExecutionRevisionEvent
from kapso.execution.memories.experiment_memory.store import ExperimentRecord
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.ideation.archive import IdeaArchiveState

_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def validate_execution_provenance(
    archive: IdeaArchiveState,
    records: Sequence[ExperimentRecord],
    events: Sequence[ExecutionRevisionEvent],
    nodes: Sequence[SearchNode],
    error_type: type[Exception],
) -> None:
    """Prove that ideas, revisions, nodes, and Git refs describe one lineage."""
    if len(records) != len(nodes):
        raise error_type("execution provenance node frontier differs from history")
    if tuple(record.node_id for record in records) != tuple(range(len(records))):
        raise error_type("execution provenance record frontier is not contiguous")
    if tuple(node.node_id for node in nodes) != tuple(range(len(nodes))):
        raise error_type("execution provenance checkpoint frontier is not contiguous")

    ideas = {idea.idea_id: idea for idea in archive.ideas}
    events_by_node: dict[int, list[ExecutionRevisionEvent]] = defaultdict(list)
    for event in events:
        events_by_node[event.node_id].append(event)
    terminal_candidate_by_node = {
        node_id: node_events[-1].artifact_refs.get("candidate_commit", "")
        for node_id, node_events in events_by_node.items()
    }

    for record, node in zip(records, nodes):
        if record.idea_id is None or record.idea_id not in ideas:
            raise error_type("execution provenance references an unknown idea")
        idea = ideas[record.idea_id]
        parent = idea.resolved_parent
        if _GIT_COMMIT_PATTERN.fullmatch(parent.git_ref) is None:
            raise error_type(
                "execution provenance resolved parent is not an immutable commit"
            )
        if (
            parent.materialized_ref != parent.git_ref
            or parent.diff_base_ref != parent.git_ref
            or parent.feedback_base_ref != parent.git_ref
        ):
            raise error_type("execution provenance resolved parent refs diverge")
        if (
            idea.experiment_node_id != record.node_id
            or idea.selected_in_batch_id != record.selection_batch_id
            or record.parent_node_id != parent.node_id
            or record.solution != idea.proposal
            or node.idea_id != idea.idea_id
            or node.selection_batch_id != idea.selected_in_batch_id
            or node.parent_node_id != parent.node_id
            or node.solution != idea.proposal
        ):
            raise error_type("execution provenance idea projection changed")
        if parent.node_id is not None:
            if parent.node_id >= record.node_id:
                raise error_type("execution provenance parent is not an earlier node")
            parent_record = records[parent.node_id]
            if (
                parent.branch_name != parent_record.branch_name
                or terminal_candidate_by_node.get(parent.node_id) != parent.git_ref
            ):
                raise error_type(
                    "execution provenance parent snapshot is not the parent frontier"
                )

        node_events = events_by_node.get(record.node_id, [])
        if tuple(event.execution_revision for event in node_events) != tuple(
            range(record.execution_revision + 1)
        ):
            raise error_type("execution provenance revisions are not an exact prefix")
        prior_candidate = ""
        for event in node_events:
            revision_record = ExperimentRecord.from_dict(
                to_json_value(event.projection)
            )
            if (
                revision_record.idea_id != idea.idea_id
                or revision_record.selection_batch_id != idea.selected_in_batch_id
                or revision_record.parent_node_id != parent.node_id
                or revision_record.solution != idea.proposal
                or revision_record.branch_name != record.branch_name
            ):
                raise error_type("execution revision changed its idea provenance")

            implementation_base = (
                parent.git_ref if event.execution_revision == 0 else prior_candidate
            )
            if not implementation_base:
                raise error_type(
                    "recovery revision has no previous immutable candidate"
                )
            expected_refs = {
                "branch": revision_record.branch_name,
                "parent_branch": parent.branch_name,
                "implementation_base": implementation_base,
                "diff_base": parent.diff_base_ref,
                "feedback_base": parent.feedback_base_ref,
            }
            candidate = event.artifact_refs.get("candidate_commit", "")
            if candidate:
                if _GIT_COMMIT_PATTERN.fullmatch(candidate) is None:
                    raise error_type("execution provenance candidate commit is invalid")
                expected_refs.update(
                    {
                        "candidate_commit": candidate,
                        "candidate_ref": (
                            f"refs/kapso/execution-revisions/{event.run_id}/"
                            f"node-{event.node_id}/"
                            f"revision-{event.execution_revision}"
                        ),
                        "implementation_base_commit": implementation_base,
                        "diff_base_commit": parent.diff_base_ref,
                        "feedback_base_commit": parent.feedback_base_ref,
                    }
                )
            elif event.execution_revision > 0:
                raise error_type("recovery revision has no immutable candidate commit")

            for position, attempt in enumerate(revision_record.evaluation_attempts):
                if not candidate or attempt.commit_sha != candidate:
                    raise error_type(
                        "evaluation evidence is not bound to the revision candidate"
                    )
                expected_refs[f"evaluation_commit_{position}"] = candidate
            if dict(event.artifact_refs) != expected_refs:
                raise error_type("execution revision artifact provenance is not exact")
            prior_candidate = candidate

        latest_event = node_events[-1]
        if (
            node.execution_revision != latest_event.execution_revision
            or node.branch_name != latest_event.artifact_refs["branch"]
            or node.parent_branch_name != latest_event.artifact_refs["parent_branch"]
            or node.implementation_base_ref
            != latest_event.artifact_refs["implementation_base"]
            or node.diff_base_ref != latest_event.artifact_refs["diff_base"]
            or node.feedback_base_ref != latest_event.artifact_refs["feedback_base"]
        ):
            raise error_type(
                "checkpoint node provenance differs from its terminal revision"
            )
