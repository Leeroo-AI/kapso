"""Deterministic projection from one sanitized RunBundle frontier."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    parse_json_bytes,
    require_content_id,
    to_json_value,
    tree_or_blob_digest,
)
from kapso.cross_run.capture.branch_evidence import validate_branch_evidence
from kapso.cross_run.capture.bundle import RunBundleReader
from kapso.cross_run.capture.bundle_lineage import (
    validate_run_bundle_root,
    validate_run_bundle_successor,
)
from kapso.cross_run.capture.evaluation_evidence import evaluation_scores_match
from kapso.cross_run.capture.exporter import BranchSnapshot, CaptureDescriptor
from kapso.cross_run.contracts import (
    ArtifactCompleteness,
    BundleArtifactRef,
    ComparisonStatus,
    CompletionState,
    EffectUncertaintyMethod,
    EpisodeEvaluationStatus,
    EvaluationFingerprint,
    ExecutionStatus,
    InterventionStructure,
    ObjectiveDirection,
    PriorIdea,
    PriorIdeaStatus,
    RelativeEffect,
    RunBundle,
    StrictContract,
    TransferAttempt,
    TransferEpisode,
)
from kapso.cross_run.record_contracts import (
    BundleProjectionError,
    BundleProjectionManifest,
    ExecutionRevisionEvent,
    SanitationReport,
)
from kapso.execution.memories.experiment_memory.record import (
    EXPERIMENT_HISTORY_SCHEMA,
    ExperimentRecord,
)
from kapso.execution.run_checkpoint import RunCheckpoint
from kapso.execution.search_strategies.node import SearchNode
from kapso.execution.search_strategies.generic.strategy import (
    GENERIC_SEARCH_STATE_FIELDS,
    GENERIC_SEARCH_STATE_SCHEMA,
)
from kapso.execution.search_strategies.generic.ideation.archive import (
    IdeaArchiveState,
)
from kapso.execution.search_strategies.generic.ideation.types import (
    IdeaBatch,
    IdeaRecord,
    IdeaStatus,
)


@dataclass(frozen=True)
class ProjectionResult:
    """Complete, disjoint projection of one latest admitted run frontier."""

    source_bundle: RunBundle
    sanitation_report: SanitationReport
    episodes: tuple[TransferEpisode, ...]
    prior_ideas: tuple[PriorIdea, ...]
    derivation_objects: tuple[ExecutionRevisionEvent, ...]

    def __post_init__(self) -> None:
        episode_sources = tuple(_episode_source_key(item) for item in self.episodes)
        prior_sources = tuple(_prior_source_key(item) for item in self.prior_ideas)
        if episode_sources != tuple(sorted(set(episode_sources))):
            raise BundleProjectionError("episode sources must be sorted and unique")
        if prior_sources != tuple(sorted(set(prior_sources))):
            raise BundleProjectionError("prior-idea sources must be sorted and unique")
        if set(episode_sources) & set(prior_sources):
            raise BundleProjectionError("projection source sets must be disjoint")
        if any(
            item.source_bundle_id != self.source_bundle.bundle_id
            for item in (*self.episodes, *self.prior_ideas)
        ):
            raise BundleProjectionError("projection names another source bundle")
        if (
            self.sanitation_report.status != "admitted"
            or self.sanitation_report.scope_id != self.source_bundle.scope_id
            or self.sanitation_report.task_family_id
            != self.source_bundle.task_context_binding.task_family_id
            or any(
                item.sanitation_report_id != self.sanitation_report.report_id
                for item in (*self.episodes, *self.prior_ideas)
            )
        ):
            raise BundleProjectionError(
                "projection subjects do not bind the admitted sanitation report"
            )
        derivation_ids = tuple(item.event_id for item in self.derivation_objects)
        if derivation_ids != tuple(sorted(set(derivation_ids))):
            raise BundleProjectionError(
                "projection derivation objects must be sorted and unique"
            )
        referenced_derivations = {
            reference
            for episode in self.episodes
            for reference in episode.derivation_refs
            if reference != self.source_bundle.bundle_id
        }
        if referenced_derivations != set(derivation_ids):
            raise BundleProjectionError(
                "projection derivation object closure is not exact"
            )

    @property
    def source_projection_ids(self) -> dict[tuple[str, str, str, str], str]:
        return {
            **{_episode_source_key(item): item.episode_id for item in self.episodes},
            **{
                _prior_source_key(item): item.prior_idea_id for item in self.prior_ideas
            },
        }

    @property
    def projection_manifest(self) -> BundleProjectionManifest:
        return BundleProjectionManifest.mint(
            source_bundle_id=self.source_bundle.bundle_id,
            sanitation_report_id=self.sanitation_report.report_id,
            episode_ids=tuple(sorted(item.episode_id for item in self.episodes)),
            prior_idea_ids=tuple(
                sorted(item.prior_idea_id for item in self.prior_ideas)
            ),
            derivation_object_ids=tuple(
                sorted(item.event_id for item in self.derivation_objects)
            ),
        )

    @property
    def catalog_facts(self) -> tuple[StrictContract, ...]:
        return (
            self.source_bundle,
            self.sanitation_report,
            *self.derivation_objects,
            *self.episodes,
            *self.prior_ideas,
            self.projection_manifest,
        )


@dataclass(frozen=True)
class _EvaluationGroup:
    fingerprint: EvaluationFingerprint
    aggregate: float


@dataclass(frozen=True)
class _BundleAuthorities:
    descriptor: CaptureDescriptor
    sanitation_report: SanitationReport
    archive: IdeaArchiveState
    history_records: tuple[ExperimentRecord, ...]
    events: tuple[ExecutionRevisionEvent, ...]


class RunBundleProjector:
    """Mechanically project immutable bundle evidence without interpretation."""

    def __init__(self, score_comparison_tolerance: float):
        if (
            isinstance(score_comparison_tolerance, bool)
            or not isinstance(score_comparison_tolerance, (int, float))
            or not math.isfinite(float(score_comparison_tolerance))
            or score_comparison_tolerance <= 0.0
        ):
            raise ValueError("score comparison tolerance must be finite and positive")
        self.score_comparison_tolerance = float(score_comparison_tolerance)

    def project(
        self,
        reader: RunBundleReader,
        previous: ProjectionResult | None = None,
    ) -> ProjectionResult:
        bundle = reader.manifest
        authorities = self._load_authorities(reader)
        self._validate_supersession_frontier(bundle, previous)
        previous_ids = {} if previous is None else previous.source_projection_ids
        ideas_by_id = {idea.idea_id: idea for idea in authorities.archive.ideas}
        events_by_node = self._events_by_node(authorities.events)
        terminal_by_node = {
            record.node_id: record for record in authorities.history_records
        }
        fingerprints = {
            item.evaluation_fingerprint_id: item
            for item in authorities.descriptor.evaluation_fingerprints
        }
        run_log_refs = tuple(
            BundleArtifactRef(
                relative_path=path,
                checksum=bundle.checksums[path],
            )
            for path in sorted(bundle.run_log_refs)
        )

        episodes: list[TransferEpisode] = []
        episode_id_by_node: dict[int, str] = {}
        for node_id in sorted(events_by_node):
            events = events_by_node[node_id]
            terminal = terminal_by_node[node_id]
            idea = ideas_by_id[terminal.idea_id]
            parent_episode_ref = (
                None
                if terminal.parent_node_id is None
                else episode_id_by_node[terminal.parent_node_id]
            )
            attempts = tuple(
                self._project_attempt(
                    bundle,
                    event,
                    terminal_by_node,
                    fingerprints,
                    authorities.descriptor,
                )
                for event in events
            )
            source_key = _source_idea_key(bundle, idea.idea_id)
            episode = TransferEpisode.mint(
                source={
                    "scope_id": bundle.scope_id,
                    "run_id": bundle.run_id,
                    "campaign_id": bundle.campaign_id,
                    "node_id": str(node_id),
                    "idea_id": idea.idea_id,
                    "batch_id": terminal.selection_batch_id,
                },
                source_bundle_id=bundle.bundle_id,
                supersedes_projection_id=previous_ids.get(source_key),
                task_context_binding=bundle.task_context_binding,
                artifact_environment=bundle.artifact_environment,
                proposal=idea.proposal,
                parent_episode_ref=parent_episode_ref,
                attempts=attempts,
                terminal_attempt_revision=attempts[-1].execution_revision,
                safe_observation_refs=run_log_refs,
                sanitation_report_id=authorities.sanitation_report.report_id,
                derivation_refs=tuple(
                    sorted({bundle.bundle_id, *(event.event_id for event in events)})
                ),
            )
            episodes.append(episode)
            episode_id_by_node[node_id] = episode.episode_id

        linked_idea_ids = {terminal.idea_id for terminal in authorities.history_records}
        batches_by_id = {batch.batch_id: batch for batch in authorities.archive.batches}
        prior_ideas = tuple(
            self._project_prior_idea(
                bundle,
                authorities.sanitation_report,
                idea,
                batches_by_id[idea.origin_batch_id],
                previous_ids.get(_source_idea_key(bundle, idea.idea_id)),
            )
            for idea in sorted(authorities.archive.ideas, key=lambda item: item.idea_id)
            if idea.idea_id not in linked_idea_ids
        )
        result = ProjectionResult(
            source_bundle=bundle,
            sanitation_report=authorities.sanitation_report,
            episodes=tuple(
                sorted(episodes, key=lambda item: _episode_source_key(item))
            ),
            prior_ideas=prior_ideas,
            derivation_objects=tuple(
                sorted(authorities.events, key=lambda item: item.event_id)
            ),
        )
        expected_sources = {
            _source_idea_key(bundle, idea.idea_id) for idea in authorities.archive.ideas
        }
        if set(result.source_projection_ids) != expected_sources:
            raise BundleProjectionError("every archive idea must project exactly once")
        if previous is not None and set(previous_ids) - expected_sources:
            raise BundleProjectionError("a successor bundle dropped a source idea")
        previous_episode_sources = (
            {_episode_source_key(item) for item in previous.episodes}
            if previous is not None
            else set()
        )
        current_prior_sources = {_prior_source_key(item) for item in prior_ideas}
        if previous_episode_sources & current_prior_sources:
            raise BundleProjectionError(
                "an executed idea cannot revert to a prior idea"
            )
        return result

    def _load_authorities(self, reader: RunBundleReader) -> _BundleAuthorities:
        bundle = reader.manifest
        if not isinstance(bundle, RunBundle):
            raise BundleProjectionError("bundle reader manifest is not a RunBundle")
        payloads: dict[str, bytes] = {}
        for path, checksum in sorted(bundle.checksums.items()):
            payload = reader.read_ref(path)
            if tree_or_blob_digest(payload) != checksum:
                raise BundleProjectionError(f"bundle checksum mismatch: {path}")
            payloads[path] = payload

        descriptor = CaptureDescriptor.from_json_bytes(
            payloads[bundle.capture_descriptor_ref]
        )
        report = SanitationReport.from_json_bytes(
            payloads[bundle.sanitation_report_ref]
        )
        checkpoint_payload = parse_json_bytes(payloads[bundle.checkpoint_ref])
        if not isinstance(checkpoint_payload, dict):
            raise BundleProjectionError("bundle checkpoint must be an object")
        checkpoint = RunCheckpoint.from_dict(checkpoint_payload)
        if canonical_json_bytes(checkpoint_payload) != canonical_json_bytes(
            checkpoint.to_dict()
        ):
            raise BundleProjectionError("bundle checkpoint schema is not exact")
        archive_payload = parse_json_bytes(payloads[bundle.idea_archive_ref])
        if not isinstance(archive_payload, dict):
            raise BundleProjectionError("bundle idea archive must be an object")
        archive = IdeaArchiveState.from_dict(archive_payload)
        history_records, history_revision, history_identity = _parse_history(
            payloads[bundle.experiment_history_ref]
        )
        events = _parse_events(payloads[bundle.execution_event_journal_ref])
        branch_snapshots = tuple(
            BranchSnapshot.from_json_bytes(payloads[path])
            for path in bundle.branch_snapshot_refs
        )
        self._validate_bundle_authorities(
            bundle,
            descriptor,
            report,
            checkpoint,
            archive,
            history_records,
            history_revision,
            history_identity,
            events,
            branch_snapshots,
            payloads,
        )
        return _BundleAuthorities(
            descriptor=descriptor,
            sanitation_report=report,
            archive=archive,
            history_records=history_records,
            events=events,
        )

    def _validate_bundle_authorities(
        self,
        bundle: RunBundle,
        descriptor: CaptureDescriptor,
        report: SanitationReport,
        checkpoint: RunCheckpoint,
        archive: IdeaArchiveState,
        history_records: tuple[ExperimentRecord, ...],
        history_revision: int,
        history_identity: tuple[str, str, str, bool],
        events: tuple[ExecutionRevisionEvent, ...],
        branch_snapshots: tuple[BranchSnapshot, ...],
        payloads: Mapping[str, bytes],
    ) -> None:
        descriptor_fields = (
            "scope_contract_id",
            "scope_id",
            "run_id",
            "campaign_id",
            "completion_state",
            "capture_generation",
            "started_at",
            "captured_at",
            "kapso_commit",
            "launch_manifest_id",
            "knowledge_snapshot_id",
            "expert_base_release_id",
            "task_context_binding",
            "artifact_environment",
            "artifact_completeness",
            "branch_snapshot_refs",
            "run_log_refs",
        )
        conflicts = tuple(
            name
            for name in descriptor_fields
            if getattr(bundle, name) != getattr(descriptor, name)
        )
        if conflicts:
            raise BundleProjectionError(
                f"bundle and capture descriptor conflict: {conflicts}"
            )
        expected_core_refs = {
            "capture_descriptor": bundle.capture_descriptor_ref,
            "checkpoint": bundle.checkpoint_ref,
            "execution_event_journal": bundle.execution_event_journal_ref,
            "idea_archive": bundle.idea_archive_ref,
            "experiment_history": bundle.experiment_history_ref,
        }
        if any(
            descriptor.artifact_refs.get(name) != path
            for name, path in expected_core_refs.items()
        ):
            raise BundleProjectionError("descriptor core artifact refs changed")
        if len(descriptor.artifact_refs) != len(set(descriptor.artifact_refs.values())):
            raise BundleProjectionError("descriptor artifact refs are not one-to-one")
        if set(descriptor.artifact_refs.values()) != (
            set(bundle.checksums) - {bundle.sanitation_report_ref}
        ):
            raise BundleProjectionError("descriptor artifact closure is not exact")
        if report.status != "admitted":
            raise BundleProjectionError("only admitted sanitation reports can project")
        if (
            report.scope_id != bundle.scope_id
            or report.task_family_id != bundle.task_context_binding.task_family_id
            or dict(report.admitted_refs)
            != {
                path: digest
                for path, digest in bundle.checksums.items()
                if path != bundle.sanitation_report_ref
            }
        ):
            raise BundleProjectionError("sanitation report does not bind the bundle")
        if archive.campaign_id != bundle.campaign_id:
            raise BundleProjectionError("archive belongs to another campaign")
        run_id, campaign_id, objective_direction, require_idea_links = history_identity
        if (
            run_id != bundle.run_id
            or campaign_id != bundle.campaign_id
            or not require_idea_links
        ):
            raise BundleProjectionError("experiment history identity is incompatible")
        if any(
            event.run_id != bundle.run_id or event.campaign_id != bundle.campaign_id
            for event in events
        ):
            raise BundleProjectionError("execution journal identity is incompatible")
        if history_revision != len(events):
            raise BundleProjectionError("history and journal revisions differ")
        terminal_records = _terminal_records(events)
        if terminal_records != history_records:
            raise BundleProjectionError(
                "history is not the journal terminal projection"
            )
        if any(
            record.objective_direction != objective_direction
            for record in history_records
        ):
            raise BundleProjectionError("experiment objective direction changed")
        if checkpoint.completed_iterations != bundle.checkpoint_frontier:
            raise BundleProjectionError("checkpoint frontier is false")
        expected_checkpoint_status = (
            "completed"
            if bundle.completion_state is CompletionState.COMPLETE
            else "running"
        )
        if checkpoint.status != expected_checkpoint_status:
            raise BundleProjectionError("checkpoint completion state is inconsistent")
        node_history = checkpoint.strategy_state.get("node_history")
        iteration_count = checkpoint.strategy_state.get("iteration_count")
        if (
            checkpoint.strategy_type != "generic"
            or set(checkpoint.strategy_state) != GENERIC_SEARCH_STATE_FIELDS
            or checkpoint.strategy_state.get("schema") != GENERIC_SEARCH_STATE_SCHEMA
            or not isinstance(node_history, list)
            or type(iteration_count) is not int
        ):
            raise BundleProjectionError("checkpoint strategy frontier is unavailable")
        nodes = tuple(SearchNode.from_dict(item) for item in node_history)
        if any(
            canonical_json_bytes(raw) != canonical_json_bytes(node.to_dict())
            for raw, node in zip(node_history, nodes)
        ):
            raise BundleProjectionError("checkpoint node schema is not exact")
        if len(nodes) != len(history_records):
            raise BundleProjectionError("checkpoint and history node counts differ")
        for node, record in zip(nodes, history_records):
            if (
                ExperimentRecord.from_node(
                    node,
                    record.objective_direction,
                    require_idea_links,
                    record.solution_embedding,
                )
                != record
            ):
                raise BundleProjectionError("checkpoint node projection changed")
        branch_keys = {
            f"branch:{event.node_id}:{event.execution_revision}" for event in events
        }
        expected_completeness = {
            "checkpoint",
            "execution_event_journal",
            "idea_archive",
            "experiment_history",
            *branch_keys,
            *(f"run_log:{path}" for path in descriptor.run_log_refs),
        }
        if set(bundle.artifact_completeness) != expected_completeness:
            raise BundleProjectionError("artifact completeness closure is not exact")
        present_branch_refs = {
            descriptor.artifact_refs[key]
            for key in branch_keys
            if bundle.artifact_completeness[key] is ArtifactCompleteness.PRESENT
        }
        if present_branch_refs != set(bundle.branch_snapshot_refs):
            raise BundleProjectionError("branch snapshot frontier is inconsistent")
        branches_by_revision = {
            (branch.node_id, branch.execution_revision): branch
            for branch in branch_snapshots
        }
        if len(branches_by_revision) != len(branch_snapshots):
            raise BundleProjectionError("branch snapshot revisions are not unique")
        source_payload_refs: set[str] = set()
        for event in events:
            branch_key = f"branch:{event.node_id}:{event.execution_revision}"
            branch = branches_by_revision.get((event.node_id, event.execution_revision))
            completeness = bundle.artifact_completeness[branch_key]
            if branch is None:
                if (
                    completeness is not ArtifactCompleteness.UNAVAILABLE
                    or event.execution_status is ExecutionStatus.COMPLETED
                ):
                    raise BundleProjectionError("branch completeness is false")
                continue
            if completeness is not ArtifactCompleteness.PRESENT:
                raise BundleProjectionError("present branch is not declared present")
            record = ExperimentRecord.from_dict(to_json_value(event.projection))
            if descriptor.artifact_refs[branch_key] not in bundle.branch_snapshot_refs:
                raise BundleProjectionError(
                    "branch logical ref does not name its manifest"
                )
            source_payload_refs.update(
                validate_branch_evidence(
                    read_ref=payloads.__getitem__,
                    descriptor=descriptor,
                    record=record,
                    event=event,
                    branch=branch,
                    error_type=BundleProjectionError,
                )
            )
        structural_refs = {
            descriptor.artifact_refs["capture_descriptor"],
            descriptor.artifact_refs["checkpoint"],
            descriptor.artifact_refs["execution_event_journal"],
            descriptor.artifact_refs["idea_archive"],
            descriptor.artifact_refs["experiment_history"],
            *descriptor.branch_snapshot_refs,
            *descriptor.run_log_refs,
        }
        unexplained_refs = set(descriptor.artifact_refs.values()) - structural_refs
        if unexplained_refs != source_payload_refs:
            raise BundleProjectionError("source payload closure is not exact")
        expected_watermarks = {
            "branch_snapshot_count": len(bundle.branch_snapshot_refs),
            "checkpoint_completed_iterations": checkpoint.completed_iterations,
            "checkpoint_node_count": len(node_history),
            "execution_journal_event_count": len(events),
            "experiment_history_count": len(history_records),
            "experiment_history_revision": history_revision,
            "idea_archive_revision": archive.revision,
            "strategy_iteration_count": iteration_count,
        }
        if dict(bundle.capture_watermarks) != expected_watermarks:
            raise BundleProjectionError("bundle capture watermarks are false")
        ideas_by_id = {idea.idea_id: idea for idea in archive.ideas}
        links: dict[int, str] = {}
        for idea in archive.ideas:
            if idea.experiment_node_id is not None:
                if idea.experiment_node_id in links:
                    raise BundleProjectionError("multiple ideas link to one node")
                links[idea.experiment_node_id] = idea.idea_id
        if set(links) != set(range(len(history_records))):
            raise BundleProjectionError("archive node linkage exceeds the frontier")
        for record in history_records:
            if record.idea_id not in ideas_by_id:
                raise BundleProjectionError("executed node has no source idea")
            idea = ideas_by_id[record.idea_id]
            if (
                links.get(record.node_id) != record.idea_id
                or idea.selected_in_batch_id != record.selection_batch_id
                or idea.experiment_node_id != record.node_id
            ):
                raise BundleProjectionError("archive and history links diverged")
        self._validate_evaluation_closure(events, descriptor.evaluation_fingerprints)

    def _validate_evaluation_closure(
        self,
        events: tuple[ExecutionRevisionEvent, ...],
        fingerprints: tuple[EvaluationFingerprint, ...],
    ) -> None:
        registry = {item.evaluation_fingerprint_id: item for item in fingerprints}
        referenced: set[str] = set()
        for event in events:
            record = ExperimentRecord.from_dict(to_json_value(event.projection))
            groups, _ = _evaluation_groups(
                record,
                fingerprints,
                self.score_comparison_tolerance,
            )
            referenced.update(
                group.fingerprint.evaluation_fingerprint_id for group in groups
            )
            expected_evaluators = tuple(
                sorted({attempt.evaluator_id for attempt in record.evaluation_attempts})
            )
            if event.evaluator_fingerprint_ids != expected_evaluators:
                raise BundleProjectionError("journal evaluator registry changed")
            expected_measurements = dict(record.metrics)
            if record.raw_score is not None:
                expected_measurements["raw_score"] = record.raw_score
            if dict(event.measurements) != expected_measurements:
                raise BundleProjectionError("journal measurements changed")
        if tuple(sorted(referenced)) != tuple(sorted(registry)):
            raise BundleProjectionError(
                "evaluation fingerprint registry is not an exact closure"
            )

    def _validate_supersession_frontier(
        self,
        bundle: RunBundle,
        previous: ProjectionResult | None,
    ) -> None:
        if previous is None:
            validate_run_bundle_root(bundle, BundleProjectionError)
            return
        prior = previous.source_bundle
        validate_run_bundle_successor(prior, bundle, BundleProjectionError)

    @staticmethod
    def _events_by_node(
        events: tuple[ExecutionRevisionEvent, ...],
    ) -> dict[int, tuple[ExecutionRevisionEvent, ...]]:
        mutable: dict[int, list[ExecutionRevisionEvent]] = {}
        for event in events:
            mutable.setdefault(event.node_id, []).append(event)
        grouped = {
            node_id: tuple(sorted(items, key=lambda item: item.execution_revision))
            for node_id, items in mutable.items()
        }
        for node_id, items in grouped.items():
            if tuple(item.execution_revision for item in items) != tuple(
                range(len(items))
            ):
                raise BundleProjectionError(
                    f"node {node_id} revisions are not zero-based and gap-free"
                )
        return grouped

    def _project_attempt(
        self,
        bundle: RunBundle,
        event: ExecutionRevisionEvent,
        terminal_by_node: Mapping[int, ExperimentRecord],
        fingerprints: Mapping[str, EvaluationFingerprint],
        descriptor: CaptureDescriptor,
    ) -> TransferAttempt:
        record = ExperimentRecord.from_dict(to_json_value(event.projection))
        groups, score_group = _evaluation_groups(
            record,
            tuple(fingerprints.values()),
            self.score_comparison_tolerance,
        )
        measurements = dict(record.metrics)
        if score_group is not None:
            metric_name = score_group.fingerprint.metric_name
            existing = measurements.get(metric_name)
            if existing is not None and not evaluation_scores_match(
                existing,
                score_group.aggregate,
                self.score_comparison_tolerance,
            ):
                raise BundleProjectionError("score-of-record metric is inconsistent")
            measurements[metric_name] = score_group.aggregate
        parent_effect = None
        comparison_status = (
            ComparisonStatus.INCONCLUSIVE
            if event.evaluation_status
            in {
                EpisodeEvaluationStatus.INVALID,
                EpisodeEvaluationStatus.PARTIAL,
                EpisodeEvaluationStatus.NOT_RUN,
            }
            else ComparisonStatus.NOT_COMPARABLE
        )
        parent = (
            None
            if record.parent_node_id is None
            else terminal_by_node.get(record.parent_node_id)
        )
        if score_group is not None and parent is not None:
            _, parent_score_group = _evaluation_groups(
                parent,
                tuple(fingerprints.values()),
                self.score_comparison_tolerance,
            )
            if parent_score_group is not None and (
                parent_score_group.fingerprint.evaluation_fingerprint_id
                == score_group.fingerprint.evaluation_fingerprint_id
            ):
                raw_delta = score_group.aggregate - parent_score_group.aggregate
                direction = (
                    1.0
                    if score_group.fingerprint.objective_direction
                    is ObjectiveDirection.MAXIMIZE
                    else -1.0
                )
                parent_effect = RelativeEffect(
                    evaluation_fingerprint_id=(
                        score_group.fingerprint.evaluation_fingerprint_id
                    ),
                    metric_name=score_group.fingerprint.metric_name,
                    objective_direction=score_group.fingerprint.objective_direction,
                    candidate_value=score_group.aggregate,
                    source_parent_value=parent_score_group.aggregate,
                    raw_delta=raw_delta,
                    normalized_delta=direction * raw_delta,
                    uncertainty=None,
                    uncertainty_method=EffectUncertaintyMethod.UNAVAILABLE,
                )
                comparison_status = ComparisonStatus.COMPARABLE
        branch_key = f"branch:{event.node_id}:{event.execution_revision}"
        branch_ref = descriptor.artifact_refs.get(branch_key)
        intervention_ref = (
            None
            if branch_ref is None
            else BundleArtifactRef(
                relative_path=branch_ref,
                checksum=bundle.checksums[branch_ref],
            )
        )
        return TransferAttempt(
            execution_revision=event.execution_revision,
            captured_at=event.recorded_at,
            execution_status=event.execution_status,
            evaluation_status=event.evaluation_status,
            evaluation_fingerprints=tuple(
                sorted(
                    (group.fingerprint for group in groups),
                    key=lambda item: item.evaluation_fingerprint_id,
                )
            ),
            score_of_record_fingerprint_id=(
                None
                if score_group is None
                else score_group.fingerprint.evaluation_fingerprint_id
            ),
            comparison_status=comparison_status,
            measurements=measurements,
            source_parent_effect=parent_effect,
            intervention_ref=intervention_ref,
            intervention_structure=InterventionStructure.UNDETERMINED,
            feedback=_observation_tuple(event.feedback),
            technical_difficulties=_observation_tuple(event.technical_difficulties),
            confounders=(),
        )

    @staticmethod
    def _project_prior_idea(
        bundle: RunBundle,
        report: SanitationReport,
        idea: IdeaRecord,
        origin_batch: IdeaBatch,
        supersedes_projection_id: str | None,
    ) -> PriorIdea:
        if idea.status is IdeaStatus.DEFERRED:
            status = PriorIdeaStatus.DEFERRED
            rationale = idea.deferral_reason
        elif idea.status in {IdeaStatus.INVALID, IdeaStatus.REJECTED}:
            status = PriorIdeaStatus.REJECTED
            rationale = idea.rejection_reason
        elif idea.status is IdeaStatus.ABANDONED:
            status = PriorIdeaStatus.UNEXECUTED
            rationale = origin_batch.abandoned_reason
        else:
            status = PriorIdeaStatus.UNEXECUTED
            rationale = idea.selection_reason or idea.directive_rationale
        if rationale is None:
            raise BundleProjectionError("prior idea has no exact source rationale")
        return PriorIdea.mint(
            source_bundle_id=bundle.bundle_id,
            supersedes_projection_id=supersedes_projection_id,
            source={
                "scope_id": bundle.scope_id,
                "run_id": bundle.run_id,
                "campaign_id": bundle.campaign_id,
                "batch_id": idea.origin_batch_id,
                "idea_id": idea.idea_id,
            },
            proposal=idea.proposal,
            descriptor=idea.descriptor.to_dict(),
            assumptions=idea.assumptions,
            source_status=status,
            source_rationale=rationale,
            source_evidence_refs=tuple(sorted(set(idea.evidence_refs))),
            task_context_binding=bundle.task_context_binding,
            sanitation_report_id=report.report_id,
        )


def _parse_history(
    payload: bytes,
) -> tuple[tuple[ExperimentRecord, ...], int, tuple[str, str, str, bool]]:
    parsed = parse_json_bytes(payload)
    expected_fields = {
        "schema",
        "run_id",
        "campaign_id",
        "revision",
        "objective_direction",
        "require_idea_links",
        "records",
    }
    if not isinstance(parsed, dict) or set(parsed) != expected_fields:
        raise BundleProjectionError("experiment history fields are invalid")
    if parsed["schema"] != EXPERIMENT_HISTORY_SCHEMA:
        raise BundleProjectionError("experiment history schema is incompatible")
    if (
        not isinstance(parsed["run_id"], str)
        or not isinstance(parsed["campaign_id"], str)
        or parsed["objective_direction"] not in {"maximize", "minimize"}
        or type(parsed["require_idea_links"]) is not bool
        or type(parsed["revision"]) is not int
        or parsed["revision"] < 0
        or not isinstance(parsed["records"], list)
    ):
        raise BundleProjectionError("experiment history identity is invalid")
    records = tuple(ExperimentRecord.from_dict(item) for item in parsed["records"])
    if tuple(record.node_id for record in records) != tuple(range(len(records))):
        raise BundleProjectionError("experiment history node ids are not contiguous")
    return (
        records,
        parsed["revision"],
        (
            parsed["run_id"],
            parsed["campaign_id"],
            parsed["objective_direction"],
            parsed["require_idea_links"],
        ),
    )


def _parse_events(payload: bytes) -> tuple[ExecutionRevisionEvent, ...]:
    if payload and not payload.endswith(b"\n"):
        raise BundleProjectionError("execution journal has an incomplete tail")
    lines = payload.splitlines()
    if any(not line for line in lines):
        raise BundleProjectionError("execution journal contains a blank event")
    events = tuple(ExecutionRevisionEvent.from_json_bytes(line) for line in lines)
    RunBundleProjector._events_by_node(events)
    first_nodes: list[int] = []
    seen: set[int] = set()
    for event in events:
        if event.node_id not in seen:
            seen.add(event.node_id)
            first_nodes.append(event.node_id)
    if first_nodes != list(range(len(first_nodes))):
        raise BundleProjectionError("execution journal node ids are not contiguous")
    return events


def _terminal_records(
    events: tuple[ExecutionRevisionEvent, ...],
) -> tuple[ExperimentRecord, ...]:
    terminal: dict[int, ExperimentRecord] = {}
    for event in events:
        terminal[event.node_id] = ExperimentRecord.from_dict(
            to_json_value(event.projection)
        )
    if tuple(sorted(terminal)) != tuple(range(len(terminal))):
        raise BundleProjectionError("journal terminal frontier is incomplete")
    return tuple(terminal[node_id] for node_id in sorted(terminal))


def _evaluation_groups(
    record: ExperimentRecord,
    fingerprints: tuple[EvaluationFingerprint, ...],
    score_comparison_tolerance: float,
) -> tuple[tuple[_EvaluationGroup, ...], _EvaluationGroup | None]:
    grouped: dict[tuple[str, str, float], list[Any]] = {}
    for attempt in record.evaluation_attempts:
        grouped.setdefault(
            (attempt.evaluator_id, attempt.fidelity, attempt.fraction), []
        ).append(attempt)
    groups: list[_EvaluationGroup] = []
    for (evaluator_id, fidelity, fraction), attempts in sorted(grouped.items()):
        seed_ids = tuple(sorted(f"seed-{attempt.seed}" for attempt in attempts))
        if len(seed_ids) != len(set(seed_ids)):
            raise BundleProjectionError("evaluation group contains duplicate seeds")
        matches = tuple(
            fingerprint
            for fingerprint in fingerprints
            if fingerprint.evaluator_fingerprint == f"sha256:{evaluator_id}"
            and fingerprint.objective_direction.value == record.objective_direction
            and fingerprint.fidelity == fidelity
            and fingerprint.fraction == fraction
            and fingerprint.seed_or_replicate_ids == seed_ids
            and (
                record.primary_metric is None
                or fingerprint.metric_name == record.primary_metric
            )
            and all(
                attempt.metrics.get(fingerprint.metric_name) == attempt.score
                for attempt in attempts
            )
        )
        if len(matches) != 1:
            raise BundleProjectionError(
                "evaluation group lacks one exact full fingerprint"
            )
        if matches[0].aggregation_protocol != "arithmetic-mean":
            raise BundleProjectionError("evaluation aggregation is unsupported")
        groups.append(
            _EvaluationGroup(
                fingerprint=matches[0],
                aggregate=sum(attempt.score for attempt in attempts) / len(attempts),
            )
        )
    score_groups = tuple(
        group
        for group in groups
        if record.raw_score is not None
        and evaluation_scores_match(
            record.raw_score,
            group.aggregate,
            score_comparison_tolerance,
        )
    )
    if record.evaluation_valid and record.raw_score is not None:
        if len(score_groups) != 1:
            raise BundleProjectionError("raw score has no unambiguous evaluator group")
        score_group = score_groups[0]
    else:
        if record.evaluation_valid and record.raw_score is None and groups:
            raise BundleProjectionError("measured valid record has no score of record")
        score_group = None
    return tuple(groups), score_group


def _observation_tuple(value: str) -> tuple[str, ...]:
    return () if not value else (value,)


def _source_idea_key(
    bundle: RunBundle,
    idea_id: str,
) -> tuple[str, str, str, str]:
    return (bundle.scope_id, bundle.run_id, bundle.campaign_id, idea_id)


def _episode_source_key(
    episode: TransferEpisode,
) -> tuple[str, str, str, str]:
    return (
        episode.source["scope_id"],
        episode.source["run_id"],
        episode.source["campaign_id"],
        episode.source["idea_id"],
    )


def _prior_source_key(prior: PriorIdea) -> tuple[str, str, str, str]:
    return (
        prior.source["scope_id"],
        prior.source["run_id"],
        prior.source["campaign_id"],
        prior.source["idea_id"],
    )
