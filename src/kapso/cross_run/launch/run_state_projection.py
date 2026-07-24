"""Pure mutually reconciled projections for one run-state checkpoint frontier."""

from __future__ import annotations

import re
from dataclasses import dataclass

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    to_json_value,
    tree_or_blob_digest,
)
from kapso.cross_run.capture.revision_projection import ExecutionRevisionProjection
from kapso.cross_run.contracts import (
    EpisodeEvaluationStatus,
    ExecutionStatus,
)
from kapso.cross_run.launch.derived_state_bundle import RunDerivedStateBundle
from kapso.cross_run.launch.checkpoint_contracts import (
    RunStrategyKind,
    RunStrategyState,
)
from kapso.cross_run.launch.contracts import BootstrapPin
from kapso.cross_run.launch.derived_state_contracts import (
    RunDerivedStateGeneration,
    RunStateAuthority,
    RunStateLayout,
    RunStatePayloadTransition,
)
from kapso.cross_run.record_contracts import ExecutionRevisionEvent
from kapso.execution.memories.experiment_memory.projection import (
    ExperimentHistoryProjection,
    require_experiment_record_successor,
)
from kapso.execution.memories.experiment_memory.record import (
    ExperimentNodeProjection,
    ExperimentRecord,
)
from kapso.execution.search_strategies.generic.ideation.archive import IdeaArchiveState
from kapso.execution.search_strategies.generic.ideation.archive_projection import (
    decode_archive_state,
    encode_archive_state,
)
from kapso.execution.search_strategies.generic.ideation.types import (
    EvaluationStatus,
    IdeaOutcome,
    IdeaRecord,
    IdeaStatus,
    ImplementationStatus,
    ParentPlanKind,
)


class RunStateProjectionError(ValueError):
    """The archive, history, and journal do not represent one exact frontier."""


_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class ReconciledRunStateProjection:
    """Typed in-memory views which are safe to stage as one generation."""

    strategy_state: RunStrategyState
    experiment_history: ExperimentHistoryProjection
    execution_journal: ExecutionRevisionProjection
    idea_archive: IdeaArchiveState | None

    def __post_init__(self) -> None:
        if type(self.strategy_state) is not RunStrategyState:
            raise RunStateProjectionError(
                "run-state projection requires one exact strategy state"
            )
        if type(self.experiment_history) is not ExperimentHistoryProjection:
            raise RunStateProjectionError(
                "run-state projection requires one exact experiment history"
            )
        if type(self.execution_journal) is not ExecutionRevisionProjection:
            raise RunStateProjectionError(
                "run-state projection requires one exact execution journal"
            )
        history = self.experiment_history
        journal = self.execution_journal
        if (
            history.run_id != journal.run_id
            or history.campaign_id != journal.campaign_id
            or history.campaign_id != self.strategy_state.campaign_id
        ):
            raise RunStateProjectionError(
                "run-state strategy, history, and journal identities differ"
            )
        if history.revision != journal.watermark:
            raise RunStateProjectionError(
                "run-state history revision differs from journal watermark"
            )
        projected_records = tuple(
            ExperimentRecord.from_dict(to_json_value(event.projection))
            for event in journal.events
        )
        prior_record_by_node: dict[int, ExperimentRecord] = {}
        for record in projected_records:
            prior = prior_record_by_node.get(record.node_id)
            if prior is not None:
                require_experiment_record_successor(prior, record)
            prior_record_by_node[record.node_id] = record
        if any(
            canonical_json_bytes(record.to_dict())
            != canonical_json_bytes(event.projection)
            for record, event in zip(
                projected_records,
                journal.events,
                strict=True,
            )
        ):
            raise RunStateProjectionError(
                "run-state journal contains a non-exact experiment projection"
            )
        for record, event in zip(
            projected_records,
            journal.events,
            strict=True,
        ):
            _require_event_matches_record(event, record)
        terminal_by_node: dict[int, ExperimentRecord] = {}
        for record in projected_records:
            terminal_by_node[record.node_id] = record
        history_node_ids = tuple(record.node_id for record in history.records)
        if set(terminal_by_node) != set(history_node_ids):
            raise RunStateProjectionError(
                "run-state history and journal node sets differ"
            )
        if (
            tuple(event.node_id for event in journal.terminal_events)
            != history_node_ids
        ):
            raise RunStateProjectionError(
                "run-state history order differs from journal execution order"
            )
        terminal_records = tuple(
            terminal_by_node[node_id] for node_id in history_node_ids
        )
        if terminal_records != history.records:
            raise RunStateProjectionError(
                "run-state history differs from journal terminal projections"
            )
        strategy_nodes = self.strategy_state.nodes()
        if self.strategy_state.strategy_kind is RunStrategyKind.BENCHMARK_TREE_SEARCH:
            history_ids = self.strategy_state.parsed_state()["node_history_ids"]
            executed_nodes = tuple(strategy_nodes[node_id] for node_id in history_ids)
        else:
            executed_nodes = strategy_nodes
        if len(executed_nodes) != len(history.records):
            raise RunStateProjectionError(
                "run-state strategy executed nodes differ from experiment history"
            )
        expected_records = tuple(
            ExperimentRecord.from_node(
                node,
                history.objective_direction,
                history.require_idea_links,
                record.solution_embedding,
            )
            for node, record in zip(
                executed_nodes,
                history.records,
                strict=True,
            )
        )
        if expected_records != history.records:
            raise RunStateProjectionError(
                "run-state strategy nodes differ from experiment history"
            )
        node_by_id = {node.node_id: node for node in strategy_nodes}
        if self.strategy_kind == "benchmark_tree_search":
            if self.idea_archive is not None or history.require_idea_links:
                raise RunStateProjectionError(
                    "tree run-state projection cannot contain generic idea authority"
                )
            _require_event_artifact_lineage(
                journal.events,
                projected_records,
                node_by_id,
                None,
            )
            return
        if type(self.idea_archive) is not IdeaArchiveState:
            raise RunStateProjectionError(
                "generic run-state projection requires one exact idea archive"
            )
        if (
            self.idea_archive.campaign_id != history.campaign_id
            or not history.require_idea_links
            or self.idea_archive != self.strategy_state.archive_state()
        ):
            raise RunStateProjectionError(
                "generic strategy, archive, or history authority differs"
            )
        _require_event_artifact_lineage(
            journal.events,
            projected_records,
            node_by_id,
            self.idea_archive,
        )
        first_nonrecoverable_context_by_node: dict[
            int,
            tuple[ExperimentRecord, dict[int, ExperimentRecord]],
        ] = {}
        latest_record_by_node: dict[int, ExperimentRecord] = {}
        for record in projected_records:
            if (
                not record.recoverable_error
                and record.node_id not in first_nonrecoverable_context_by_node
            ):
                first_nonrecoverable_context_by_node[record.node_id] = (
                    record,
                    dict(latest_record_by_node),
                )
            latest_record_by_node[record.node_id] = record
        ideas_by_id = {idea.idea_id: idea for idea in self.idea_archive.ideas}
        records_by_node = {record.node_id: record for record in history.records}
        for record in history.records:
            idea = ideas_by_id.get(record.idea_id)
            if (
                idea is None
                or idea.selected_in_batch_id != record.selection_batch_id
                or idea.experiment_node_id != record.node_id
            ):
                raise RunStateProjectionError(
                    "generic experiment record lacks its exact archive link"
                )
            outcome_context = first_nonrecoverable_context_by_node.get(record.node_id)
            if outcome_context is None:
                if (
                    idea.status is not IdeaStatus.IMPLEMENTING
                    or idea.outcome is not None
                ):
                    raise RunStateProjectionError(
                        "recoverable-only experiment has a terminal archive outcome"
                    )
            else:
                outcome_record, records_before_outcome = outcome_context
                _require_outcome_matches_record(
                    idea.status,
                    idea.outcome,
                    outcome_record,
                    objective_direction=history.objective_direction,
                    parent_record=_require_outcome_parent_record(
                        idea,
                        outcome_record,
                        records_before_outcome,
                    ),
                )
        for idea in self.idea_archive.ideas:
            if idea.outcome is None and idea.status not in {
                IdeaStatus.EVALUATED,
                IdeaStatus.FAILED_TECHNICAL,
            }:
                continue
            record = records_by_node.get(idea.experiment_node_id)
            outcome_context = first_nonrecoverable_context_by_node.get(
                idea.experiment_node_id
            )
            if (
                record is None
                or outcome_context is None
                or record.idea_id != idea.idea_id
                or record.selection_batch_id != idea.selected_in_batch_id
            ):
                raise RunStateProjectionError(
                    "terminal archive idea lacks its executed history closure"
                )
            outcome_record, records_before_outcome = outcome_context
            _require_outcome_matches_record(
                idea.status,
                idea.outcome,
                outcome_record,
                objective_direction=history.objective_direction,
                parent_record=_require_outcome_parent_record(
                    idea,
                    outcome_record,
                    records_before_outcome,
                ),
            )

    @property
    def strategy_kind(self) -> str:
        """Return the checkpoint strategy namespace owned by this projection."""
        return self.strategy_state.strategy_kind.value

    @property
    def revision_by_authority(self) -> dict[RunStateAuthority, int]:
        """Return the semantic revision named by each canonical payload."""
        revisions = {
            RunStateAuthority.EXPERIMENT_HISTORY: self.experiment_history.revision,
            RunStateAuthority.EXECUTION_JOURNAL: self.execution_journal.watermark,
        }
        if self.idea_archive is not None:
            revisions[RunStateAuthority.IDEA_ARCHIVE] = self.idea_archive.revision
        return revisions

    @property
    def payload_by_authority(self) -> dict[RunStateAuthority, bytes]:
        """Return the exact canonical bytes for every strategy-owned authority."""
        payloads = {
            RunStateAuthority.EXPERIMENT_HISTORY: (
                self.experiment_history.to_json_bytes()
            ),
            RunStateAuthority.EXECUTION_JOURNAL: (self.execution_journal.jsonl_bytes),
        }
        if self.idea_archive is not None:
            payloads[RunStateAuthority.IDEA_ARCHIVE] = encode_archive_state(
                self.idea_archive
            )
        return payloads

    def build_bundle(
        self,
        *,
        bootstrap_pin: BootstrapPin,
        run_state_layout: RunStateLayout,
        predecessor_checkpoint_head_id: str,
        predecessor_checkpoint_id: str | None,
        predecessor_evidence_id: str | None,
        target_evidence_id: str,
        predecessor_bundle: RunDerivedStateBundle | None,
        predecessor_strategy_state: RunStrategyState | None,
    ) -> RunDerivedStateBundle:
        """Build the exact immutable generation transition for these projections."""
        self = self._canonical_copy()
        self.require_bootstrap_pin(bootstrap_pin)
        if (
            type(run_state_layout) is not RunStateLayout
            or run_state_layout.strategy_kind != self.strategy_kind
        ):
            raise RunStateProjectionError(
                "run-state projection layout differs from its strategy"
            )
        has_predecessor = predecessor_bundle is not None
        if (
            has_predecessor != (predecessor_checkpoint_id is not None)
            or (has_predecessor != (predecessor_evidence_id is not None))
            or (has_predecessor != (predecessor_strategy_state is not None))
        ):
            raise RunStateProjectionError(
                "run-state predecessor bundle, strategy, and checkpoint fields differ"
            )
        predecessor_transitions = {}
        if predecessor_bundle is not None:
            if type(predecessor_bundle) is not RunDerivedStateBundle:
                raise RunStateProjectionError(
                    "run-state predecessor must be one exact retained bundle"
                )
            predecessor_generation = predecessor_bundle.generation
            if (
                predecessor_generation.bootstrap_pin_id
                != bootstrap_pin.bootstrap_pin_id
                or predecessor_generation.run_state_layout != run_state_layout
                or predecessor_generation.target_evidence_id != predecessor_evidence_id
            ):
                raise RunStateProjectionError(
                    "run-state predecessor bundle names another authority frontier"
                )
            predecessor_projection = type(self).from_bundle(
                predecessor_bundle,
                strategy_state=predecessor_strategy_state,
                bootstrap_pin=bootstrap_pin,
            )
            self.require_predecessor(predecessor_projection)
            predecessor_transitions = {
                transition.authority_binding_id: transition
                for transition in predecessor_generation.payload_transitions
            }
        payload_by_authority = self.payload_by_authority
        revision_by_authority = self.revision_by_authority
        if set(payload_by_authority) != {
            binding.authority for binding in run_state_layout.bindings
        }:
            raise RunStateProjectionError(
                "run-state projection payload set differs from its layout"
            )
        transitions = []
        payloads = []
        for binding in run_state_layout.bindings:
            payload = payload_by_authority[binding.authority]
            previous = predecessor_transitions.get(binding.authority_binding_id)
            transitions.append(
                RunStatePayloadTransition.mint(
                    authority_binding_id=binding.authority_binding_id,
                    predecessor_digest=(
                        None if previous is None else previous.target_digest
                    ),
                    predecessor_revision=(
                        None if previous is None else previous.target_revision
                    ),
                    predecessor_size_bytes=(
                        None if previous is None else previous.target_size_bytes
                    ),
                    target_digest=tree_or_blob_digest(payload),
                    target_revision=revision_by_authority[binding.authority],
                    target_size_bytes=len(payload),
                )
            )
            payloads.append(payload)
        generation = RunDerivedStateGeneration.build(
            bootstrap_pin_id=bootstrap_pin.bootstrap_pin_id,
            run_state_layout=run_state_layout,
            predecessor_checkpoint_head_id=predecessor_checkpoint_head_id,
            predecessor_checkpoint_id=predecessor_checkpoint_id,
            predecessor_evidence_id=predecessor_evidence_id,
            target_evidence_id=target_evidence_id,
            payload_transitions=tuple(transitions),
        )
        return RunDerivedStateBundle(
            generation=generation,
            payloads=tuple(payloads),
        )

    def _canonical_copy(self) -> ReconciledRunStateProjection:
        """Seal mutable nested containers and rerun all reconciliation checks."""
        history = ExperimentHistoryProjection.from_json_bytes(
            self.experiment_history.to_json_bytes()
        )
        journal = ExecutionRevisionProjection.from_jsonl_bytes(
            self.execution_journal.jsonl_bytes,
            run_id=history.run_id,
            campaign_id=history.campaign_id,
            require_contiguous_node_ids=history.require_idea_links,
        )
        archive = (
            None
            if self.idea_archive is None
            else decode_archive_state(encode_archive_state(self.idea_archive))
        )
        return type(self)(
            strategy_state=self.strategy_state,
            experiment_history=history,
            execution_journal=journal,
            idea_archive=archive,
        )

    def require_predecessor(
        self,
        predecessor: ReconciledRunStateProjection,
    ) -> None:
        """Prove that this frontier preserves every predecessor-owned fact."""
        if type(predecessor) is not ReconciledRunStateProjection:
            raise RunStateProjectionError(
                "run-state predecessor must be one reconciled projection"
            )
        current_history = self.experiment_history
        previous_history = predecessor.experiment_history
        if (
            self.strategy_kind != predecessor.strategy_kind
            or current_history.run_id != previous_history.run_id
            or current_history.campaign_id != previous_history.campaign_id
            or current_history.embedding_space_id != previous_history.embedding_space_id
            or current_history.embedding_provider != previous_history.embedding_provider
            or current_history.embedding_model != previous_history.embedding_model
            or current_history.embedding_dimensions
            != previous_history.embedding_dimensions
            or current_history.embedding_canonicalizer_version
            != previous_history.embedding_canonicalizer_version
            or current_history.objective_direction
            != previous_history.objective_direction
            or current_history.require_idea_links
            is not previous_history.require_idea_links
        ):
            raise RunStateProjectionError(
                "run-state predecessor projection identity or policy changed"
            )
        self.strategy_state.require_predecessor(predecessor.strategy_state)
        if not self.execution_journal.jsonl_bytes.startswith(
            predecessor.execution_journal.jsonl_bytes
        ):
            raise RunStateProjectionError(
                "run-state execution journal rewrote predecessor evidence"
            )

    def require_bootstrap_pin(self, bootstrap_pin: BootstrapPin) -> None:
        """Join every local state identity to one exact installed launch."""
        if type(bootstrap_pin) is not BootstrapPin:
            raise RunStateProjectionError(
                "run-state projection requires one exact bootstrap pin"
            )
        self.strategy_state.require_bootstrap_pin(bootstrap_pin)
        receipt = bootstrap_pin.installation_receipt
        if (
            self.experiment_history.run_id != receipt.run_id
            or self.experiment_history.campaign_id != receipt.campaign_id
            or self.execution_journal.run_id != receipt.run_id
            or self.execution_journal.campaign_id != receipt.campaign_id
            or self.experiment_history.embedding_space_id
            != bootstrap_pin.launch_manifest.experiment_embedding_space.embedding_space_id
            or self.experiment_history.embedding_provider
            != bootstrap_pin.launch_manifest.experiment_embedding_space.provider
            or self.experiment_history.embedding_model
            != bootstrap_pin.launch_manifest.experiment_embedding_space.model
            or self.experiment_history.embedding_dimensions
            != bootstrap_pin.launch_manifest.experiment_embedding_space.dimensions
            or self.experiment_history.embedding_canonicalizer_version
            != bootstrap_pin.launch_manifest.experiment_embedding_space.canonicalizer_version
        ):
            raise RunStateProjectionError(
                "run-state projection belongs to another installed run"
            )

    @classmethod
    def from_bundle(
        cls,
        bundle: RunDerivedStateBundle,
        *,
        strategy_state: RunStrategyState,
        bootstrap_pin: BootstrapPin,
    ) -> "ReconciledRunStateProjection":
        """Decode and cross-reconcile every typed payload in one retained bundle."""
        if type(bundle) is not RunDerivedStateBundle:
            raise RunStateProjectionError(
                "run-state decoding requires one exact retained bundle"
            )
        if (
            type(bootstrap_pin) is not BootstrapPin
            or bundle.generation.bootstrap_pin_id != bootstrap_pin.bootstrap_pin_id
        ):
            raise RunStateProjectionError(
                "run-state bundle belongs to another bootstrap pin"
            )
        payload_by_authority = {
            binding.authority: payload
            for binding, payload in zip(
                bundle.generation.run_state_layout.bindings,
                bundle.payloads,
                strict=True,
            )
        }
        history = ExperimentHistoryProjection.from_json_bytes(
            payload_by_authority[RunStateAuthority.EXPERIMENT_HISTORY]
        )
        journal = ExecutionRevisionProjection.from_jsonl_bytes(
            payload_by_authority[RunStateAuthority.EXECUTION_JOURNAL],
            run_id=history.run_id,
            campaign_id=history.campaign_id,
            require_contiguous_node_ids=history.require_idea_links,
        )
        strategy_kind = bundle.generation.run_state_layout.strategy_kind
        archive = (
            decode_archive_state(payload_by_authority[RunStateAuthority.IDEA_ARCHIVE])
            if strategy_kind == "generic"
            else None
        )
        projection = cls(
            strategy_state=strategy_state,
            experiment_history=history,
            execution_journal=journal,
            idea_archive=archive,
        )
        declared_revisions = {
            binding.authority: transition.target_revision
            for binding, transition in zip(
                bundle.generation.run_state_layout.bindings,
                bundle.generation.payload_transitions,
                strict=True,
            )
        }
        if declared_revisions != projection.revision_by_authority:
            raise RunStateProjectionError(
                "run-state payload revisions differ from their generation"
            )
        projection.require_bootstrap_pin(bootstrap_pin)
        return projection


def _require_event_matches_record(
    event: ExecutionRevisionEvent,
    record: ExperimentRecord,
) -> None:
    expected_execution_status = (
        ExecutionStatus.INTERRUPTED
        if record.recoverable_error
        else (
            ExecutionStatus.FAILED_TECHNICAL
            if record.had_error
            else ExecutionStatus.COMPLETED
        )
    )
    expected_evaluation_status = (
        EpisodeEvaluationStatus.NOT_RUN
        if record.had_error
        else (
            EpisodeEvaluationStatus.INVALID
            if not record.evaluation_valid
            else (
                EpisodeEvaluationStatus.VALID
                if record.raw_score is not None and record.evaluation_attempts
                else EpisodeEvaluationStatus.PARTIAL
            )
        )
    )
    measurements = dict(record.metrics)
    if record.raw_score is not None:
        measurements["raw_score"] = record.raw_score
    evaluator_fingerprint_ids = tuple(
        sorted({attempt.evaluator_id for attempt in record.evaluation_attempts})
    )
    expected_evaluation_commits = {
        f"evaluation_commit_{position}": attempt.commit_sha
        for position, attempt in enumerate(record.evaluation_attempts)
    }
    actual_evaluation_commits = {
        name: value
        for name, value in event.artifact_refs.items()
        if name.startswith("evaluation_commit_")
    }
    attempt_commits = {attempt.commit_sha for attempt in record.evaluation_attempts}
    candidate_commit = event.artifact_refs.get("candidate_commit")
    candidate_ref = event.artifact_refs.get("candidate_ref")
    expected_candidate_ref = (
        f"refs/kapso/execution-revisions/{event.run_id}/"
        f"node-{record.node_id}/revision-{record.execution_revision}"
    )
    if (
        event.started_at != record.timestamp
        or event.execution_status is not expected_execution_status
        or event.evaluation_status is not expected_evaluation_status
        or dict(event.measurements) != measurements
        or event.evaluator_fingerprint_ids != evaluator_fingerprint_ids
        or event.artifact_refs.get("branch") != record.branch_name
        or actual_evaluation_commits != expected_evaluation_commits
        or len(attempt_commits) > 1
        or (
            attempt_commits
            and candidate_commit is not None
            and candidate_commit not in attempt_commits
        )
        or (candidate_ref is not None and candidate_ref != expected_candidate_ref)
    ):
        raise RunStateProjectionError(
            "run-state journal semantics differ from their experiment record"
        )


def _require_event_artifact_lineage(
    events: tuple[ExecutionRevisionEvent, ...],
    records: tuple[ExperimentRecord, ...],
    node_by_id: dict[int, ExperimentNodeProjection],
    archive: IdeaArchiveState | None,
) -> None:
    ideas_by_id = (
        {} if archive is None else {idea.idea_id: idea for idea in archive.ideas}
    )
    prior_candidate_by_node: dict[int, str] = {}
    prior_record_by_node: dict[int, ExperimentRecord] = {}
    for event, record in zip(events, records, strict=True):
        node = node_by_id[record.node_id]
        if archive is None:
            parent_branch = node.parent_branch_name
            implementation_base = node.implementation_base_ref
            diff_base = node.diff_base_ref
            feedback_base = node.feedback_base_ref
            if (
                not parent_branch
                or _GIT_COMMIT_PATTERN.fullmatch(implementation_base) is None
                or diff_base != implementation_base
                or feedback_base != implementation_base
            ):
                raise RunStateProjectionError(
                    "tree strategy node lacks one immutable execution base"
                )
            ancestor = _closest_prior_scored_tree_ancestor(
                node,
                node_by_id,
                prior_record_by_node,
            )
            if ancestor is not None:
                ancestor_record = prior_record_by_node[ancestor.node_id]
                ancestor_commit = prior_candidate_by_node.get(ancestor.node_id)
                if (
                    ancestor_commit is None
                    or parent_branch != ancestor_record.branch_name
                    or implementation_base != ancestor_commit
                    or diff_base != ancestor_commit
                    or feedback_base != ancestor_commit
                ):
                    raise RunStateProjectionError(
                        "tree strategy execution base differs from its closest "
                        "prior scored ancestor"
                    )
        else:
            idea = ideas_by_id[record.idea_id]
            parent = idea.resolved_parent
            if _GIT_COMMIT_PATTERN.fullmatch(parent.git_ref) is None:
                raise RunStateProjectionError(
                    "generic idea parent is not one immutable Git commit"
                )
            if idea.parent_plan.experiment_node_id != parent.node_id:
                raise RunStateProjectionError(
                    "generic idea parent plan differs from its resolved parent"
                )
            parent_branch = parent.branch_name
            implementation_base = (
                parent.git_ref
                if record.execution_revision == 0
                else prior_candidate_by_node.get(record.node_id, "")
            )
            diff_base = parent.diff_base_ref
            feedback_base = parent.feedback_base_ref
            if not implementation_base or (
                record.execution_revision == node.execution_revision
                and (
                    node.parent_branch_name != parent_branch
                    or node.implementation_base_ref != implementation_base
                    or node.diff_base_ref != diff_base
                    or node.feedback_base_ref != feedback_base
                )
            ):
                raise RunStateProjectionError(
                    "strategy node artifact lineage differs from its idea parent"
                )
            if record.execution_revision == 0 and parent.node_id is not None:
                parent_record = prior_record_by_node.get(parent.node_id)
                parent_commit = prior_candidate_by_node.get(parent.node_id)
                if (
                    parent_record is None
                    or parent_commit is None
                    or parent_record.had_error
                    or not parent_record.evaluation_valid
                    or parent_record.raw_score is None
                    or parent_record.branch_name != parent.branch_name
                    or parent_commit != parent.git_ref
                    or idea.parent_plan.kind
                    not in {
                        ParentPlanKind.BEST_VALID,
                        ParentPlanKind.SPECIFIC_EXPERIMENT,
                    }
                ):
                    raise RunStateProjectionError(
                        "generic idea parent differs from its prior valid scored "
                        "candidate"
                    )
        expected_refs = {
            name: value
            for name, value in {
                "branch": record.branch_name,
                "parent_branch": parent_branch,
                "implementation_base": implementation_base,
                "diff_base": diff_base,
                "feedback_base": feedback_base,
            }.items()
            if value
        }
        candidate_commit = event.artifact_refs.get("candidate_commit")
        if candidate_commit is not None:
            if _GIT_COMMIT_PATTERN.fullmatch(candidate_commit) is None:
                raise RunStateProjectionError(
                    "journal candidate commit is not immutable"
                )
            expected_refs["candidate_commit"] = candidate_commit
            expected_refs["candidate_ref"] = (
                f"refs/kapso/execution-revisions/{event.run_id}/"
                f"node-{record.node_id}/revision-{record.execution_revision}"
            )
            for name, base_ref in {
                "implementation_base_commit": implementation_base,
                "diff_base_commit": diff_base,
                "feedback_base_commit": feedback_base,
            }.items():
                if not base_ref:
                    continue
                base_commit = event.artifact_refs.get(name)
                if (
                    base_commit is None
                    or _GIT_COMMIT_PATTERN.fullmatch(base_commit) is None
                    or (
                        _GIT_COMMIT_PATTERN.fullmatch(base_ref) is not None
                        and base_commit != base_ref
                    )
                ):
                    raise RunStateProjectionError(
                        "journal base commit is not the immutable strategy base"
                    )
                expected_refs[name] = base_commit
        elif record.execution_revision > 0 or record.evaluation_attempts:
            raise RunStateProjectionError(
                "journal revision evidence lacks its immutable candidate"
            )
        for position, attempt in enumerate(record.evaluation_attempts):
            if attempt.commit_sha != candidate_commit:
                raise RunStateProjectionError(
                    "journal evaluation is not bound to its candidate"
                )
            expected_refs[f"evaluation_commit_{position}"] = attempt.commit_sha
        if dict(event.artifact_refs) != expected_refs:
            raise RunStateProjectionError(
                "journal artifact refs differ from strategy lineage"
            )
        if candidate_commit is not None:
            prior_candidate_by_node[record.node_id] = candidate_commit
        prior_record_by_node[record.node_id] = record


def _closest_prior_scored_tree_ancestor(
    node: ExperimentNodeProjection,
    node_by_id: dict[int, ExperimentNodeProjection],
    prior_record_by_node: dict[int, ExperimentRecord],
) -> ExperimentNodeProjection | None:
    ancestor_id = node.parent_node_id
    while ancestor_id is not None:
        ancestor = node_by_id[ancestor_id]
        prior_record = prior_record_by_node.get(ancestor_id)
        if (
            prior_record is not None
            and not prior_record.had_error
            and prior_record.evaluation_valid
            and prior_record.raw_score is not None
        ):
            return ancestor
        ancestor_id = ancestor.parent_node_id
    return None


def _require_outcome_matches_record(
    idea_status: IdeaStatus,
    outcome: IdeaOutcome | None,
    record: ExperimentRecord,
    *,
    objective_direction: str,
    parent_record: ExperimentRecord | None,
) -> None:
    if outcome is None:
        raise RunStateProjectionError(
            "terminal experiment record lacks its archive outcome"
        )
    if record.had_error:
        expected_evaluation_status = EvaluationStatus.NOT_RUN
        expected_normalized_delta = None
        expected_validation_tier = None
        matches = (
            idea_status is IdeaStatus.FAILED_TECHNICAL
            and outcome.implementation_status is ImplementationStatus.FAILED_TECHNICAL
        )
    else:
        expected_validation_tier = (
            "full"
            if record.eval_fidelity == "full" and record.build_fidelity == "full"
            else ("validated" if record.eval_fidelity == "full" else "probe")
        )
        if not record.evaluation_valid:
            expected_evaluation_status = EvaluationStatus.INVALID
            expected_normalized_delta = None
        elif record.raw_score is None:
            expected_evaluation_status = EvaluationStatus.INCONCLUSIVE
            expected_normalized_delta = None
        elif parent_record is None:
            expected_evaluation_status = EvaluationStatus.INCONCLUSIVE
            expected_normalized_delta = None
        else:
            normalized_score = _comparable_record_score(record)
            comparison_score = _comparable_record_score(
                parent_record,
                comparability_source=record,
            )
            if normalized_score is None or comparison_score is None:
                expected_evaluation_status = EvaluationStatus.INCONCLUSIVE
                expected_normalized_delta = None
            else:
                sign = 1.0 if objective_direction == "maximize" else -1.0
                expected_evaluation_status = EvaluationStatus.VALID
                expected_normalized_delta = sign * (normalized_score - comparison_score)
        matches = (
            idea_status is IdeaStatus.EVALUATED
            and outcome.implementation_status is ImplementationStatus.COMPLETED
        )
    if (
        not matches
        or outcome.evaluation_status is not expected_evaluation_status
        or outcome.normalized_delta != expected_normalized_delta
        or outcome.validation_tier != expected_validation_tier
        or outcome.actual_cost != record.cost_usd
        or outcome.actual_duration != record.duration_seconds
    ):
        raise RunStateProjectionError(
            "archive outcome semantics differ from their experiment record"
        )


def _require_outcome_parent_record(
    idea: IdeaRecord,
    outcome_record: ExperimentRecord,
    records_before_outcome: dict[int, ExperimentRecord],
) -> ExperimentRecord | None:
    parent_node_id = idea.resolved_parent.node_id
    if parent_node_id is None or parent_node_id == outcome_record.node_id:
        return None
    parent = records_before_outcome.get(parent_node_id)
    if parent is None or parent.had_error or not parent.evaluation_valid:
        raise RunStateProjectionError(
            "archive outcome comparison parent lacks prior valid evidence"
        )
    return parent


def _comparable_record_score(
    record: ExperimentRecord,
    *,
    comparability_source: ExperimentRecord | None = None,
) -> float | None:
    source = record if comparability_source is None else comparability_source
    if not source.evaluation_attempts:
        return None
    comparability = source.evaluation_attempts[-1].comparability_class
    if comparability_source is None and comparability.fidelity != record.eval_fidelity:
        return None
    matching_scores = tuple(
        attempt.score
        for attempt in record.evaluation_attempts
        if attempt.comparability_class == comparability
    )
    if not matching_scores:
        return None
    score = sum(matching_scores) / len(matching_scores)
    if comparability_source is None and score != record.raw_score:
        return None
    return score


__all__ = [
    "ReconciledRunStateProjection",
    "RunStateProjectionError",
]
