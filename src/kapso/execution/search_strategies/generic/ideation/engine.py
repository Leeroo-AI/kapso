"""Single orchestration boundary for one evidence-directed ideation decision."""

import hashlib
import json
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Callable, ContextManager, Mapping, Optional, Sequence, Tuple

from kapso.execution.search_strategies.generic.ideation.analyzer import (
    CandidateAnalyzer,
)
from kapso.execution.search_strategies.generic.ideation.archive import IdeaArchive
from kapso.execution.search_strategies.generic.ideation.evidence import (
    CampaignEvidenceBuilder,
    ExperimentInput,
    GapPrioritySettings,
    rank_evaluation_gaps,
)
from kapso.execution.search_strategies.generic.ideation.generator import (
    CandidateGenerator,
)
from kapso.execution.search_strategies.generic.ideation.operators import (
    OperatorSettings,
    find_resurfaceable_ideas,
    plan_search_directive,
)
from kapso.execution.search_strategies.generic.ideation.policy import choose_policy
from kapso.execution.search_strategies.generic.ideation.selector import (
    CandidateSelector,
)
from kapso.execution.search_strategies.generic.ideation.types import (
    CampaignAction,
    CampaignEvidenceSnapshot,
    CodingAgentCallResult,
    EmbeddingTelemetry,
    GapState,
    IdeaBatch,
    IdeaRecord,
    IdeaStatus,
    IdeationCapacityView,
    ObjectiveDirection,
    OperatorBrief,
    ParentPlan,
    ResolvedParentSnapshot,
    SearchDirective,
    SelectionDecision,
    new_identifier,
    utc_now,
)

ParentResolver = Callable[[ParentPlan], ResolvedParentSnapshot]
ParentMaterializer = Callable[[ResolvedParentSnapshot], ContextManager[str]]


@dataclass(frozen=True)
class IdeationEngineTelemetry:
    generation_calls: Tuple[CodingAgentCallResult, ...]
    selection_call: Optional[CodingAgentCallResult]
    embedding: Optional[EmbeddingTelemetry]

    @property
    def coding_agent_call_count(self) -> int:
        return len(self.generation_calls) + int(self.selection_call is not None)

    @property
    def coding_agent_duration_seconds(self) -> float:
        calls = self.generation_calls + (
            () if self.selection_call is None else (self.selection_call,)
        )
        return sum(call.duration_seconds for call in calls)

    @property
    def known_coding_agent_cost_usd(self) -> float:
        calls = self.generation_calls + (
            () if self.selection_call is None else (self.selection_call,)
        )
        return sum(call.cost_usd for call in calls if call.cost_usd is not None)

    @property
    def unpriced_coding_agent_call_count(self) -> int:
        calls = self.generation_calls + (
            () if self.selection_call is None else (self.selection_call,)
        )
        return sum(call.cost_usd is None for call in calls)


@dataclass(frozen=True)
class IdeationEngineResult:
    action: CampaignAction
    evidence_snapshot: CampaignEvidenceSnapshot
    directive: SearchDirective
    batch_id: Optional[str]
    selected_idea: Optional[IdeaRecord]
    selection: Optional[SelectionDecision]
    resolved_parent: Optional[ResolvedParentSnapshot]
    archive_revision: int
    telemetry: IdeationEngineTelemetry

    def __post_init__(self) -> None:
        if self.action != self.directive.decision.action:
            raise ValueError("engine action must match its search directive")
        if self.action == CampaignAction.FINALIZE:
            if any(
                item is not None
                for item in (
                    self.batch_id,
                    self.selected_idea,
                    self.selection,
                    self.resolved_parent,
                )
            ):
                raise ValueError("finalize results cannot select an experiment")
            return
        if (
            self.batch_id is None
            or self.selected_idea is None
            or self.resolved_parent is None
        ):
            raise ValueError("executable engine results require full provenance")
        if self.selected_idea.selected_in_batch_id != self.batch_id:
            raise ValueError("engine result idea and selection batch do not match")
        if self.action == CampaignAction.IDEATE and self.selection is None:
            raise ValueError("ideation results require a persisted selection")
        if self.action == CampaignAction.RECOVER:
            if self.selection is None:
                raise ValueError("recovery requires the original selection")
            if self.selected_idea.experiment_node_id is None:
                raise ValueError("recovery requires the original experiment node")
            if self.telemetry.coding_agent_call_count != 0:
                raise ValueError("recovery cannot make generation or selection calls")


class IdeationEngine:
    """Run policy, generation, analysis, and selection in archive order."""

    def __init__(
        self,
        *,
        archive: IdeaArchive,
        evidence_builder: CampaignEvidenceBuilder,
        operator_settings: OperatorSettings,
        gap_priority_settings: GapPrioritySettings,
        generator: CandidateGenerator,
        analyzer: CandidateAnalyzer,
        selector: CandidateSelector,
    ):
        self.archive = archive
        self.evidence_builder = evidence_builder
        self.operator_settings = operator_settings
        self.gap_priority_settings = gap_priority_settings
        self.generator = generator
        self.analyzer = analyzer
        self.selector = selector

    def run(
        self,
        *,
        campaign_id: str,
        iteration_index: int,
        problem_statement: str,
        objective_direction: ObjectiveDirection,
        experiments: Sequence[ExperimentInput],
        capacity: IdeationCapacityView,
        selector_workspace: str,
        parent_resolver: ParentResolver,
        parent_materializer: ParentMaterializer,
        generated_at: Optional[str] = None,
        gap_evidence_confidence_by_id: Optional[Mapping[str, float]] = None,
        gap_uncertainty_reduction_by_id: Optional[Mapping[str, float]] = None,
    ) -> IdeationEngineResult:
        if not isinstance(problem_statement, str) or not problem_statement.strip():
            raise ValueError("ideation problem statement must be non-empty")
        if self.archive.campaign_id != campaign_id:
            raise ValueError("engine campaign does not match idea archive")
        timestamp = utc_now() if generated_at is None else generated_at
        archive_state = self.archive.state
        evidence_snapshot = self.evidence_builder.build(
            campaign_id=campaign_id,
            objective_direction=objective_direction,
            experiments=experiments,
            archive_state=archive_state,
            capacity=capacity,
            generated_at=timestamp,
        )
        policy = choose_policy(evidence_snapshot, capacity)
        gap_priorities = rank_evaluation_gaps(
            evidence_snapshot.gaps,
            evidence_confidence_by_gap=(
                {}
                if gap_evidence_confidence_by_id is None
                else gap_evidence_confidence_by_id
            ),
            uncertainty_reduction_by_gap=(
                {}
                if gap_uncertainty_reduction_by_id is None
                else gap_uncertainty_reduction_by_id
            ),
            as_of=timestamp,
            settings=self.gap_priority_settings,
        )
        directive = plan_search_directive(
            policy,
            snapshot=evidence_snapshot,
            capacity=capacity,
            archive_state=archive_state,
            gap_priorities=gap_priorities,
            settings=self.operator_settings,
        )
        empty_telemetry = IdeationEngineTelemetry((), None, None)
        if policy.action == CampaignAction.FINALIZE:
            return IdeationEngineResult(
                action=policy.action,
                evidence_snapshot=evidence_snapshot,
                directive=directive,
                batch_id=None,
                selected_idea=None,
                selection=None,
                resolved_parent=None,
                archive_revision=self.archive.revision,
                telemetry=empty_telemetry,
            )
        if policy.action == CampaignAction.RECOVER:
            return self._recover(
                evidence_snapshot=evidence_snapshot,
                directive=directive,
                parent_resolver=parent_resolver,
            )
        resolved_parents = tuple(
            parent_resolver(brief.parent_plan) for brief in directive.operator_briefs
        )
        resurfaced = self._resurfaceable(evidence_snapshot, archive_state)
        batch_id = new_identifier("batch")
        context_hash = self._context_hash(
            problem_statement=problem_statement,
            evidence_snapshot=evidence_snapshot,
            capacity=capacity,
            directive=directive,
            resolved_parents=resolved_parents,
            archive_revision=archive_state.revision,
        )
        batch = IdeaBatch(
            batch_id=batch_id,
            campaign_id=campaign_id,
            iteration_index=iteration_index,
            context_hash=context_hash,
            evidence_snapshot_id=evidence_snapshot.snapshot_id,
            directive=directive,
            created_at=timestamp,
            updated_at=timestamp,
        )
        self.archive.create_batch(batch, expected_revision=archive_state.revision)
        generation_calls = []
        with ExitStack() as stack:
            workspace_by_ref = {}
            for parent in resolved_parents:
                if parent.materialized_ref not in workspace_by_ref:
                    workspace_by_ref[parent.materialized_ref] = stack.enter_context(
                        parent_materializer(parent)
                    )
            workspaces = tuple(
                workspace_by_ref[parent.materialized_ref] for parent in resolved_parents
            )
            generated = self.generator.generate(
                batch_id=batch_id,
                problem_statement=problem_statement,
                evidence_snapshot=evidence_snapshot,
                directive=directive,
                archive_state=self.archive.state,
                resolved_parents=resolved_parents,
                workspaces=workspaces,
            )
            generation_calls.extend(item.call for item in generated)
            self.archive.add_ideas(
                batch_id,
                (item.idea for item in generated),
                resurfaced_ideas=resurfaced,
                expected_revision=self.archive.revision,
            )
            pool = self._batch_pool(batch_id)
            preliminary_analyzer = CandidateAnalyzer(
                self.analyzer.settings,
                embedding_provider=None,
            )
            preliminary = preliminary_analyzer.analyze_pool(
                batch_id=batch_id,
                candidates=pool,
                archive_state=self.archive.state,
                evidence_snapshot=evidence_snapshot,
                directive=directive,
                capacity=capacity,
            )
            repair_request = preliminary_analyzer.repair_request(
                preliminary,
                directive,
            )
            if repair_request is not None:
                repair_brief = self._repair_brief(
                    directive.operator_briefs,
                    repair_request.missing_descriptor_targets,
                )
                repair_index = directive.operator_briefs.index(repair_brief)
                repair = self.generator.generate_repair(
                    batch_id=batch_id,
                    problem_statement=problem_statement,
                    evidence_snapshot=evidence_snapshot,
                    directive=directive,
                    archive_state=self.archive.state,
                    operator_brief=repair_brief,
                    resolved_parent=resolved_parents[repair_index],
                    repair_request=repair_request.to_dict(),
                    workspace=workspaces[repair_index],
                )
                generation_calls.append(repair.call)
                self.archive.add_repair_idea(
                    batch_id,
                    repair.idea,
                    expected_revision=self.archive.revision,
                )
            pool = self._batch_pool(batch_id)
            analysis_result = self.analyzer.analyze_pool(
                batch_id=batch_id,
                candidates=pool,
                archive_state=self.archive.state,
                evidence_snapshot=evidence_snapshot,
                directive=directive,
                capacity=capacity,
            )
            for analyzed in analysis_result.candidates:
                self.archive.record_analysis(
                    batch_id,
                    analyzed.analysis,
                    embedding=analyzed.embedding,
                    nearest_experiment_node_ids=(analyzed.nearest_experiment_node_ids),
                    similarity_flags=analyzed.similarity_flags,
                    expected_revision=self.archive.revision,
                )
            pool = self._batch_pool(batch_id)
            selection_result = self.selector.select(
                problem_statement=problem_statement,
                evidence_snapshot=evidence_snapshot,
                directive=directive,
                candidates=pool,
                analyses=tuple(
                    analyzed.analysis for analyzed in analysis_result.candidates
                ),
                workspace=selector_workspace,
            )
            self.archive.record_selection(
                batch_id,
                selection_result.decision,
                expected_revision=self.archive.revision,
            )
        selected = self._idea(selection_result.decision.selected_idea_id)
        return IdeationEngineResult(
            action=policy.action,
            evidence_snapshot=evidence_snapshot,
            directive=directive,
            batch_id=batch_id,
            selected_idea=selected,
            selection=selection_result.decision,
            resolved_parent=selected.resolved_parent,
            archive_revision=self.archive.revision,
            telemetry=IdeationEngineTelemetry(
                generation_calls=tuple(generation_calls),
                selection_call=selection_result.call,
                embedding=analysis_result.embedding_telemetry,
            ),
        )

    def _recover(
        self,
        *,
        evidence_snapshot: CampaignEvidenceSnapshot,
        directive: SearchDirective,
        parent_resolver: ParentResolver,
    ) -> IdeationEngineResult:
        latest_node_id = evidence_snapshot.latest_node_id
        if latest_node_id is None:
            raise ValueError("recovery requires a latest experiment")
        latest = next(
            experiment
            for experiment in evidence_snapshot.experiments
            if experiment.node_id == latest_node_id
        )
        idea = self._idea(latest.idea_id)
        if idea.status != IdeaStatus.IMPLEMENTING or idea.outcome is not None:
            raise ValueError("recovery requires an unfinished implementing idea")
        if idea.experiment_node_id != latest_node_id:
            raise ValueError("recovery must preserve the original experiment node")
        batch_id = idea.selected_in_batch_id
        if batch_id is None:
            raise ValueError("recovery idea has no selection batch")
        batch = next(
            batch for batch in self.archive.state.batches if batch.batch_id == batch_id
        )
        if batch.selection is None or batch.selection.selected_idea_id != idea.idea_id:
            raise ValueError("recovery selection provenance is invalid")
        resolved_parent = parent_resolver(directive.operator_briefs[0].parent_plan)
        if resolved_parent.node_id != latest_node_id:
            raise ValueError("recovery parent must be the original experiment node")
        return IdeationEngineResult(
            action=CampaignAction.RECOVER,
            evidence_snapshot=evidence_snapshot,
            directive=directive,
            batch_id=batch_id,
            selected_idea=idea,
            selection=batch.selection,
            resolved_parent=resolved_parent,
            archive_revision=self.archive.revision,
            telemetry=IdeationEngineTelemetry((), None, None),
        )

    def _resurfaceable(self, snapshot, archive_state):
        changed_conditions = {}
        for idea in archive_state.ideas:
            if idea.status != IdeaStatus.DEFERRED:
                continue
            prior_batches = tuple(
                batch
                for batch in archive_state.batches
                if idea.idea_id in batch.considered_idea_ids
            )
            if not prior_batches:
                raise ValueError("deferred idea has no prior consideration batch")
            last_batch = max(
                prior_batches,
                key=lambda batch: (batch.iteration_index, batch.updated_at),
            )
            if last_batch.evidence_snapshot_id != snapshot.snapshot_id:
                changed_conditions[idea.idea_id] = (
                    "evidence_snapshot_changed:"
                    f"{last_batch.evidence_snapshot_id}->{snapshot.snapshot_id}",
                )
        closed_gap_ids = tuple(
            gap.gap_id for gap in snapshot.gaps if gap.state == GapState.CLOSED
        )
        return find_resurfaceable_ideas(
            archive_state.ideas,
            changed_conditions_by_idea=changed_conditions,
            newly_resolved_gap_ids=closed_gap_ids,
        )

    def _batch_pool(self, batch_id: str) -> Tuple[IdeaRecord, ...]:
        state = self.archive.state
        batch = next(batch for batch in state.batches if batch.batch_id == batch_id)
        idea_by_id = {idea.idea_id: idea for idea in state.ideas}
        return tuple(idea_by_id[idea_id] for idea_id in batch.considered_idea_ids)

    def _idea(self, idea_id: str) -> IdeaRecord:
        return next(
            idea for idea in self.archive.state.ideas if idea.idea_id == idea_id
        )

    @staticmethod
    def _repair_brief(
        briefs: Tuple[OperatorBrief, ...],
        missing_descriptors,
    ) -> OperatorBrief:
        missing = set(missing_descriptors)
        matching = tuple(
            brief for brief in briefs if brief.descriptor_target in missing
        )
        if not matching:
            raise ValueError("diversity repair has no assigned missing descriptor")
        return matching[0]

    @staticmethod
    def _context_hash(
        *,
        problem_statement: str,
        evidence_snapshot: CampaignEvidenceSnapshot,
        capacity: IdeationCapacityView,
        directive: SearchDirective,
        resolved_parents: Tuple[ResolvedParentSnapshot, ...],
        archive_revision: int,
    ) -> str:
        content = {
            "problem_statement": problem_statement,
            "evidence_snapshot": evidence_snapshot.to_dict(),
            "capacity": capacity.to_dict(),
            "directive": directive.to_dict(),
            "resolved_parents": [parent.to_dict() for parent in resolved_parents],
            "archive_revision": archive_revision,
        }
        return hashlib.sha256(
            json.dumps(
                content,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
