"""Deterministic operator allocation, parent intent, and resurfacing."""

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Tuple

from kapso.execution.search_strategies.generic.ideation.archive import (
    IdeaArchiveState,
)
from kapso.execution.search_strategies.generic.ideation.evidence import GapPriority
from kapso.execution.search_strategies.generic.ideation.types import (
    CampaignAction,
    CampaignEvidenceSnapshot,
    ClaimKind,
    EvidenceStatus,
    GapState,
    IdeaDescriptor,
    IdeaRecord,
    IdeaStatus,
    IdeationCapacityView,
    IdeationMode,
    OperatorBrief,
    OperatorKind,
    ParentPlan,
    ParentPlanKind,
    PolicyDecision,
    ResurfacedIdea,
    SearchDirective,
)


@dataclass(frozen=True)
class OperatorSettings:
    candidate_quota: int
    repair_quota: int
    reserve_gap_slot: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.candidate_quota, bool)
            or not isinstance(self.candidate_quota, int)
            or self.candidate_quota < 1
        ):
            raise ValueError("operator candidate quota must be positive")
        if (
            isinstance(self.repair_quota, bool)
            or not isinstance(self.repair_quota, int)
            or self.repair_quota not in {0, 1}
        ):
            raise ValueError("operator repair quota must be zero or one")
        if not isinstance(self.reserve_gap_slot, bool):
            raise ValueError("operator gap reservation must be boolean")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OperatorSettings":
        expected = {"candidate_quota", "repair_quota", "reserve_gap_slot"}
        if not isinstance(data, Mapping) or set(data) != expected:
            raise ValueError("operator settings fields are invalid")
        return cls(**data)


def _experiment_by_node(
    snapshot: CampaignEvidenceSnapshot,
    node_id: int,
):
    matching = tuple(
        experiment
        for experiment in snapshot.experiments
        if experiment.node_id == node_id
    )
    if not matching:
        raise ValueError(f"evidence snapshot has no experiment node {node_id}")
    return matching[0]


def _idea_by_id(archive_state: IdeaArchiveState, idea_id: str) -> IdeaRecord:
    matching = tuple(idea for idea in archive_state.ideas if idea.idea_id == idea_id)
    if not matching:
        raise ValueError(f"idea archive has no idea {idea_id}")
    return matching[0]


def _incumbent_idea(
    snapshot: CampaignEvidenceSnapshot,
    archive_state: IdeaArchiveState,
):
    if snapshot.incumbent_node_id is None:
        return None
    experiment = _experiment_by_node(snapshot, snapshot.incumbent_node_id)
    return _idea_by_id(archive_state, experiment.idea_id)


def _latest_idea(
    snapshot: CampaignEvidenceSnapshot,
    archive_state: IdeaArchiveState,
):
    if snapshot.latest_node_id is None:
        return None
    experiment = _experiment_by_node(snapshot, snapshot.latest_node_id)
    return _idea_by_id(archive_state, experiment.idea_id)


def _operator_palette(mode: IdeationMode) -> Tuple[OperatorKind, ...]:
    palettes = {
        IdeationMode.BOOTSTRAP: (
            OperatorKind.INDEPENDENT_DRAFT,
            OperatorKind.MECHANISM_SHIFT,
            OperatorKind.TARGET_GAP,
        ),
        IdeationMode.EXPLOIT: (
            OperatorKind.ATOMIC_REFINE,
            OperatorKind.CROSSOVER,
            OperatorKind.ABLATE,
            OperatorKind.TARGET_GAP,
        ),
        IdeationMode.EXPLORE: (
            OperatorKind.MECHANISM_SHIFT,
            OperatorKind.TARGET_GAP,
            OperatorKind.INDEPENDENT_DRAFT,
            OperatorKind.CROSSOVER,
        ),
        IdeationMode.VERIFY: (OperatorKind.VERIFY,),
        IdeationMode.RECOVER: (OperatorKind.RECOVER,),
    }
    return palettes[mode]


def _distinct_crossover_sources(
    snapshot: CampaignEvidenceSnapshot,
    archive_state: IdeaArchiveState,
    incumbent: IdeaRecord,
) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    supported_hypothesis_idea_ids = {
        idea_id
        for claim in snapshot.claims
        if claim.kind == ClaimKind.HYPOTHESIS
        and claim.status == EvidenceStatus.SUPPORTED
        for idea_id in claim.affected_idea_ids
    }
    source_ideas = []
    source_nodes = []
    for experiment in sorted(
        snapshot.experiments,
        key=lambda item: (
            -(
                item.normalized_utility
                if item.normalized_utility is not None
                else float("-inf")
            ),
            item.node_id,
        ),
    ):
        idea = _idea_by_id(archive_state, experiment.idea_id)
        if (
            experiment.comparable
            and experiment.idea_id in supported_hypothesis_idea_ids
            and idea.idea_id != incumbent.idea_id
            and idea.descriptor.approach_family != incumbent.descriptor.approach_family
        ):
            source_ideas.append(idea.idea_id)
            source_nodes.append(experiment.node_id)
            break
    return tuple(source_ideas), tuple(source_nodes)


def _brief(
    operator: OperatorKind,
    *,
    snapshot: CampaignEvidenceSnapshot,
    capacity: IdeationCapacityView,
    archive_state: IdeaArchiveState,
    reserved_gap_id: str | None,
) -> OperatorBrief | None:
    incumbent = _incumbent_idea(snapshot, archive_state)
    latest = _latest_idea(snapshot, archive_state)
    if operator == OperatorKind.INDEPENDENT_DRAFT:
        return OperatorBrief(
            operator=operator,
            rationale="Draft an independent task-grounded mechanism from the immutable baseline.",
            descriptor_target=IdeaDescriptor(
                approach_family="independent_solution",
                intervention_target="end_to_end_solution",
                mechanism="task_evidence_synthesis",
                expected_effect="establish_comparable_utility",
            ),
            parent_plan=ParentPlan(kind=ParentPlanKind.BASELINE),
        )
    if operator == OperatorKind.TARGET_GAP:
        if reserved_gap_id is None:
            return None
        gap = next(gap for gap in snapshot.gaps if gap.gap_id == reserved_gap_id)
        return OperatorBrief(
            operator=operator,
            rationale=f"Reduce the highest-priority uncertainty on {gap.axis}.",
            descriptor_target=IdeaDescriptor(
                approach_family="evaluation_gap_intervention",
                intervention_target=gap.axis,
                mechanism="decisive_gap_test",
                expected_effect="reduce_measured_uncertainty",
            ),
            parent_plan=ParentPlan(
                kind=(
                    ParentPlanKind.BEST_VALID
                    if incumbent is not None
                    else ParentPlanKind.BASELINE
                ),
                experiment_node_id=(
                    snapshot.incumbent_node_id if incumbent is not None else None
                ),
            ),
            target_gap_id=gap.gap_id,
        )
    if operator == OperatorKind.MECHANISM_SHIFT:
        current_family = (
            "unmeasured" if incumbent is None else incumbent.descriptor.approach_family
        )
        return OperatorBrief(
            operator=operator,
            rationale="Change mechanism family instead of rewording the current approach.",
            descriptor_target=IdeaDescriptor(
                approach_family=f"alternative_to_{current_family}",
                intervention_target="dominant_failure_mode",
                mechanism="orthogonal_mechanism",
                expected_effect="escape_current_plateau",
            ),
            parent_plan=ParentPlan(kind=ParentPlanKind.BASELINE),
        )
    if operator == OperatorKind.ATOMIC_REFINE:
        if incumbent is None:
            return None
        return OperatorBrief(
            operator=operator,
            rationale="Change one supported lever around the delivery incumbent.",
            descriptor_target=IdeaDescriptor(
                approach_family=incumbent.descriptor.approach_family,
                intervention_target=incumbent.descriptor.intervention_target,
                mechanism=f"atomic_refine_{incumbent.descriptor.mechanism}",
                expected_effect="increase_supported_effect",
            ),
            parent_plan=ParentPlan(
                kind=ParentPlanKind.BEST_VALID,
                experiment_node_id=snapshot.incumbent_node_id,
            ),
        )
    if operator == OperatorKind.ABLATE:
        if incumbent is None or snapshot.incumbent_node_id is None:
            return None
        return OperatorBrief(
            operator=operator,
            rationale="Isolate the incumbent mechanism's causal contribution.",
            descriptor_target=IdeaDescriptor(
                approach_family=incumbent.descriptor.approach_family,
                intervention_target=incumbent.descriptor.intervention_target,
                mechanism=f"ablate_{incumbent.descriptor.mechanism}",
                expected_effect="measure_causal_contribution",
            ),
            parent_plan=ParentPlan(
                kind=ParentPlanKind.SPECIFIC_EXPERIMENT,
                experiment_node_id=snapshot.incumbent_node_id,
            ),
        )
    if operator == OperatorKind.CROSSOVER:
        if incumbent is None:
            return None
        source_idea_ids, source_node_ids = _distinct_crossover_sources(
            snapshot,
            archive_state,
            incumbent,
        )
        if not source_idea_ids:
            return None
        source = _idea_by_id(archive_state, source_idea_ids[0])
        return OperatorBrief(
            operator=operator,
            rationale="Transfer one compatible mechanism from a distinct measured lineage.",
            descriptor_target=IdeaDescriptor(
                approach_family=f"{incumbent.descriptor.approach_family}_crossover",
                intervention_target=incumbent.descriptor.intervention_target,
                mechanism=f"compose_{source.descriptor.mechanism}",
                expected_effect="combine_distinct_supported_signal",
            ),
            parent_plan=ParentPlan(
                kind=ParentPlanKind.BEST_VALID,
                experiment_node_id=snapshot.incumbent_node_id,
                source_idea_ids=source_idea_ids,
                source_experiment_node_ids=source_node_ids,
            ),
        )
    if operator == OperatorKind.VERIFY:
        target_node_id = capacity.target_node_id or snapshot.incumbent_node_id
        if target_node_id is None:
            raise ValueError("verification requires a target experiment")
        target = _idea_by_id(
            archive_state,
            _experiment_by_node(snapshot, target_node_id).idea_id,
        )
        return OperatorBrief(
            operator=operator,
            rationale="Raise fidelity or replicate without changing the hypothesis.",
            descriptor_target=target.descriptor,
            parent_plan=ParentPlan(
                kind=ParentPlanKind.SPECIFIC_EXPERIMENT,
                experiment_node_id=target_node_id,
                source_idea_ids=(target.idea_id,),
                source_experiment_node_ids=(target_node_id,),
            ),
        )
    if latest is None or snapshot.latest_node_id is None:
        raise ValueError("recovery requires the latest failed experiment")
    return OperatorBrief(
        operator=OperatorKind.RECOVER,
        rationale="Complete the same intervention on its failed branch before judging it.",
        descriptor_target=latest.descriptor,
        parent_plan=ParentPlan(
            kind=ParentPlanKind.RECOVER_BRANCH,
            experiment_node_id=snapshot.latest_node_id,
            source_idea_ids=(latest.idea_id,),
            source_experiment_node_ids=(snapshot.latest_node_id,),
        ),
    )


def plan_search_directive(
    decision: PolicyDecision,
    *,
    snapshot: CampaignEvidenceSnapshot,
    capacity: IdeationCapacityView,
    archive_state: IdeaArchiveState,
    gap_priorities: Iterable[GapPriority],
    settings: OperatorSettings,
) -> SearchDirective:
    if decision.action == CampaignAction.FINALIZE:
        return SearchDirective(
            decision=decision,
            evidence_snapshot_id=snapshot.snapshot_id,
            capacity_snapshot_id=capacity.capacity_snapshot_id,
            operator_briefs=(),
            candidate_quota=0,
            repair_quota=0,
            validation_requirements=(),
            allowed_parent_plan_kinds=(),
            terminal_constraints=("deliver the best delivery-grade incumbent",),
        )
    if decision.action == CampaignAction.RECOVER:
        recovery_brief = _brief(
            OperatorKind.RECOVER,
            snapshot=snapshot,
            capacity=capacity,
            archive_state=archive_state,
            reserved_gap_id=None,
        )
        if recovery_brief is None:
            raise ValueError("recovery action requires a recoverable experiment")
        return SearchDirective(
            decision=decision,
            evidence_snapshot_id=snapshot.snapshot_id,
            capacity_snapshot_id=capacity.capacity_snapshot_id,
            operator_briefs=(recovery_brief,),
            candidate_quota=0,
            repair_quota=0,
            validation_requirements=(
                f"evaluator:{capacity.eval_fidelity}",
                f"fraction:{capacity.eval_fraction}",
            ),
            allowed_parent_plan_kinds=(ParentPlanKind.RECOVER_BRANCH,),
            terminal_constraints=("resume the same idea and experiment node",),
        )
    if decision.mode is None:
        raise ValueError("ideation decisions require a mode")
    actionable_gaps = tuple(
        priority
        for priority in gap_priorities
        if priority.score > 0
        and any(
            gap.gap_id == priority.gap_id and gap.state == GapState.OPEN
            for gap in snapshot.gaps
        )
    )
    reserved_gap_id = (
        actionable_gaps[0].gap_id
        if settings.reserve_gap_slot and actionable_gaps
        else None
    )
    palette = list(_operator_palette(decision.mode))
    if reserved_gap_id is not None and OperatorKind.TARGET_GAP in palette:
        palette.remove(OperatorKind.TARGET_GAP)
        palette.insert(0, OperatorKind.TARGET_GAP)
    quota = (
        1
        if decision.mode in {IdeationMode.RECOVER, IdeationMode.VERIFY}
        else settings.candidate_quota
    )
    briefs = []
    for operator in palette:
        brief = _brief(
            operator,
            snapshot=snapshot,
            capacity=capacity,
            archive_state=archive_state,
            reserved_gap_id=reserved_gap_id,
        )
        if brief is not None:
            briefs.append(brief)
        if len(briefs) == quota:
            break
    if len(briefs) != quota:
        raise ValueError(
            "operator settings request more distinct briefs than available"
        )
    descriptors = tuple(brief.descriptor_target for brief in briefs)
    if len(set(descriptors)) != len(descriptors) and decision.mode not in {
        IdeationMode.RECOVER,
        IdeationMode.VERIFY,
    }:
        raise ValueError("operator allocation did not produce distinct descriptors")
    allowed_parent_kinds = tuple(
        dict.fromkeys(brief.parent_plan.kind for brief in briefs)
    )
    repair_quota = (
        0
        if decision.mode in {IdeationMode.RECOVER, IdeationMode.VERIFY}
        else settings.repair_quota
    )
    return SearchDirective(
        decision=decision,
        evidence_snapshot_id=snapshot.snapshot_id,
        capacity_snapshot_id=capacity.capacity_snapshot_id,
        operator_briefs=tuple(briefs),
        candidate_quota=quota,
        repair_quota=repair_quota,
        validation_requirements=(
            f"evaluator:{capacity.eval_fidelity}",
            f"fraction:{capacity.eval_fraction}",
        ),
        allowed_parent_plan_kinds=allowed_parent_kinds,
        terminal_constraints=("preserve finalization reserve",),
        reserved_gap_id=reserved_gap_id,
    )


def find_resurfaceable_ideas(
    ideas: Iterable[IdeaRecord],
    *,
    changed_conditions_by_idea: Mapping[str, Iterable[str]],
    newly_resolved_gap_ids: Iterable[str],
) -> Tuple[ResurfacedIdea, ...]:
    resolved_gap_ids = set(newly_resolved_gap_ids)
    resurfaced = []
    for idea in ideas:
        if (
            idea.status != IdeaStatus.DEFERRED
            or idea.experiment_node_id is not None
            or idea.outcome is not None
        ):
            continue
        reasons = tuple(changed_conditions_by_idea.get(idea.idea_id, ()))
        resolved_targets = tuple(
            gap_id for gap_id in idea.target_gap_ids if gap_id in resolved_gap_ids
        )
        reasons += tuple(f"resolved_gap:{gap_id}" for gap_id in resolved_targets)
        if reasons:
            resurfaced.append(
                ResurfacedIdea(
                    idea_id=idea.idea_id,
                    changed_conditions=reasons,
                )
            )
    return tuple(resurfaced)
