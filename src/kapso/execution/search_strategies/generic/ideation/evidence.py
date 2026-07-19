"""Deterministic campaign evidence and evaluation-gap prioritization."""

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from kapso.execution.search_strategies.generic.ideation.archive import (
    IdeaArchiveState,
)
from kapso.execution.search_strategies.generic.ideation.types import (
    CampaignEvidenceSnapshot,
    EvaluationGap,
    EvaluationStatus,
    EvidenceSignal,
    EvidenceStatus,
    ExperimentEvidence,
    GapState,
    IdeationCapacityView,
    ImplementationStatus,
    JsonRecord,
    ObjectiveDirection,
    content_identifier,
)


@dataclass(frozen=True)
class EvaluationAttemptInput(JsonRecord):
    evaluator_id: str
    fidelity: str
    fraction: float
    seed: int
    score: float
    duration_seconds: Optional[float]

    def __post_init__(self) -> None:
        if not isinstance(self.evaluator_id, str) or not self.evaluator_id.strip():
            raise ValueError("evaluation attempt evaluator must be non-empty")
        if not isinstance(self.fidelity, str) or not self.fidelity.strip():
            raise ValueError("evaluation attempt fidelity must be non-empty")
        for value, name in (
            (self.fraction, "fraction"),
            (self.score, "score"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"evaluation attempt {name} must be finite")
        if not 0.0 < float(self.fraction) <= 1.0:
            raise ValueError("evaluation attempt fraction must be in (0, 1]")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("evaluation attempt seed must be an integer")
        if self.duration_seconds is not None and (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, (int, float))
            or not math.isfinite(float(self.duration_seconds))
            or self.duration_seconds < 0
        ):
            raise ValueError("evaluation attempt duration must be non-negative")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationAttemptInput":
        expected = {
            "evaluator_id",
            "fidelity",
            "fraction",
            "seed",
            "score",
            "duration_seconds",
        }
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("evaluation attempt fields are invalid")
        return cls(**data)


@dataclass(frozen=True)
class ExperimentInput(JsonRecord):
    node_id: int
    idea_id: str
    selection_batch_id: str
    parent_node_id: Optional[int]
    proposal: str
    score: Optional[float]
    evaluation_valid: bool
    had_error: bool
    recoverable_error: bool
    build_fidelity: str
    attempts: Tuple[EvaluationAttemptInput, ...]
    feedback: str
    technical_difficulty: Optional[str]
    created_at: str

    def __post_init__(self) -> None:
        if isinstance(self.node_id, bool) or not isinstance(self.node_id, int):
            raise ValueError("experiment input node id must be an integer")
        if self.node_id < 0:
            raise ValueError("experiment input node id must be non-negative")
        if self.parent_node_id is not None and (
            isinstance(self.parent_node_id, bool)
            or not isinstance(self.parent_node_id, int)
            or self.parent_node_id < 0
        ):
            raise ValueError("experiment input parent node must be non-negative")
        for value, name in (
            (self.idea_id, "idea id"),
            (self.selection_batch_id, "selection batch id"),
            (self.proposal, "proposal"),
            (self.build_fidelity, "build fidelity"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"experiment input {name} must be non-empty")
        if self.score is not None and (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(float(self.score))
        ):
            raise ValueError("experiment input score must be finite")
        for value, name in (
            (self.evaluation_valid, "evaluation validity"),
            (self.had_error, "error status"),
            (self.recoverable_error, "recovery status"),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"experiment input {name} must be boolean")
        if not isinstance(self.attempts, (list, tuple)) or not all(
            isinstance(attempt, EvaluationAttemptInput) for attempt in self.attempts
        ):
            raise ValueError("experiment input attempts are invalid")
        object.__setattr__(self, "attempts", tuple(self.attempts))
        if not isinstance(self.feedback, str):
            raise ValueError("experiment input feedback must be a string")
        if self.technical_difficulty is not None and (
            not isinstance(self.technical_difficulty, str)
            or not self.technical_difficulty.strip()
        ):
            raise ValueError("technical difficulty must be non-empty when present")
        timestamp = datetime.fromisoformat(self.created_at)
        if timestamp.utcoffset() is None:
            raise ValueError("experiment input timestamp must include a UTC offset")
        if self.had_error and (self.score is not None or self.attempts):
            raise ValueError("technical failures cannot contain evaluation evidence")
        if self.recoverable_error and not self.had_error:
            raise ValueError("only technical failures may be recoverable")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentInput":
        expected = {
            "node_id",
            "idea_id",
            "selection_batch_id",
            "parent_node_id",
            "proposal",
            "score",
            "evaluation_valid",
            "had_error",
            "recoverable_error",
            "build_fidelity",
            "attempts",
            "feedback",
            "technical_difficulty",
            "created_at",
        }
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("experiment input fields are invalid")
        return cls(
            node_id=data["node_id"],
            idea_id=data["idea_id"],
            selection_batch_id=data["selection_batch_id"],
            parent_node_id=data["parent_node_id"],
            proposal=data["proposal"],
            score=data["score"],
            evaluation_valid=data["evaluation_valid"],
            had_error=data["had_error"],
            recoverable_error=data["recoverable_error"],
            build_fidelity=data["build_fidelity"],
            attempts=tuple(
                EvaluationAttemptInput.from_dict(attempt)
                for attempt in data["attempts"]
            ),
            feedback=data["feedback"],
            technical_difficulty=data["technical_difficulty"],
            created_at=data["created_at"],
        )


@dataclass(frozen=True)
class EvidenceSettings:
    evaluator_id: str
    comparable_fidelity: str
    comparable_fraction: float
    comparable_seed: int
    minimum_repeat_measurements: int
    minimum_credible_delta: float
    surprising_gain_multiplier: float
    proxy_divergence_threshold: float
    plateau_window: int
    diversity_window: int
    gap_debt_threshold: int

    def __post_init__(self) -> None:
        if not isinstance(self.evaluator_id, str) or not self.evaluator_id.strip():
            raise ValueError("evidence evaluator id must be non-empty")
        if (
            not isinstance(self.comparable_fidelity, str)
            or not self.comparable_fidelity.strip()
        ):
            raise ValueError("comparable fidelity must be non-empty")
        if (
            isinstance(self.comparable_fraction, bool)
            or not isinstance(self.comparable_fraction, (int, float))
            or not math.isfinite(float(self.comparable_fraction))
            or not 0.0 < self.comparable_fraction <= 1.0
        ):
            raise ValueError("comparable fraction must be in (0, 1]")
        if isinstance(self.comparable_seed, bool) or not isinstance(
            self.comparable_seed, int
        ):
            raise ValueError("comparable seed must be an integer")
        for value, name, minimum in (
            (self.minimum_repeat_measurements, "minimum repeats", 2),
            (self.plateau_window, "plateau window", 2),
            (self.diversity_window, "diversity window", 2),
            (self.gap_debt_threshold, "gap debt threshold", 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"evidence {name} must be >= {minimum}")
        for value, name in (
            (self.minimum_credible_delta, "minimum credible delta"),
            (self.proxy_divergence_threshold, "proxy divergence threshold"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"evidence {name} must be non-negative")
        if (
            isinstance(self.surprising_gain_multiplier, bool)
            or not isinstance(self.surprising_gain_multiplier, (int, float))
            or not math.isfinite(float(self.surprising_gain_multiplier))
            or self.surprising_gain_multiplier < 1.0
        ):
            raise ValueError("surprising gain multiplier must be >= 1")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceSettings":
        expected = {
            "evaluator_id",
            "comparable_fidelity",
            "comparable_fraction",
            "comparable_seed",
            "minimum_repeat_measurements",
            "minimum_credible_delta",
            "surprising_gain_multiplier",
            "proxy_divergence_threshold",
            "plateau_window",
            "diversity_window",
            "gap_debt_threshold",
        }
        if not isinstance(data, Mapping) or set(data) != expected:
            raise ValueError("evidence settings fields are invalid")
        return cls(**data)


@dataclass(frozen=True)
class GapPriority(JsonRecord):
    gap_id: str
    score: float
    evidence_confidence: float
    uncertainty_reduction: float
    cost: float
    deferral_count: int
    age_seconds: float
    assumptions: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.gap_id, str) or not self.gap_id.startswith("gap_"):
            raise ValueError("gap priority id must be a typed gap identifier")
        for value, name in (
            (self.score, "score"),
            (self.evidence_confidence, "evidence confidence"),
            (self.uncertainty_reduction, "uncertainty reduction"),
            (self.cost, "cost"),
            (self.age_seconds, "age"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"gap priority {name} must be non-negative")
        if self.evidence_confidence > 1 or self.uncertainty_reduction > 1:
            raise ValueError("gap priority probabilities must be <= 1")
        if self.cost == 0:
            raise ValueError("gap priority cost must be greater than zero")
        if (
            isinstance(self.deferral_count, bool)
            or not isinstance(self.deferral_count, int)
            or self.deferral_count < 0
        ):
            raise ValueError("gap priority deferral count must be non-negative")
        if not isinstance(self.assumptions, (list, tuple)) or not all(
            isinstance(assumption, str) and assumption.strip()
            for assumption in self.assumptions
        ):
            raise ValueError("gap priority assumptions must be strings")
        if len(set(self.assumptions)) != len(self.assumptions):
            raise ValueError("gap priority assumptions must be unique")
        object.__setattr__(self, "assumptions", tuple(self.assumptions))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GapPriority":
        expected = {
            "gap_id",
            "score",
            "evidence_confidence",
            "uncertainty_reduction",
            "cost",
            "deferral_count",
            "age_seconds",
            "assumptions",
        }
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("gap priority fields are invalid")
        return cls(**data)


@dataclass(frozen=True)
class GapPrioritySettings:
    default_evidence_confidence: float
    default_uncertainty_reduction: float
    default_cost: float
    minimum_cost: float

    def __post_init__(self) -> None:
        for value, name in (
            (self.default_evidence_confidence, "default evidence confidence"),
            (
                self.default_uncertainty_reduction,
                "default uncertainty reduction",
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"gap {name} must be in [0, 1]")
        for value, name in (
            (self.default_cost, "default cost"),
            (self.minimum_cost, "minimum cost"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"gap {name} must be greater than zero")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GapPrioritySettings":
        expected = {
            "default_evidence_confidence",
            "default_uncertainty_reduction",
            "default_cost",
            "minimum_cost",
        }
        if not isinstance(data, Mapping) or set(data) != expected:
            raise ValueError("gap priority settings fields are invalid")
        return cls(**data)


class CampaignEvidenceBuilder:
    """Build a content-addressed evidence snapshot without mutating inputs."""

    def __init__(self, settings: EvidenceSettings):
        self.settings = settings

    def build(
        self,
        *,
        campaign_id: str,
        objective_direction: ObjectiveDirection,
        experiments: Iterable[ExperimentInput],
        archive_state: IdeaArchiveState,
        capacity: IdeationCapacityView,
        generated_at: str,
    ) -> CampaignEvidenceSnapshot:
        if archive_state.campaign_id != campaign_id:
            raise ValueError("evidence campaign does not match idea archive")
        experiment_inputs = tuple(
            sorted(
                experiments,
                key=lambda item: (item.created_at, item.node_id),
            )
        )
        if len({item.node_id for item in experiment_inputs}) != len(experiment_inputs):
            raise ValueError("experiment input node ids must be unique")
        archive_idea_ids = {idea.idea_id for idea in archive_state.ideas}
        archive_batch_ids = {batch.batch_id for batch in archive_state.batches}
        idea_by_id = {idea.idea_id: idea for idea in archive_state.ideas}
        for experiment in experiment_inputs:
            if experiment.idea_id not in archive_idea_ids:
                raise ValueError("experiment input references an unknown idea")
            if experiment.selection_batch_id not in archive_batch_ids:
                raise ValueError("experiment input references an unknown batch")
            linked_idea = idea_by_id[experiment.idea_id]
            if (
                linked_idea.experiment_node_id != experiment.node_id
                or linked_idea.selected_in_batch_id != experiment.selection_batch_id
            ):
                raise ValueError("experiment input conflicts with idea archive links")
        projected = tuple(
            self._project_experiment(experiment, objective_direction)
            for experiment in experiment_inputs
        )
        comparable = tuple(item for item in projected if item.comparable)
        incumbent = (
            max(comparable, key=lambda item: item.normalized_utility)
            if comparable
            else None
        )
        latest = projected[-1] if projected else None
        noise_floor = self._noise_floor(experiment_inputs)
        relevant_idea_ids = self._relevant_idea_ids(
            archive_state,
            incumbent,
            latest,
        )
        signals = self._signals(
            experiment_inputs=experiment_inputs,
            projected=projected,
            archive_state=archive_state,
            capacity=capacity,
            incumbent=incumbent,
            latest=latest,
            noise_floor=noise_floor,
        )
        claims = tuple(sorted(archive_state.claims, key=lambda claim: claim.claim_id))
        gaps = tuple(sorted(archive_state.gaps, key=lambda gap: gap.gap_id))
        content = {
            "campaign_id": campaign_id,
            "objective_direction": objective_direction.value,
            "experiments": [item.to_dict() for item in projected],
            "claims": [claim.to_dict() for claim in claims],
            "gaps": [gap.to_dict() for gap in gaps],
            "relevant_idea_ids": list(relevant_idea_ids),
            "incumbent_node_id": None if incumbent is None else incumbent.node_id,
            "latest_node_id": None if latest is None else latest.node_id,
            "noise_floor": noise_floor,
            "signals": [signal.value for signal in signals],
        }
        content_hash = hashlib.sha256(
            json.dumps(
                content,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        return CampaignEvidenceSnapshot(
            snapshot_id=content_identifier("evidence_snapshot", content_hash),
            campaign_id=campaign_id,
            objective_direction=objective_direction,
            generated_at=generated_at,
            content_hash=content_hash,
            experiments=projected,
            claims=claims,
            gaps=gaps,
            relevant_idea_ids=relevant_idea_ids,
            incumbent_node_id=None if incumbent is None else incumbent.node_id,
            latest_node_id=None if latest is None else latest.node_id,
            noise_floor=noise_floor,
            signals=signals,
        )

    def _matching_attempts(
        self,
        experiment: ExperimentInput,
    ) -> Tuple[EvaluationAttemptInput, ...]:
        return tuple(
            attempt
            for attempt in experiment.attempts
            if attempt.evaluator_id == self.settings.evaluator_id
            and attempt.fidelity == self.settings.comparable_fidelity
            and attempt.fraction == self.settings.comparable_fraction
            and attempt.seed == self.settings.comparable_seed
        )

    def _project_experiment(
        self,
        experiment: ExperimentInput,
        objective_direction: ObjectiveDirection,
    ) -> ExperimentEvidence:
        matching = self._matching_attempts(experiment)
        raw_score = (
            sum(attempt.score for attempt in matching) / len(matching)
            if matching
            else experiment.score
        )
        if experiment.had_error:
            evaluation_status = EvaluationStatus.NOT_RUN
            implementation_status = ImplementationStatus.FAILED_TECHNICAL
        elif not experiment.evaluation_valid:
            evaluation_status = EvaluationStatus.INVALID
            implementation_status = ImplementationStatus.COMPLETED
        elif raw_score is None:
            evaluation_status = EvaluationStatus.INCONCLUSIVE
            implementation_status = ImplementationStatus.COMPLETED
        else:
            evaluation_status = EvaluationStatus.VALID
            implementation_status = ImplementationStatus.COMPLETED
        comparable = bool(
            matching
            and evaluation_status == EvaluationStatus.VALID
            and implementation_status == ImplementationStatus.COMPLETED
        )
        sign = 1.0 if objective_direction == ObjectiveDirection.MAXIMIZE else -1.0
        normalized_utility = (
            sign * raw_score
            if evaluation_status == EvaluationStatus.VALID and raw_score is not None
            else None
        )
        observed_attempt = (
            matching[-1]
            if matching
            else (experiment.attempts[-1] if experiment.attempts else None)
        )
        return ExperimentEvidence(
            node_id=experiment.node_id,
            idea_id=experiment.idea_id,
            selection_batch_id=experiment.selection_batch_id,
            parent_node_id=experiment.parent_node_id,
            proposal=experiment.proposal,
            raw_score=raw_score,
            normalized_utility=normalized_utility,
            evaluation_status=evaluation_status,
            implementation_status=implementation_status,
            evaluator_id=(
                None if observed_attempt is None else observed_attempt.evaluator_id
            ),
            build_fidelity=experiment.build_fidelity,
            eval_fidelity=(
                None if observed_attempt is None else observed_attempt.fidelity
            ),
            eval_fraction=(
                None if observed_attempt is None else observed_attempt.fraction
            ),
            seed=None if observed_attempt is None else observed_attempt.seed,
            comparable=comparable,
            feedback=experiment.feedback,
            technical_difficulty=experiment.technical_difficulty,
            created_at=experiment.created_at,
        )

    def _noise_floor(
        self,
        experiments: Tuple[ExperimentInput, ...],
    ) -> Optional[float]:
        repeated_deviations = []
        for experiment in experiments:
            scores = [attempt.score for attempt in self._matching_attempts(experiment)]
            if len(scores) >= self.settings.minimum_repeat_measurements:
                repeated_deviations.append(statistics.stdev(scores))
        if not repeated_deviations:
            return None
        return float(statistics.median(repeated_deviations))

    @staticmethod
    def _relevant_idea_ids(
        archive_state: IdeaArchiveState,
        incumbent: Optional[ExperimentEvidence],
        latest: Optional[ExperimentEvidence],
    ) -> Tuple[str, ...]:
        prioritized = []
        for experiment in (incumbent, latest):
            if experiment is not None and experiment.idea_id not in prioritized:
                prioritized.append(experiment.idea_id)
        for idea in sorted(
            archive_state.ideas,
            key=lambda item: (item.created_at, item.idea_id),
            reverse=True,
        ):
            if idea.idea_id not in prioritized:
                prioritized.append(idea.idea_id)
        return tuple(prioritized)

    def _signals(
        self,
        *,
        experiment_inputs: Tuple[ExperimentInput, ...],
        projected: Tuple[ExperimentEvidence, ...],
        archive_state: IdeaArchiveState,
        capacity: IdeationCapacityView,
        incumbent: Optional[ExperimentEvidence],
        latest: Optional[ExperimentEvidence],
        noise_floor: Optional[float],
    ) -> Tuple[EvidenceSignal, ...]:
        signals = []
        comparable = tuple(item for item in projected if item.comparable)
        if not comparable:
            signals.append(EvidenceSignal.NO_COMPARABLE_EXPERIMENT)
        if incumbent is not None:
            signals.append(EvidenceSignal.DELIVERY_INCUMBENT)
        latest_input = experiment_inputs[-1] if experiment_inputs else None
        if latest_input is not None and latest_input.recoverable_error:
            signals.append(EvidenceSignal.RECOVERABLE_TECHNICAL_FAILURE)
        if (
            capacity.fidelity_profile == "validate"
            or capacity.target_node_id is not None
        ):
            signals.append(EvidenceSignal.FIDELITY_PROMOTION_REQUIRED)
        if any(self._has_proxy_divergence(item) for item in experiment_inputs):
            signals.append(EvidenceSignal.PROXY_FULL_DIVERGENCE)
        threshold = max(
            self.settings.minimum_credible_delta,
            0.0 if noise_floor is None else noise_floor,
        )
        if noise_floor is None:
            signals.append(EvidenceSignal.PROVISIONAL_NOISE)
        latest_comparable = latest if latest is not None and latest.comparable else None
        prior_comparable = tuple(
            item
            for item in comparable
            if latest_comparable is None or item.node_id != latest_comparable.node_id
        )
        if latest_comparable is not None and prior_comparable:
            prior_best = max(item.normalized_utility for item in prior_comparable)
            gain = latest_comparable.normalized_utility - prior_best
            if gain > threshold:
                signals.append(EvidenceSignal.CREDIBLE_IMPROVEMENT)
            if gain > self.settings.surprising_gain_multiplier * threshold:
                signals.append(EvidenceSignal.SURPRISING_GAIN)
        if len(comparable) >= self.settings.plateau_window:
            recent = comparable[-self.settings.plateau_window :]
            utilities = [item.normalized_utility for item in recent]
            if max(utilities) - min(utilities) <= threshold:
                signals.append(EvidenceSignal.PLATEAU)
        causal_idea_ids = {
            item.idea_id for item in (incumbent, latest) if item is not None
        }
        if any(
            claim.status == EvidenceStatus.SUPPORTED
            and bool(causal_idea_ids & set(claim.affected_idea_ids))
            for claim in archive_state.claims
        ):
            signals.append(EvidenceSignal.SUPPORTED_LEVER)
        if any(
            claim.status == EvidenceStatus.CONTRADICTED
            and bool(causal_idea_ids & set(claim.affected_idea_ids))
            for claim in archive_state.claims
        ):
            signals.append(EvidenceSignal.CONTRADICTED_LEVER)
        recent_ideas = sorted(
            archive_state.ideas,
            key=lambda item: (item.created_at, item.idea_id),
        )[-self.settings.diversity_window :]
        if (
            len(recent_ideas) >= self.settings.diversity_window
            and len({idea.descriptor.approach_family for idea in recent_ideas}) == 1
        ):
            signals.append(EvidenceSignal.DIVERSITY_COLLAPSE)
        if any(
            gap.state == GapState.OPEN
            and gap.deferral_count >= self.settings.gap_debt_threshold
            for gap in archive_state.gaps
        ):
            signals.append(EvidenceSignal.GAP_DEBT)
        return tuple(signals)

    def _has_proxy_divergence(self, experiment: ExperimentInput) -> bool:
        proxy_scores = [
            attempt.score
            for attempt in experiment.attempts
            if attempt.evaluator_id == self.settings.evaluator_id
            and attempt.fidelity != self.settings.comparable_fidelity
            and attempt.seed == self.settings.comparable_seed
        ]
        full_scores = [attempt.score for attempt in self._matching_attempts(experiment)]
        if not proxy_scores or not full_scores:
            return False
        return (
            abs(statistics.mean(proxy_scores) - statistics.mean(full_scores))
            > self.settings.proxy_divergence_threshold
        )


def rank_evaluation_gaps(
    gaps: Iterable[EvaluationGap],
    *,
    evidence_confidence_by_gap: Mapping[str, float],
    uncertainty_reduction_by_gap: Mapping[str, float],
    as_of: str,
    settings: GapPrioritySettings,
) -> Tuple[GapPriority, ...]:
    now = datetime.fromisoformat(as_of)
    if now.utcoffset() is None:
        raise ValueError("gap ranking timestamp must include a UTC offset")
    priorities = []
    for gap in gaps:
        if gap.state != GapState.OPEN:
            continue
        assumptions = []
        if gap.gap_id in evidence_confidence_by_gap:
            evidence_confidence = evidence_confidence_by_gap[gap.gap_id]
        else:
            evidence_confidence = settings.default_evidence_confidence
            assumptions.append("default evidence confidence")
        if gap.gap_id in uncertainty_reduction_by_gap:
            uncertainty_reduction = uncertainty_reduction_by_gap[gap.gap_id]
        else:
            uncertainty_reduction = settings.default_uncertainty_reduction
            assumptions.append("default uncertainty reduction")
        for value, name in (
            (evidence_confidence, "evidence confidence"),
            (uncertainty_reduction, "uncertainty reduction"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"gap {name} must be in [0, 1]")
        if gap.estimated_cost is None:
            cost = settings.default_cost
            assumptions.append("default cost")
        else:
            cost = max(gap.estimated_cost, settings.minimum_cost)
            if gap.estimated_cost < settings.minimum_cost:
                assumptions.append("minimum cost floor")
        opened = datetime.fromisoformat(gap.opened_at)
        age_seconds = (now - opened).total_seconds()
        if age_seconds < 0:
            raise ValueError("gap ranking timestamp precedes gap opening")
        score = (
            gap.impact
            * float(evidence_confidence)
            * float(uncertainty_reduction)
            / cost
        )
        priorities.append(
            GapPriority(
                gap_id=gap.gap_id,
                score=score,
                evidence_confidence=float(evidence_confidence),
                uncertainty_reduction=float(uncertainty_reduction),
                cost=float(cost),
                deferral_count=gap.deferral_count,
                age_seconds=age_seconds,
                assumptions=tuple(assumptions),
            )
        )
    return tuple(
        sorted(
            priorities,
            key=lambda item: (
                -item.score,
                -item.deferral_count,
                -item.age_seconds,
                item.gap_id,
            ),
        )
    )
