"""Mechanical binding between runtime attempts and full evaluator contracts."""

from __future__ import annotations

import re
from typing import Any

from kapso.cross_run.contracts import EvaluationFingerprint
from kapso.execution.memories.experiment_memory.store import ExperimentRecord


def evaluation_scores_match(
    score: float,
    aggregate: float,
    tolerance: float,
) -> bool:
    """Use the one configured predicate at capture and projection boundaries."""

    return abs(score - aggregate) <= tolerance


def validate_evaluation_fingerprints(
    records: tuple[ExperimentRecord, ...],
    fingerprints: tuple[EvaluationFingerprint, ...],
    score_comparison_tolerance: float,
    error_type: type[Exception],
) -> tuple[EvaluationFingerprint, ...]:
    registry = {
        fingerprint.evaluation_fingerprint_id: fingerprint
        for fingerprint in fingerprints
    }
    referenced: set[str] = set()
    for record in records:
        grouped: dict[tuple[str, str, float], list[Any]] = {}
        for attempt in record.evaluation_attempts:
            if re.fullmatch(r"[0-9a-f]{64}", attempt.evaluator_id) is None:
                raise error_type("runtime evaluator id is not a SHA-256")
            grouped.setdefault(
                (attempt.evaluator_id, attempt.fidelity, attempt.fraction), []
            ).append(attempt)
        group_means: list[float] = []
        for (evaluator_id, fidelity, fraction), attempts in grouped.items():
            seed_ids = tuple(sorted(f"seed-{attempt.seed}" for attempt in attempts))
            if len(seed_ids) != len(set(seed_ids)):
                raise error_type(
                    "evaluation attempts contain a duplicate seed in one evaluator class"
                )
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
                raise error_type("evaluation attempts lack one exact full fingerprint")
            fingerprint = matches[0]
            if fingerprint.aggregation_protocol != "arithmetic-mean":
                raise error_type("evaluation aggregation protocol is unsupported")
            referenced.add(fingerprint.evaluation_fingerprint_id)
            group_means.append(
                sum(attempt.score for attempt in attempts) / len(attempts)
            )
        if record.evaluation_valid and record.raw_score is not None:
            if (
                not grouped
                or sum(
                    evaluation_scores_match(
                        record.raw_score,
                        mean,
                        score_comparison_tolerance,
                    )
                    for mean in group_means
                )
                != 1
            ):
                raise error_type("raw score is not one unambiguous evaluator aggregate")
        if record.evaluation_valid and record.raw_score is None and grouped:
            raise error_type("measured valid experiment has no raw score of record")
    return tuple(registry[identifier] for identifier in sorted(referenced))
