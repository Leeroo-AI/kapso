# Scorecard — frame-computed aggregation, calibration pooling, verdict checks.
#
# Design: learn-from-trajectories-grader-scoring.md §2. All arithmetic here is
# FRAME math — an agent-written number is never trusted for aggregation
# (§6.6); the scorecard-assessor writes only the decision and rationale, and
# this module validates that block against the recomputed numbers. Paired
# comparisons are valid only within one split_version; `within-noise` is a
# first-class decision, never rounded to a win; gates dominate scores.

from math import sqrt
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

DIMENSIONS = ("foresight", "accuracy", "serving")
DECISIONS = ("accept", "reject", "within-noise")


def _dimension_values(reports: List[Dict[str, Any]], dimension: str) -> List[float]:
    values = []
    for report in reports:
        value = (report.get("hindcast") or {}).get(dimension)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def aggregate(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-dimension mean ± SE with per-trajectory values and null counts —
    at small n the reader must see the distribution, not just its mean (§2.1)."""
    out: Dict[str, Any] = {}
    for dimension in DIMENSIONS:
        values = _dimension_values(reports, dimension)
        n = len(values)
        out[dimension] = {
            "values": values,
            "n": n,
            "nulls": len(reports) - n,
            "mean": round(mean(values), 4) if n else None,
            "se": round(stdev(values) / sqrt(n), 4) if n >= 2 else None,
        }
    return out


def paired_deltas(
    candidate: List[Dict[str, Any]],
    incumbent: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Candidate-vs-incumbent, paired per trajectory (§2.1): both sides must
    have sat the SAME exam — same split_version, same trajectory set — or the
    comparison is invalid and raises."""
    def keyed(reports):
        by_id = {}
        for report in reports:
            trajectory = report.get("trajectory")
            if not trajectory:
                raise ValueError("a report is missing its trajectory id")
            by_id[trajectory] = report
        return by_id

    cand, inc = keyed(candidate), keyed(incumbent)
    if set(cand) != set(inc):
        raise ValueError(
            "paired comparison requires identical trajectory sets; "
            f"only-candidate={sorted(set(cand) - set(inc))} "
            f"only-incumbent={sorted(set(inc) - set(cand))}"
        )
    versions = {r.get("split_version") for r in candidate + incumbent}
    if len(versions) != 1:
        raise ValueError(
            f"paired comparison across split versions {sorted(map(str, versions))} "
            "is invalid — both learner versions must sit the same exam"
        )

    out: Dict[str, Any] = {"split_version": versions.pop(), "n_pairs": len(cand)}
    for dimension in DIMENSIONS:
        deltas = []
        for trajectory, report in cand.items():
            candidate_value = (report.get("hindcast") or {}).get(dimension)
            incumbent_value = (inc[trajectory].get("hindcast") or {}).get(dimension)
            if isinstance(candidate_value, (int, float)) and isinstance(
                incumbent_value, (int, float)
            ):
                deltas.append(float(candidate_value) - float(incumbent_value))
        n = len(deltas)
        out[dimension] = {
            "deltas": [round(d, 4) for d in deltas],
            "n": n,
            "mean": round(mean(deltas), 4) if n else None,
            "se": round(stdev(deltas) / sqrt(n), 4) if n >= 2 else None,
        }
    return out


def calibration_table(
    settlements: List[Dict[str, Any]], graders_config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Pool settled claims by the CLAIMED reliability at serving time and
    report realized agreement per bucket (§2.2). Below `calibration_min`
    pooled settlements the table is None with the count reported by the
    caller — calibration is the slowest number in the suite and is absent
    until it exists."""
    minimum = graders_config["calibration_min"]
    cuts = list(graders_config["calibration_buckets"])
    if len(settlements) < minimum:
        return None
    edges = [0.0] + cuts + [1.0]
    buckets = []
    for low, high in zip(edges, edges[1:]):
        rows = [
            s for s in settlements
            if low <= float(s["claimed"]) < high
            or (high == 1.0 and float(s["claimed"]) == 1.0)
        ]
        agreed = sum(1 for s in rows if s["agreed"])
        buckets.append({
            "claimed": f"[{low:.1f}–{high:.1f})",
            "n": len(rows),
            "agreed": agreed,
            "rate": round(agreed / len(rows), 3) if rows else None,
        })
    return {"pooled": len(settlements), "buckets": buckets}


def validate_verdict(
    verdict: Dict[str, Any],
    computed_deltas: Dict[str, Any],
    gauntlet_rollup: str,
) -> List[str]:
    """The verdict block (§2.4) against frame-recomputed numbers: decision in
    vocabulary, rationale present (no naked tags), gates dominate scores, and
    any agent-quoted delta must equal the frame's arithmetic."""
    findings = []
    decision = verdict.get("decision")
    if decision not in DECISIONS:
        findings.append(f"verdict decision {decision!r} is not one of {DECISIONS}")
    if not str(verdict.get("rationale", "")).strip():
        findings.append("verdict rationale is missing — no naked tags")
    if gauntlet_rollup == "FAIL" and decision == "accept":
        findings.append(
            "gauntlet FAIL with decision `accept` — gates dominate scores"
        )
    for dimension in DIMENSIONS:
        quoted = verdict.get(f"{dimension}_delta")
        if quoted is None:
            continue
        frame_value = computed_deltas.get(dimension, {}).get("mean")
        if frame_value is None or abs(float(quoted) - frame_value) > 1e-9:
            findings.append(
                f"verdict quotes {dimension}_delta={quoted} but the frame "
                f"computed {frame_value} — agent numbers are never trusted "
                f"for arithmetic"
            )
    return findings
