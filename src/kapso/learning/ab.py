# A/B arm verdict math — rung 4's causal instrument (grader-scoring §2.5).
#
# The frame's half: paired per-task deltas ± SE, the guard KPI (no
# regression where the candidate bank is thin — "never worse where
# irrelevant", measured), and the promotion gate semantics: with
# `required: false` a not-run A/B is recorded as not-run, never as passed;
# when required, a within-noise result BLOCKS (noise is never rounded to a
# win). Arm execution is ordinary evolve runs differing only in the served
# bank ref — launching them is an operator go/no-go at the time; this module
# judges their results.

import math
from typing import Any, Dict, List

AB_VERDICTS = ("not-run", "win", "within-noise", "regression")


def ab_verdict(
    pairs: List[Dict[str, Any]], ab_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Judge a completed set of same-task arm pairs.

    Each pair: {task, candidate, incumbent, maximize, thin} — the two arms'
    primary-metric results, the metric direction, and whether the candidate
    bank was thin for the task (from the brief's gap analysis). An empty
    list is the not-run record. Deltas are oriented so positive always means
    the candidate arm did better."""
    required = ab_config["required"]
    expected_pairs = ab_config["pairs"]
    if not pairs:
        return {
            "verdict": "not-run",
            "blocking": bool(required),
            "n": 0,
            "expected_pairs": expected_pairs,
            "deltas": [],
            "mean": None,
            "se": None,
            "guard": {"thin_pairs": 0, "regressions": []},
        }

    deltas = []
    for pair in pairs:
        delta = float(pair["candidate"]) - float(pair["incumbent"])
        if not pair["maximize"]:
            delta = -delta
        deltas.append({"task": pair["task"], "delta": delta,
                       "thin": bool(pair["thin"])})

    values = [row["delta"] for row in deltas]
    mean = sum(values) / len(values)
    se = (
        math.sqrt(
            sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            / len(values)
        )
        if len(values) > 1 else None
    )

    # Guard KPI: on thin tasks the candidate must not be worse beyond noise
    # (a thin brief should be inert, never harmful). With no SE (one pair),
    # any thin regression trips the guard — conservative.
    noise = se if se is not None else 0.0
    guard_regressions = [
        row["task"] for row in deltas
        if row["thin"] and row["delta"] < -noise
    ]

    if guard_regressions or (se is not None and mean < -se):
        verdict = "regression"
    elif se is not None and mean > se:
        verdict = "win"
    elif se is None and mean > 0 and not guard_regressions:
        # A single pair can only suggest; it never certifies a win.
        verdict = "within-noise"
    else:
        verdict = "within-noise"

    return {
        "verdict": verdict,
        # Promotion gate: required + anything short of a win blocks; the
        # waived path never blocks but records exactly what happened.
        "blocking": bool(required) and verdict != "win",
        "n": len(deltas),
        "expected_pairs": expected_pairs,
        "deltas": deltas,
        "mean": round(mean, 6),
        "se": round(se, 6) if se is not None else None,
        "guard": {
            "thin_pairs": sum(1 for row in deltas if row["thin"]),
            "regressions": guard_regressions,
        },
    }
