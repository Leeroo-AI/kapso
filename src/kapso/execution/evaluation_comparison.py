"""Deterministic baseline comparisons; these never rank search candidates."""

from typing import Any, Dict, Mapping


def compare_metrics(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    directions: Mapping[str, str],
) -> Dict[str, Any]:
    """Compare shared metrics, with explicit maximize/minimize semantics.

    Inputs must already be normalized. A missing baseline or direction is not
    evidence of improvement. Deltas are always candidate minus baseline.
    """
    comparisons = {}
    for name, value in candidate.items():
        if name not in baseline:
            continue
        delta = value - baseline[name]
        direction = directions.get(name)
        verdict = "unclassified"
        if direction is not None:
            improvement = delta if direction == "maximize" else -delta
            verdict = (
                "improved"
                if improvement > 0
                else "regressed" if improvement < 0 else "unchanged"
            )
        comparisons[name] = {
            "baseline": baseline[name],
            "candidate": value,
            "delta": delta,
            "direction": direction,
            "verdict": verdict,
        }
    return {
        "metrics": comparisons,
        "missing_baseline": sorted(set(candidate) - set(baseline)),
    }


def comparison_summary(metadata: Mapping[str, Any]) -> str:
    """Render stored comparisons without assigning a promotion decision."""
    report = metadata.get("baseline_comparison") or {}
    if not isinstance(report, Mapping):
        return ""
    metrics = report.get("metrics") or {}
    if not isinstance(metrics, Mapping):
        return ""
    parts = []
    for name, entry in metrics.items():
        try:
            parts.append(
                f"{name}: {entry['candidate']:.6g} "
                f"vs {entry['baseline']:.6g} "
                f"(delta {entry['delta']:+.6g}, {entry['verdict']})"
            )
        except (KeyError, TypeError, ValueError):
            continue  # Older callbacks may have used this metadata key.
    missing = report.get("missing_baseline", [])
    if (
        isinstance(missing, list)
        and missing
        and all(isinstance(name, str) for name in missing)
    ):
        parts.append("no baseline: " + ", ".join(missing))
    return "; ".join(parts)
