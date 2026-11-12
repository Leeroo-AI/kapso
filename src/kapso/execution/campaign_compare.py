"""`kapso compare`: two campaigns set side by side.

A pure reader over what each campaign already wrote (its run checkpoint
and its status file): it never writes into a campaign and makes no model
calls. The delivered node and the ranking are the ones `kapso report`
uses, so a campaign reads the same in both commands. A comparison names a
winner only when the two scores mean the same thing: the same scoring
direction and the same registered evaluator.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from kapso.execution.campaign_report import (
    campaign_standing,
    campaign_state,
    delivered_score,
    load_campaign_state,
    markdown_table,
)
from kapso.execution.search_strategies.base import SearchNode

SIDES = ("a", "b")


class CampaignCompareError(ValueError):
    """A comparison that cannot be made, with the line that would make it."""


def campaign_facts(campaign: str | Path) -> Dict[str, Any]:
    """What one campaign says about itself, as data. A directory without a
    checkpoint or a status file raises, and so does any corrupt state file."""
    campaign = Path(campaign).expanduser()
    checkpoint, status, _, nodes = load_campaign_state(campaign)
    ranked, evaluator_id, deliverable = campaign_standing(checkpoint, nodes)
    return {
        "campaign": str(campaign),
        "goal": checkpoint.goal.strip(),
        "state": campaign_state(status),
        "maximize_scoring": checkpoint.maximize_scoring,
        "evaluator_id": evaluator_id,
        "iterations": checkpoint.completed_iterations,
        "elapsed_minutes": checkpoint.elapsed_seconds / 60,
        "model_spend_usd": checkpoint.cumulative_cost,
        "candidates": len(nodes),
        "scored": len(ranked),
        "failed": sum(1 for node in nodes if node.had_error),
        "delivered": None if deliverable is None else {
            "node": deliverable.node_id,
            "branch": deliverable.branch_name,
            "score": delivered_score(deliverable, evaluator_id),
        },
        "progress": _progress(
            nodes, {node.node_id for node in ranked}, checkpoint.maximize_scoring
        ),
    }


def _progress(
    nodes: Sequence[SearchNode], scored_ids: set, maximize: bool
) -> List[Optional[float]]:
    """The best score so far after each candidate, in the order the
    campaign ran them; None until the first scored candidate."""
    best: Optional[float] = None
    progress: List[Optional[float]] = []
    for node in nodes:
        if node.node_id in scored_ids and (
            best is None or (node.score > best if maximize else node.score < best)
        ):
            best = node.score
        progress.append(best)
    return progress


def _evaluator(facts: Dict[str, Any]) -> str:
    return facts["evaluator_id"][:12] or "none"


def _comparability(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, List[str]]:
    """What stands between the two scores and a fair comparison: blockers
    make a verdict meaningless, warnings leave it to the reader."""
    blockers: List[str] = []
    warnings: List[str] = []
    if a["maximize_scoring"] != b["maximize_scoring"]:
        blockers.append(
            "the two campaigns score in opposite directions "
            f"(A: {_direction(a)}, B: {_direction(b)})"
        )
    if a["evaluator_id"] != b["evaluator_id"]:
        blockers.append(
            "the two campaigns were scored by different evaluators "
            f"(A: {_evaluator(a)}, B: {_evaluator(b)})"
        )
    elif not a["evaluator_id"]:
        warnings.append(
            "neither campaign registered an evaluator, so nothing shows "
            "the two scores were measured the same way"
        )
    if a["goal"] != b["goal"]:
        warnings.append("the two campaigns have different goals")
    return {"blockers": blockers, "warnings": warnings}


def _direction(facts: Dict[str, Any]) -> str:
    return "higher is better" if facts["maximize_scoring"] else "lower is better"


def _verdict(
    a: Dict[str, Any], b: Dict[str, Any], blockers: Sequence[str]
) -> Dict[str, Any]:
    if blockers:
        return {"winner": None, "reason": "not comparable: " + "; ".join(blockers)}
    missing = [side.upper() for side, facts in zip(SIDES, (a, b)) if facts["delivered"] is None]
    if missing:
        return {
            "winner": None,
            "reason": f"no verdict: {' and '.join(missing)} has no scored candidate",
        }
    gap = a["delivered"]["score"] - b["delivered"]["score"]
    if gap == 0:
        return {"winner": "tie", "reason": "the two campaigns deliver the same score"}
    winner = "a" if (gap > 0) == a["maximize_scoring"] else "b"
    return {
        "winner": winner,
        "reason": f"{winner.upper()} delivers the better score by {abs(gap):.6g} ({_direction(a)})",
    }


def compare_campaigns(campaign_a: str | Path, campaign_b: str | Path) -> Dict[str, Any]:
    """The comparison as data: each campaign's facts, what stands between
    their scores and a fair comparison, and the verdict. Comparing a
    campaign with itself raises CampaignCompareError."""
    if Path(campaign_a).expanduser().resolve() == Path(campaign_b).expanduser().resolve():
        raise CampaignCompareError(f"compare needs two different campaigns, got {campaign_a} twice")
    a = campaign_facts(campaign_a)
    b = campaign_facts(campaign_b)
    comparability = _comparability(a, b)
    return {
        "a": a,
        "b": b,
        "comparability": comparability,
        "verdict": _verdict(a, b, comparability["blockers"]),
    }


def render_comparison(comparison: Dict[str, Any]) -> str:
    """The comparison as Markdown."""
    a, b = comparison["a"], comparison["b"]
    sections = [
        "\n".join([
            "# Campaign comparison",
            "",
            f"- **A:** `{a['campaign']}`",
            f"- **B:** `{b['campaign']}`",
        ]),
        _comparability_section(comparison["comparability"]),
        _summary_section(a, b),
        _verdict_section(comparison["verdict"], a, b),
        _progress_section(a["progress"], b["progress"]),
    ]
    return "\n\n".join(sections) + "\n"


def _comparability_section(comparability: Dict[str, List[str]]) -> str:
    lines = [f"- **Blocks a verdict:** {text}" for text in comparability["blockers"]]
    lines += [f"- **Worth knowing:** {text}" for text in comparability["warnings"]]
    return "\n".join([
        "## Comparability",
        "",
        *(lines or ["Nothing found that makes the two scores incomparable."]),
    ])


def _summary_section(a: Dict[str, Any], b: Dict[str, Any]) -> str:
    rows = [
        ("State", lambda facts: facts["state"]),
        ("Iterations", lambda facts: str(facts["iterations"])),
        ("Elapsed (min)", lambda facts: f"{facts['elapsed_minutes']:.1f}"),
        ("Model spend", lambda facts: f"${facts['model_spend_usd']:.2f}"),
        ("Candidates", lambda facts: str(facts["candidates"])),
        ("Scored", lambda facts: str(facts["scored"])),
        ("Failed", lambda facts: str(facts["failed"])),
        ("Direction", _direction),
        ("Delivered score", lambda facts: _delivered(facts, "score")),
        ("Delivered branch", lambda facts: _delivered(facts, "branch")),
    ]
    return "## Summary\n\n" + markdown_table(
        ["", "A", "B"], [[label, read(a), read(b)] for label, read in rows]
    )


def _delivered(facts: Dict[str, Any], field: str) -> str:
    delivered = facts["delivered"]
    if delivered is None:
        return "none"
    return f"{delivered[field]:.6g}" if field == "score" else f"`{delivered[field]}`"


def _verdict_section(verdict: Dict[str, Any], a: Dict[str, Any], b: Dict[str, Any]) -> str:
    lines = ["## Verdict", "", verdict["reason"][0].upper() + verdict["reason"][1:] + "."]
    if verdict["winner"] in SIDES:
        lines.append("")
        lines.extend(
            f"- **{side.upper()}** spent ${facts['model_spend_usd']:.2f} and "
            f"{facts['elapsed_minutes']:.1f} min over {facts['iterations']} iterations."
            for side, facts in zip(SIDES, (a, b))
        )
    return "\n".join(lines)


def _progress_section(progress_a: Sequence[Optional[float]], progress_b: Sequence[Optional[float]]) -> str:
    """One row for each candidate that improved either campaign's best, a
    campaign past its last candidate holding its final best."""
    rows: List[List[str]] = []
    last: tuple = (None, None)
    for index in range(max(len(progress_a), len(progress_b))):
        current = (
            progress_a[min(index, len(progress_a) - 1)] if progress_a else None,
            progress_b[min(index, len(progress_b) - 1)] if progress_b else None,
        )
        if current != last and current != (None, None):
            rows.append([str(index + 1), *("-" if value is None else f"{value:.6g}" for value in current)])
        last = current
    if not rows:
        return "## Progress\n\nNo scored candidate in either campaign."
    return "\n".join([
        "## Progress",
        "",
        "The best score so far after each candidate that improved either campaign's best.",
        "",
        markdown_table(["Candidate", "A best", "B best"], rows),
    ])
