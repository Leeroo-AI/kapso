# Retriever — the serving component's push core (P3 slice).
#
# Design: learn-from-trajectories-design.md §5.1. Push is a PURE FUNCTION of
# (task, bank checkout, bank_head): eligibility is law (task ∈ scope,
# quarantine excluded), rank is vectorless (reliability order with
# ledger-derived coverage discounts; tags never eligibility), the brief closes
# with the gap analysis, and every serving event lands in the serving record —
# the attribution ground truth. No agent, no clock, no randomness ever sits
# inside push: the hindcast replays it at historical heads, byte-identical.
# The pull tools and the probe rider wire in at P5/P6 on this same core.

from typing import Any, Dict, List, Optional, Tuple

from kapso.learning.bank import Bank, Card

PITFALL_TAG = "pitfall"


def _dataset_of_trajectory(trajectory_id: str) -> str:
    """rel-amazon--user-churn/<stamp> -> rel-amazon (the id's dataset half)."""
    task_dir = trajectory_id.split("/", 1)[0]
    return task_dir.split("--", 1)[0]


def _visited(card: Card, task_coords: Dict[str, str]) -> bool:
    """Ledger-derived: has any evidence entry's trajectory visited this task's
    dataset? Coverage is never authored — it is read off the ledger."""
    dataset = task_coords.get("dataset")
    if not dataset:
        return False
    for entry in card.evidence:
        source = entry.get("source") or {}
        trajectory = source.get("trajectory")
        if trajectory and _dataset_of_trajectory(str(trajectory)) == dataset:
            return True
    return False


def _rank_key(card: Card, effective_score: float) -> Tuple[float, str]:
    return (-effective_score, card.name)


def _render_card(card: Card) -> str:
    """The served-card projection (§3.2): title + hero + reliability line +
    scope + THE FACT + evidence digest + probe. Assembled at serve time,
    never stored; the fact is carried whole (Rule 6)."""
    reliability = card.reliability
    scores = " ".join(
        f"{dim}={reliability.get(dim)}"
        for dim in ("validity", "boundary", "coverage", "score")
    )
    verdicts: Dict[str, int] = {}
    for entry in card.evidence:
        verdict = str(entry.get("verdict"))
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
    digest = ", ".join(f"{v} ×{n}" for v, n in sorted(verdicts.items()))
    lines = [
        f"### {card.frontmatter.get('title', card.name)}  [card:{card.name}]",
        f"*{card.hero}*",
        f"- reliability: state={card.state} {scores}",
        f"- scope: {card.scope}"
        + (f" — {card.frontmatter['scope_conditions']}"
           if card.frontmatter.get("scope_conditions") else ""),
        f"- evidence: {len(card.evidence)} entries ({digest})",
        "",
        card.body,
    ]
    probe = card.frontmatter.get("probe")
    if probe:
        lines.append(f"\n*probe:* {str(probe).strip()}")
    return "\n".join(lines)


def compile_brief(
    bank: Bank,
    task_coords: Dict[str, str],
    bank_head: str,
    retriever_config: Dict[str, Any],
) -> Dict[str, Any]:
    """The push brief + its serving record.

    Returns {brief, record}: `brief` is the rendered markdown that replaces
    the static context notes; `record` is the serving record — per served
    card: name, version, state, score, visited, effective rank score, and
    exposure level `got` (push renders in full). Deterministic throughout.
    """
    k_insights = retriever_config["k_insights"]
    k_procedures = retriever_config["k_procedures"]
    k_pitfalls = retriever_config["k_pitfalls"]
    discount = retriever_config["unvisited_discount"]

    eligible = [c for c in bank.servable() if c.eligible_for(task_coords)]
    ranked: List[Tuple[Card, bool, float]] = []
    for card in eligible:
        visited = _visited(card, task_coords)
        base = float(card.score or 0.0)
        effective = base if visited else base * discount
        ranked.append((card, visited, effective))
    ranked.sort(key=lambda item: _rank_key(item[0], item[2]))

    def is_pitfall(card: Card) -> bool:
        return PITFALL_TAG in (card.frontmatter.get("tags") or [])

    insights = [r for r in ranked if r[0].type == "insight" and not is_pitfall(r[0])]
    procedures = [r for r in ranked if r[0].type == "procedure"]
    pitfalls = [r for r in ranked if r[0].type == "insight" and is_pitfall(r[0])]

    selected = (
        insights[:k_insights] + procedures[:k_procedures] + pitfalls[:k_pitfalls]
    )
    selected_names = {card.name for card, _, _ in selected}

    # Co-serving guard: a contradicts pair served together always names the
    # tension — never silently side by side (§5.1).
    tensions = []
    for card, _, _ in selected:
        for other in card.frontmatter.get("contradicts") or []:
            pair = tuple(sorted((card.name, str(other))))
            if str(other) in selected_names and pair not in tensions:
                tensions.append(pair)

    sections = []
    if insights[:k_insights]:
        sections.append("## Practice notes (from the knowledge bank)\n\n"
                        + "\n\n".join(_render_card(c) for c, _, _ in insights[:k_insights]))
    if procedures[:k_procedures]:
        sections.append("## Procedures\n\n"
                        + "\n\n".join(_render_card(c) for c, _, _ in procedures[:k_procedures]))
    if pitfalls[:k_pitfalls]:
        sections.append("## Pitfall guardrails\n\n"
                        + "\n\n".join(_render_card(c) for c, _, _ in pitfalls[:k_pitfalls]))
    for first, second in tensions:
        sections.append(
            f"**Contested:** [card:{first}] and [card:{second}] disagree on "
            f"overlapping scope; the boundary is unresolved — treat as contested."
        )

    # Gap analysis — the honest "what the bank does not know" (§5.1).
    gaps = []
    if not insights:
        gaps.append("no insight in the bank covers this task's scope")
    if not procedures:
        gaps.append("no procedure in the bank covers this task's scope")
    unvisited_served = [c.name for c, visited, _ in selected if not visited]
    if unvisited_served:
        gaps.append(
            "served at reduced confidence (scope claimed, this dataset never "
            "visited by their evidence): " + ", ".join(sorted(unvisited_served))
        )
    sections.append(
        "## What the bank does not know\n\n"
        + ("\n".join(f"- {g}" for g in gaps) if gaps
           else "- no gaps flagged for this task's scope")
    )

    record = {
        "mode": "push",
        "task": dict(task_coords),
        "bank_head": bank_head,
        "served": [
            {
                "card": card.name,
                "version": (card.frontmatter.get("provenance") or {}).get("version"),
                "state": card.state,
                "score": card.score,
                "visited": visited,
                "effective": round(effective, 6),
                "exposure": "got",
            }
            for card, visited, effective in selected
        ],
        "tensions": [list(pair) for pair in tensions],
        "gaps": gaps,
    }
    return {"brief": "\n\n".join(sections), "record": record}
