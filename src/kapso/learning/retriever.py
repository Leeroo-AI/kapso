# Retriever — the serving component: push brief + pull tools over one core.
#
# Design: learn-from-trajectories-design.md §5.1. Push is a PURE FUNCTION of
# (task, bank checkout, bank_head): eligibility is law (task ∈ scope,
# quarantine excluded), rank is vectorless (reliability order with
# ledger-derived coverage discounts; tags never eligibility), the brief closes
# with the gap analysis, and every serving event lands in the serving record —
# the attribution ground truth. No agent, no clock, no randomness ever sits
# inside push: the hindcast replays it at historical heads, byte-identical.
#
# Pull (P5) is the same eligibility law exposed as session tools:
# `pull_shortlist` returns the WHOLE eligible set as hero lines (the calling
# agent is the reranker — the query is logged, never a filter; the census
# line replaces padding), `pull_projections` renders full cards on request
# and refuses quarantine by name (eligibility is law even on direct request).
# Selection discounts have nothing to select in pull, so hero lines carry
# `visited` instead. Every pull event appends to a JSONL log — the serving
# record's second exposure level: `searched` (hero shown) vs `got` (full
# card rendered); attribution binds to `got`.

import json
from datetime import datetime, timezone
from pathlib import Path
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


def _tension_pairs(cards: List[Card]) -> List[Tuple[str, str]]:
    """Co-serving guard input: contradicts pairs inside one returned set —
    always named, never silently side by side (§5.1). Runs in both modes."""
    names = {card.name for card in cards}
    pairs: List[Tuple[str, str]] = []
    for card in cards:
        for other in card.frontmatter.get("contradicts") or []:
            pair = tuple(sorted((card.name, str(other))))
            if str(other) in names and pair not in pairs:
                pairs.append(pair)
    return pairs


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
    tensions = _tension_pairs([card for card, _, _ in selected])

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


# ------------------------------------------------------------------ pull

def _card_row(card: Card, exposure: str) -> Dict[str, Any]:
    return {
        "card": card.name,
        "version": (card.frontmatter.get("provenance") or {}).get("version"),
        "state": card.state,
        "score": card.score,
        "exposure": exposure,
    }


def pull_shortlist(
    bank: Bank, task_coords: Dict[str, str], query: str
) -> Dict[str, Any]:
    """bank_search: the whole eligible set as reliability-ordered hero lines.

    The query is logged, never a filter — the calling agent is the reranker,
    reading hero lines exactly as crews read index.md. The closing census
    line is the thin-set honesty: the bank's whole answer is on screen, so
    nothing can be padded in unmarked."""
    eligible = [c for c in bank.servable() if c.eligible_for(task_coords)]
    eligible.sort(key=lambda card: _rank_key(card, float(card.score or 0.0)))
    lines = []
    for card in eligible:
        kind = card.type + (
            ", pitfall" if PITFALL_TAG in (card.frontmatter.get("tags") or [])
            else ""
        )
        visited = "yes" if _visited(card, task_coords) else "no"
        lines.append(
            f"- [card:{card.name}] ({kind}) state={card.state} "
            f"score={card.score} visited-this-dataset={visited} — {card.hero}"
        )
    census = (
        f"Eligible set: {len(eligible)} card(s) for this task's scope — "
        f"this is the bank's whole answer; nothing else matches. Use "
        f"bank_get with card names for full cards."
    )
    text = (
        "\n".join(lines) + "\n\n" + census if lines
        else "The bank holds NO eligible card for this task's scope. " + census
    )
    return {
        "text": text,
        "shown": [_card_row(card, "searched") for card in eligible],
        "eligible": len(eligible),
    }


def pull_projections(
    bank: Bank, task_coords: Dict[str, str], card_names: List[str]
) -> Dict[str, Any]:
    """bank_get: full served projections for the requested cards. Quarantine
    and eligibility are law even on direct request — refusals are named, so
    the agent learns the boundary instead of silence."""
    decoys = bank.decoy_names
    got: List[Card] = []
    refused: List[Dict[str, str]] = []
    for name in card_names:
        card = bank.cards.get(name)
        if card is None or name in decoys:
            # Decoys refuse as unknown — naming the quarantine would mark
            # the bait (§2.3: quarantine is frame-side knowledge).
            refused.append({"card": name, "reason": "no such card"})
        elif card.state not in ("candidate", "active"):
            refused.append(
                {"card": name, "reason": f"not servable (state={card.state})"}
            )
        elif not card.eligible_for(task_coords):
            refused.append(
                {"card": name,
                 "reason": f"out of scope for this task ({card.scope})"}
            )
        else:
            got.append(card)
    tensions = _tension_pairs(got)
    sections = [_render_card(card) for card in got]
    for first, second in tensions:
        sections.append(
            f"**Contested:** [card:{first}] and [card:{second}] disagree on "
            f"overlapping scope; the boundary is unresolved — treat as "
            f"contested."
        )
    for refusal in refused:
        sections.append(f"*not served:* {refusal['card']} — {refusal['reason']}")
    return {
        "text": "\n\n".join(sections) if sections else "nothing served",
        "got": [_card_row(card, "got") for card in got],
        "refused": refused,
        "tensions": [list(pair) for pair in tensions],
    }


def append_pull_event(log_path: str, event: Dict[str, Any]) -> None:
    """One serving event to the pull log — append-only JSONL beside the push
    record; the harvester collects both. Timestamps are operational (pull is
    live agent interaction, never replayed like push)."""
    path = Path(log_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    stamped = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **event,
    }
    with open(path, "a") as handle:
        handle.write(json.dumps(stamped) + "\n")
