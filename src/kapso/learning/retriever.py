# Retriever — serving v2: agentic retrieval over the bank.
#
# Design: serving-agentic-redesign.md (supersedes §5.1's push/pull split).
# Push is an INTRODUCTION only — what the bank is and the three tools; no
# card content is selected or injected by the frame. Selection belongs to
# the reading agent: `render_index` returns the WHOLE BANK as book-index
# lines (name + hero + score + applies-when), and the agent opens cards
# with `render_cards` at two depths (body / body+evidence). Nothing is
# scope-filtered on either surface — scope is displayed information and
# relevance judgment belongs to the reader; only quarantine is law
# (decoys and retired states, frame-side, silent). Rank in the index is
# plain reliability order — no discounts, no k caps, no ranking cut. Everything here is a pure function of (task, bank
# checkout, bank_head): no agent, no clock, no randomness — the hindcast
# replays it at historical heads, byte-identical.
#
# Every tool call appends to a JSONL pull log — the serving record's
# exposure ladder source: `indexed` (index line shown) → `read` (body
# rendered) → `evidence-read` (body + reliability + evidence rendered).
# Attribution binds at `read` and above.

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kapso.learning.bank import Bank, Card
from kapso.learning.bank_invariants import usage_claims_serving

# Probe-queue row grammar: the derived index is one file, readable by
# crews and parsed by the serve-time offer — structural constant, not a knob.
PROBE_QUEUE_RELPATH = "index/probe-queue.md"
_PROBE_ROW_PATTERN = re.compile(r"^\d+\. \[card:([a-z0-9][a-z0-9-]*)\] \(([^)]*)\) — (.+)$")

# The intro is the entire push surface (serving-agentic-redesign.md §4):
# what the bank is, the three tools in reading order, and the arbitrate
# line. No census, no card content, no citation instruction — the citation
# convention lives in the module prompts.
INTRO_TEXT = """## Knowledge bank (measured practice from past campaigns)

A bank of evidence-priced cards distilled from earlier campaigns on this
benchmark: insights (mechanisms that paid or failed, with the conditions
they hold under) and procedures (runnable harnesses — their code ships with
the card). Every card carries a reliability score and the evidence behind
it. It complements the practice notes above; where they disagree, let your
own measurements arbitrate.

Three tools, in reading order:
- bank_index() — the whole bank as one index page: card name, one-liner,
  score, and when it applies. Cheap; call it whenever your next decision
  might have been faced before.
- bank_get_card(cards) — full card bodies (procedures include their code
  path).
- bank_get_card_with_evidence(cards) — the card plus its reliability and
  evidence trail, for due diligence before you stake real budget on its
  advice."""

# Probe offers ride card reads (§3.2), never the index or the intro. The
# cost clause is the wave-A lesson: a probe protocol silently adopted as
# the default gate taxed every experiment on a large dataset.
_PROBE_COST_CLAUSE = (
    "This is an optional measurement offer, not your default gate: adopt "
    "its protocol only if it is affordable at this dataset's scale, and "
    "say so explicitly. An ignored probe stays queued."
)


def _tension_pairs(cards: List[Card]) -> List[Tuple[str, str]]:
    """Co-serving guard input: contradicts pairs inside one returned set —
    always named, never silently side by side."""
    names = {card.name for card in cards}
    pairs: List[Tuple[str, str]] = []
    for card in cards:
        for other in card.frontmatter.get("contradicts") or []:
            pair = tuple(sorted((card.name, str(other))))
            if str(other) in names and pair not in pairs:
                pairs.append(pair)
    return pairs


def _gaps(eligible: List[Card]) -> List[str]:
    """The honest "what the bank does not know" — stated, never padded."""
    gaps = []
    if not any(card.type == "insight" for card in eligible):
        gaps.append("no insight in the bank covers this task's scope")
    if not any(card.type == "procedure" for card in eligible):
        gaps.append("no procedure in the bank covers this task's scope")
    return gaps


def compile_intro(
    bank: Bank, task_coords: Dict[str, str], bank_head: str
) -> Dict[str, Any]:
    """The push side of serving v2: the fixed intro + the launch record.

    Returns {intro, record}. `intro` is the markdown appended after the
    static context notes; `record` is the launch serving record —
    mode/task/bank_head/gaps. Exposure is not recorded here: it is derived
    from the pull log, which starts empty. Deterministic throughout.
    """
    eligible = [c for c in bank.servable() if c.eligible_for(task_coords)]
    gaps = _gaps(eligible)
    stamp = (
        f"Serving from the knowledge bank at head `{bank_head}` "
        f"(pinned for this whole campaign)."
    )
    return {
        "intro": stamp + "\n\n" + INTRO_TEXT,
        "record": {
            "mode": "agentic",
            "task": dict(task_coords),
            "bank_head": bank_head,
            "gaps": gaps,
        },
    }


# ------------------------------------------------------------------ index

def _index_row(card: Card) -> Dict[str, Any]:
    return {
        "card": card.name,
        "version": (card.frontmatter.get("provenance") or {}).get("version"),
        "state": card.state,
        "score": card.score,
        "exposure": "indexed",
    }


def render_index(
    bank: Bank, task_coords: Dict[str, str], section: Optional[str] = None
) -> Dict[str, Any]:
    """bank_index: the WHOLE BANK as book-index lines — no filtering.

    One line-group per card — `[card:name] score X` / hero one-liner /
    `applies-when:` (the card's authored scope_conditions, when present) —
    reliability-ordered within `## Insights` / `## Procedures` sections,
    closing with the gaps footer. Nothing is pre-selected or scoped away
    for the reader: scope is displayed information (the applies-when
    line), and relevance judgment belongs to the calling agent, scanning
    with its own plan in mind. Only quarantine stays frame-side
    (servable() excludes decoys and retired states, silently). The gaps
    footer still reports scope coverage for THIS task — informational,
    never a filter."""
    listed_cards = list(bank.servable())
    gaps = _gaps([c for c in listed_cards if c.eligible_for(task_coords)])
    if section:
        if section not in ("insights", "procedures"):
            raise ValueError(f"unknown index section: {section}")
        wanted = "insight" if section == "insights" else "procedure"
        listed_cards = [c for c in listed_cards if c.type == wanted]
    listed_cards.sort(key=lambda card: (-float(card.score or 0.0), card.name))

    sections: List[str] = []
    for kind, heading in (("insight", "## Insights"),
                          ("procedure", "## Procedures")):
        cards = [c for c in listed_cards if c.type == kind]
        if not cards:
            continue
        lines = [heading]
        for card in cards:
            lines.append(f"[card:{card.name}] score {card.score}")
            lines.append(f"  — {card.hero}")
            conditions = card.frontmatter.get("scope_conditions")
            if conditions:
                lines.append(
                    "  applies-when: " + " ".join(str(conditions).split())
                )
        sections.append("\n".join(lines))

    if not listed_cards:
        sections.append(
            "The bank holds NO card yet — this is its whole answer; "
            "nothing else exists."
        )
    for gap in gaps:
        sections.append(f"gaps: {gap}")
    return {
        "text": "\n\n".join(sections),
        "listed": [_index_row(card) for card in listed_cards],
        "gaps": gaps,
        "listed_count": len(listed_cards),
    }


# ------------------------------------------------------------------ cards

def probe_offers(
    bank: Bank, task_coords: Dict[str, str], probe_budget: int
) -> Dict[str, str]:
    """The serve-time probe offers: the first `probe_budget` queue rows
    whose cards are servable and eligible for this task. Offers ride card
    reads only; the cap keeps learning from cannibalizing doing. An
    ignored probe stays queued."""
    offers: Dict[str, str] = {}
    queue_path = bank.root / PROBE_QUEUE_RELPATH
    if not probe_budget or not queue_path.is_file():
        return offers
    servable_names = {card.name for card in bank.servable()}
    for line in queue_path.read_text().splitlines():
        match = _PROBE_ROW_PATTERN.match(line)
        if not match:
            continue
        name, _tier, probe_text = match.groups()
        card = bank.cards.get(name)
        if (
            card is None or name not in servable_names
            or not card.eligible_for(task_coords)
        ):
            continue
        offers[name] = probe_text
        if len(offers) >= probe_budget:
            break
    return offers


def _procedure_locations(bank: Bank, card: Card) -> List[str]:
    """A procedure's on-disk payload in the serving checkout: the code dir
    (entrypoint inside it) and the replay dir. A prose-only procedure (not
    yet code-flipped) has no code dir — the documented default is to omit
    the block rather than point at nothing."""
    lines: List[str] = []
    base = bank.root / "procedures" / card.name
    code_dir = base / "code"
    if code_dir.is_dir():
        lines.append(f"code: {code_dir}")
        entrypoint = card.frontmatter.get("entrypoint")
        if entrypoint:
            lines.append(f"entrypoint: {base / str(entrypoint)}")
    replay_dir = base / "replay"
    if replay_dir.is_dir():
        lines.append(f"replay: {replay_dir}")
    return lines


def _reliability_block(card: Card) -> str:
    reliability = card.reliability
    scores = " ".join(
        f"{dim}={reliability.get(dim)}"
        for dim in ("validity", "boundary", "coverage", "score")
    )
    lines = [
        "### Reliability",
        f"- state={card.state} version="
        f"{(card.frontmatter.get('provenance') or {}).get('version')} {scores}",
        f"- plain: {reliability.get('plain')}",
        f"- rationale: {' '.join(str(reliability.get('rationale') or '').split())}",
    ]
    return "\n".join(lines)


def _evidence_block(card: Card) -> str:
    """The full evidence trail — every entry whole (Rule 6: depth selects
    sections, never cuts within one)."""
    lines = [f"### Evidence ({len(card.evidence)} entries)"]
    for entry in card.evidence:
        source = entry.get("source") or {}
        lines.append(
            f"- verdict={entry.get('verdict')} "
            f"trajectory={source.get('trajectory')} ref={source.get('ref')}"
        )
        for field in ("usage", "effect", "note"):
            value = entry.get(field)
            if value:
                lines.append(f"  {field}: {' '.join(str(value).split())}")
    return "\n".join(lines)


def _render_card(
    bank: Bank,
    card: Card,
    with_evidence: bool,
    offers: Dict[str, str],
) -> str:
    """One card at the requested depth: citation tag + the full v2 body
    (the body IS the engineer-facing card, carried whole — Rule 6), plus
    the procedure payload paths, the probe offer when one is queued, and —
    at evidence depth — the reliability block and full evidence trail."""
    lines = [f"[card:{card.name}]", card.body]
    locations = (
        _procedure_locations(bank, card) if card.type == "procedure" else []
    )
    if locations:
        lines.append("\n".join(locations))
    if card.name in offers:
        lines.append(
            f"*probe:* {offers[card.name]}\n*{_PROBE_COST_CLAUSE}*"
        )
    if with_evidence:
        lines.append(_reliability_block(card))
        lines.append(_evidence_block(card))
    return "\n\n".join(lines)


def render_cards(
    bank: Bank,
    task_coords: Dict[str, str],
    card_names: List[str],
    with_evidence: bool,
    offers: Dict[str, str],
) -> Dict[str, Any]:
    """bank_get_card / bank_get_card_with_evidence: full cards on request.

    Anything the index shows can be opened — scope is the reader's
    judgment, never a gate (a visible-but-unreadable card would be
    incoherent). Quarantine stays law: decoys refuse as unknown, and
    non-servable states refuse by name. The co-serving guard names
    contradicts pairs inside one returned set."""
    exposure = "evidence-read" if with_evidence else "read"
    decoys = bank.decoy_names
    got: List[Card] = []
    refused: List[Dict[str, str]] = []
    for name in card_names:
        card = bank.cards.get(name)
        if card is None or name in decoys:
            # Decoys refuse as unknown — naming the quarantine would mark
            # the bait (quarantine is frame-side knowledge).
            refused.append({"card": name, "reason": "no such card"})
        elif card.state not in ("candidate", "active"):
            refused.append(
                {"card": name, "reason": f"not servable (state={card.state})"}
            )
        else:
            got.append(card)
    tensions = _tension_pairs(got)
    sections = [
        _render_card(bank, card, with_evidence, offers) for card in got
    ]
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
        "served": [
            {
                "card": card.name,
                "version": (card.frontmatter.get("provenance") or {}).get("version"),
                "state": card.state,
                "score": card.score,
                "exposure": exposure,
            }
            for card in got
        ],
        "refused": refused,
        "tensions": [list(pair) for pair in tensions],
    }


# ----------------------------------------------------------- probe queue

def compile_probe_queue(bank: Bank) -> str:
    """The VoI-ranked open-probe queue — recompiled by the frame each
    update run, every input ledger-derived, no agent in the ranking.

    Rows are servable, non-decoy cards with an open `probe:` field, in three
    tiers: (1) served-unverified — voi = uncertainty (1 − validity; missing
    counts as 1) × serving exposure (evidence entries whose usage claims
    participation), heavily-served-thinly-verified first; (2) boundary —
    cards in a contradicts pair, whose probe would carve a scope; (3)
    blocked — candidates with zero outcome verdicts, promotion pending
    measurement. Remaining open probes tail the queue by name."""
    decoys = bank.decoy_names
    open_probes = [
        card for card in bank.servable()
        if card.name not in decoys
        and str(card.frontmatter.get("probe") or "").strip()
    ]
    contested = {
        card.name for card in open_probes
        if card.frontmatter.get("contradicts")
    }

    def uncertainty(card: Card) -> float:
        validity = card.reliability.get("validity")
        return 1.0 - float(validity) if isinstance(validity, (int, float)) else 1.0

    def exposure(card: Card) -> int:
        return sum(
            1 for entry in card.evidence
            if usage_claims_serving(str(entry.get("usage") or ""))
        )

    def outcomes(card: Card) -> int:
        return sum(
            1 for entry in card.evidence
            if entry.get("verdict") in ("confirm", "weaken", "refute")
        )

    tier1 = sorted(
        (card for card in open_probes if exposure(card) > 0),
        key=lambda c: (-(uncertainty(c) * exposure(c)), c.name),
    )
    listed = {card.name for card in tier1}
    tier2 = sorted(
        (c for c in open_probes if c.name in contested and c.name not in listed),
        key=lambda c: c.name,
    )
    listed |= {card.name for card in tier2}
    tier3 = sorted(
        (c for c in open_probes
         if c.state == "candidate" and outcomes(c) == 0 and c.name not in listed),
        key=lambda c: c.name,
    )
    listed |= {card.name for card in tier3}
    tail = sorted(
        (c for c in open_probes if c.name not in listed), key=lambda c: c.name
    )

    rows = []
    for card in tier1:
        voi = uncertainty(card) * exposure(card)
        rows.append((card, f"served-unverified voi={voi:.2f}"))
    rows += [(card, "boundary: contradicts unresolved") for card in tier2]
    rows += [(card, "blocked: candidate with no outcome verdict") for card in tier3]
    rows += [(card, "queued") for card in tail]

    lines = ["# Probe queue — value-of-information ranked (derived; frame-compiled)", ""]
    for index, (card, tier) in enumerate(rows, start=1):
        probe = " ".join(str(card.frontmatter["probe"]).split())
        lines.append(f"{index}. [card:{card.name}] ({tier}) — {probe}")
    if not rows:
        lines.append("(no open probes)")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------- pull log

def append_pull_event(log_path: str, event: Dict[str, Any]) -> None:
    """One serving event to the pull log — append-only JSONL beside the
    launch record; the harvester collects both and the exam derives the
    exposure ladder from it. Timestamps are operational (tool calls are
    live agent interaction, never replayed like the index)."""
    path = Path(log_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    stamped = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **event,
    }
    with open(path, "a") as handle:
        handle.write(json.dumps(stamped) + "\n")
