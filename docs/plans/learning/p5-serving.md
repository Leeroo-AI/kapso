# P5 — Serving goes live

**Goal:** the retriever augments the static context notes (brief appended
after them — additive, per MD§5.3); every new campaign
is briefed, tooled, stamped, and exam-before-lesson runs on it (MD§8.5,
MD§4.1, MD§5.1–5.3). **Design sources:** MD§5.1 (both modes, serving record,
co-serving guard), MD§5.3 (the read-only-substrate law + the sanctioned
citation convention), GS§6 (exam mode). **Depends on:** P4. **Doubts:** D3
(remote bank repo now), D5b (framework-core approval for the §5.3 edits).

## Deliverables

`retriever.py` complete (pull tools on the push core); gated-MCP preset
entries; the §5.3 evolve-side edits (gated); exam-before-lesson wiring;
`kapso learn brief`; bank remote + durable local clone serving
(`bank.remote`, `bank.local_path` config — MD§3.1).

## Work items

1. **Pull tools** (MD§5.1): `bank_search(query)` → scope-and-quarantine
   filtered, reliability-ordered **hero shortlist** (whole eligible set at
   bank scale; gap note when thin, never padded); `bank_get(card_ids)` →
   full projections + co-serving guard; both logged to the serving record
   (two exposure levels — attribution binds to `got`). Served through the
   existing gated-MCP registry (`src/kapso/gated_mcp/presets.py`) to
   ideation + implementation sessions only — **never the feedback judge**;
   per-benchmark off-switch in config.
2. **(G) GATED — the §5.3 evolve edits** (approval D5b, diffs before merge):
   push brief appends after the two static context constants
   (`MODELLING_PRACTICE_NOTE` / `FEATURE_ENGINEERING_NOTE` stay the
   permanent base — additive, flipped from replacement 2026-08-22); the
   citation-contract paragraph in ideation/implementation prompts +
   `cards_load_bearing` in the judge template (the one sanctioned
   convention); `bank_head` stamped into campaign meta **by the retriever,
   not by sessions**. Nothing else in evolve changes — the learner runs
   after campaigns, never inside them.
3. **Serving operationally** (MD§3.1): durable local clone as the serving
   source, pinned at launch, network never on the campaign path (unreachable
   remote → serve local head, staleness noted loudly in the brief); single
   writer pushes after each learning run; boxes pull at campaign start. Bank
   remote created per D3.
4. **Exam-before-lesson live** (MD§4.1 step 3, GS§6 exam mode): post-campaign
   chain save → mine → grade(single: writer+verifier) → update; the running
   scorecard curve starts.
5. **Plumbing validation** (MD§8.5): the A/B-able-immediately check —
   founding cards vs the static notes should be ≈ neutral (same content, now
   scoped and cited); run it as the phase's acceptance experiment.

## Tests

- Eligibility law: out-of-scope card never returned; decoy/retired never
  returned (this is also the gauntlet decoy invariant's serving half).
- Serving record completeness: every search and get logged with card
  versions; push stamp reproducible (pure function).
- Judge tool-lock: the feedback judge's session config carries no bank
  tools.
- Local-clone pinning: mid-campaign bank commits do not change what a
  running campaign is served.

## Done gate

One real campaign briefed + tooled end to end; its trajectory saved, mined,
hindcast (a real rung-2 point on the curve), and ingested; the
neutrality check reviewed.
