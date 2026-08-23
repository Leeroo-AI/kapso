# Serving v2 — agentic retrieval over the bank (design for iteration)

**Status:** DESIGN — not implemented. This doc is the iteration surface; on
approval it supersedes the push-brief/pull-tools section of the serving design
(learn-from-trajectories §5.1) and the §5.3 brief-injection wiring. Companion
edits it forces on the learning side are listed in §7.

**Decision driver (user, 2026-08-23):** stop pushing selected card bodies.
Push only an *introduction* to the bank and per-module guidance on *when* to
consult it; give every module agentic search tools over the whole bank —
"like the index of a book" — instead of frame-side top-k selection.
Inspiration: PageIndex (VectifyAI) — vectorless, reasoning-based retrieval.
Refined 2026-08-23 (user redlines): no pitfall card type anywhere; index
lines simplified (no visited-here, no probe flags); intro stripped to what
the bank is + the tools; explicit tool names.

---

## 1. Why the current design loses (measured, mini A/B 2026-08-22)

The current serving is **push-heavy, pull-dead**: the frame picks top-k cards
by reliability score at minute zero and injects their full bodies; two query
tools (`bank_search(query)` / `bank_get`) exist but were called **zero times
across all six arms measured**.

- **Plan-blind selection (wave A):** the retriever ranks before any plan
  exists. Lane 3 committed to an AFT survival reformulation; the bank's
  exactly-scoped guardrail (`direct-target-head-anchors-survival-reformulations`,
  scope_conditions matching the plan verbatim) scored 0.45 and lost the
  `k_insights: 4` cut to generic 0.65–0.73 cards. Half a campaign burned;
  the post-mortem re-derived the unserved card's rule by hand.
- **Push works only where the steer is novel (wave B win, waves A/C flat):**
  pushed bodies steered a thin-dataset campaign to a real win, but on mined
  datasets the same four generic cards were redundant with the static notes.
  Fixed top-k cannot adapt to which situation it is in.
- **Query-shaped pull has too much friction:** `bank_search(query)` demands
  the session invent a query before knowing what the bank holds. Nobody ever
  did. Browsing an index is cheaper than composing a search.

## 2. What we take from PageIndex (and what we don't)

PageIndex replaces vector similarity with an LLM *reasoning over a
hierarchical, human-readable index* — title + one-line summary per node,
natural document units (no chunking), the reader bringing its full working
context to the scan, and a transparent retrieval trajectory.

Adopted, mapped to the bank:

| PageIndex | Serving v2 |
|---|---|
| tree node = natural section | card = natural unit (already atomic) |
| node title + summary | card name + hero line |
| page range (where to go deeper) | the get-card tools by card name |
| LLM navigates with full conversation context | module scans the index holding its own plan/failure — situational match happens in the reader's head, not in a ranker |
| traceable retrieval trajectory | pull log + serving record exposure ladder |
| no vectors, no chunking | no embeddings anywhere; selection is reading |

Not adopted (Rule 10): PDF structure extraction (nothing to extract — cards
are born structured); deep tree navigation (29 cards fit one page; §8 gives
the growth path); retrieval accuracy benchmarks (our certification is the A/B
arm, §9).

The one principle that carries the whole redesign: **"similarity ≠
relevance" becomes "score-rank ≠ relevance."** Wave A's guardrail was the
lowest-scored card in the bank and the single most relevant document to lane
3's plan. Only the agent, holding the plan, could have known — so the agent
must be the selector.

## 3. The three tools

Served through the existing gated-MCP registry to ideation, selector,
implementation, and lens-planner sessions — never the feedback judge (§5.3
law, unchanged). All three are read-only, deterministic, filtered by task
eligibility + quarantine exactly as today, and logged to the pull log.

Names state exactly what returns: `bank_index` (the index page),
`bank_get_card` (the card), `bank_get_card_with_evidence` (the card plus the
evidence it stands on).

### 3.1 `bank_index` — the book's index (entry point)

**Input:** none required. Optional `section: insights | procedures` filter
(irrelevant at 29 cards; the contract that survives scale).
**Output:** the WHOLE eligible set, one line-group per card,
reliability-ordered within type sections. No query, no ranking cut —
browsing replaces querying.

Line format — three facts per card, enough to decide "open or skip":

```
## Insights
[card:renewal-units-beat-raw-event-counts] score 0.73
  — Test whether renewal units beat raw event counts when actions cluster in bursts
  applies-when: repeated actions per entity cluster in bursts within short periods
[card:direct-target-head-anchors-survival-reformulations] score 0.45
  — Keep a direct binary-target head anchored when reformulating to survival/hazard objectives
  applies-when: fixed-horizon binary target is also reformulated into survival, hazard, or waiting-time auxiliaries
...

## Procedures
[card:six-fold-forward-gate-runner] score 0.61
  — Precommitted expanding-origin gate harness with clustered uncertainty readout
  applies-when: admitting feature/model changes on temporal tasks; cost scales with dataset size

gaps: no procedure covers <task-family> end-to-end
```

Design notes:
- `applies-when` is the card's `scope_conditions` (compressed by the
  card-writer at authoring time, not at serve time) — **this line is the
  wave-A fix**: an agent scanning with "I'm about to reformulate to survival"
  in its head matches it by reading, no ranker involved.
- The closing `gaps:` line (only when non-empty) preserves v1's honest
  "what the bank does not know".
- Deliberately absent: visited-this-dataset flags, probe flags, census
  counts — the index is name + one-liner + score + applies-when, nothing
  else. Dataset history and probes surface on the card itself (§3.2).

### 3.2 `bank_get_card` — the card body

**Input:** `cards: [name, ...]` (co-serving guard as today: reading a card
pulls in its tension partners' index lines so contested scope stays visible).
**Output per card:** citation tag + the full v2 body (title → Rule →
Is-this-your-situation → What-to-do → Why-believe-this → Confidence) — the
body already IS the engineer-facing card; nothing else added. Plus:

- if the card has a queued probe (at most `probe_budget` offered per
  campaign): the probe paragraph + the cost clause (§5, wave-A lesson) —
  an optional measurement offer, visible only when the card is opened.
- **if a procedure: the code location** —
  `code: <bank_checkout>/procedures/<name>/code/` (entrypoint `code/main.py`)
  and `replay: <bank_checkout>/procedures/<name>/replay/` — the on-disk
  serving clone paths, so a session can read, copy, or execute the harness
  directly.

### 3.3 `bank_get_card_with_evidence` — the card plus its track record

**Input:** `cards: [name, ...]`.
**Output per card:** everything `bank_get_card` returns **plus** the machine
ledger's trust surface: the reliability block (validity/boundary/coverage/
score + rationale + plain line), the full evidence entries (per-entry
trajectory, ref, verdict, note), and current state/version. Procedures again
include code + replay locations.

Purpose split (why two depths): `bank_get_card` is for *acting on* a card
mid-flow; `bank_get_card_with_evidence` is for *betting on* one — before a
lane commits real budget to a card's advice, it can audit where the claim
comes from and how it has fared. Rule 6 note: neither tool ever truncates a
body or an evidence note; depth selects *which sections*, never cuts within
one.

## 4. The push side — an introduction, not a payload

`knowledge_section()` keeps the two static notes (additive protocol,
2026-08-22 decision) and replaces the compiled brief with a fixed short
**intro** (draft, to iterate):

```
## Knowledge bank (measured practice from past campaigns)

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
  advice.
```

That is the whole push. When and how to use the tools is each module's
paragraph (§5); the citation convention also lives there, not in the intro.

## 5. Per-module integration — "when to use it"

Each module's prompt gains one short, imperative paragraph. Drafts:

**Ideation (and its selector):**
> Before writing ideas, call `bank_index()` once and scan it against the
> task: open (`bank_get_card`) any card whose applies-when matches a
> direction you are considering, and steal or steelman it. When an idea
> adopts a card's move, cite `[card:<name>]` in the idea. A card you open
> may carry a probe — an optional measurement offer; adopt one only if its
> protocol is affordable at this dataset's scale, and say so explicitly.

**Implementation sessions:**
> Consult the bank at decision points, not continuously: (a) before
> committing a lane to a reformulation, architecture family, or any bet
> worth >30 min of budget — `bank_index()` and scan applies-when lines
> against your plan; (b) when a gate fails in a way that surprises you —
> the failure mode may be carded; (c) before adopting a card's advice
> wholesale — `bank_get_card_with_evidence` it and weigh its evidence.
> Cite `[card:<name>]` in specs and features_history entries the card
> shaped. Following a card is never mandatory; departing from one you read
> is worth one line of why.

**Lens planner:** gets the intro; when allocating lanes it may call
`bank_index()` to check whether a planned lane theme is carded (either as
support or as guardrail).

**Feedback judge:** unchanged — tool-locked, card-blind (§5.3). The
`cards_load_bearing` template field stays for now; its inertness is a known
separate issue (wave-B finding: judges emit none even under heavy real
uptake) tracked in §9 open questions.

**Context handler (`knowledge_section`):** intro instead of brief; static
notes untouched; additive order preserved.

## 6. Serving record v2 and the exposure ladder

`mode: push` dies. The record becomes the trace of an agentic session:

```yaml
mode: agentic
bank_head: <sha>
gaps: [...]
exposure:                # per card, highest level reached
  - {card: renewal-units-beat-raw-event-counts, level: read, by: [ideation, generic_exp_1]}
  - {card: direct-target-head-anchors-survival-reformulations, level: indexed}
  - {card: six-fold-forward-gate-runner, level: evidence-read, by: [generic_exp_2]}
index_calls: 3
probes_offered: [forward-fold-referee-governs-temporal-regime-change]
```

Exposure ladder (replaces v1's `got`/`searched`), each level derived from the
pull log, all frame-side:

`offered` (intro seen — every card, implicitly) → `indexed` (a bank_index
call returned its line) → `read` (`bank_get_card`) → `evidence-read`
(`bank_get_card_with_evidence`) → `cited` (`[card:]` in specs/living docs)
→ `probe-settled`. Attribution binds at `read` and above; `cited` keeps its
rung; the grading side maps v1 vocabulary onto this ladder (§7).

## 7. What dies, and the companion edits (Rule 7 — no dual paths)

**Deleted:** `compile_brief`'s top-k selection and card rendering into the
context; the pitfall card category end to end — the `k_pitfalls` quota, the
`PITFALL_TAG` routing in the retriever, and the "pitfall guardrails" brief
section (a hazard-shaped lesson is simply an insight whose applies-when
names the hazard); `k_insights`/`k_procedures`/`unvisited_discount` config
knobs; the probe rider on the brief (probes surface on card read, §3.2);
`bank_search(query)` and `bank_get` (replaced by the three tools);
`mode: push` records.

**Survives unchanged:** bank_head stamping (frame-side), citation contract
(now carried by module prompts, §5), quarantine/decoy filtering at the tool
layer, judge tool-lock, serving record as the graded receipt, `probe_budget`.

**Companion edits (learning side, same change-set when implemented):**
- Grading contracts (`report_writer_prompt.md` and friends): serving-section
  vocabulary moves to the exposure ladder. "Withholding is not a serving
  event" generalizes to "an empty exposure list above `offered` is a
  tools-unused campaign — serving: null unless the intro itself misled."
  SERVED-USED / UPTAKE-FAIL re-anchor on `read`+`cited` rather than `got`.
- Update-frame serving-feedback worksheet rows: same vocabulary swap.
- Card-writer: `scope_conditions` becomes a first-class authored field with
  a floor (it is now user-facing as `applies-when` — today it is optional
  frontmatter); one compressed line, abstraction law applies.
- Tests: serving-wiring suite re-pins intro + three tools; exam fixtures
  re-shape serving records.

## 8. Scale path (documented, not built)

At ~100+ cards the flat index outgrows one page. The PageIndex-shaped
growth: `bank_index()` returns section summaries (tag-level nodes: one line
per theme with card counts) and `bank_index(section=...)` opens a node —
same three tools, one more level of the same reasoning loop. The bank's
existing `index.md`/tag structure is already the tree; no new storage.
Trigger: when a full index exceeds ~150 lines.

## 9. Review (adversarial, against our own measurements)

**R1 — the dead-tool risk inverts: what if nobody calls anything?**
Strongest objection, backed by our own data (0 pull calls in 6 arms). Wave
B's win came from *pushed* bodies; under v2 that campaign wins only if
ideation actually calls `bank_index()` and opens the renewal card.
Mitigations: (a) query-friction removed — `bank_index()` takes no arguments,
browsing beats composing; (b) imperative per-module cues with named moments,
not vague availability; (c) the serving record makes non-use visible and
gradable, so the learning loop prices it. Residual risk is real and is
exactly what the certification A/B (§R4) must measure. If sessions still
don't call, the fallback lever is one mandated `bank_index()` at ideation
start (still agentic selection, guaranteed exposure of the index page) —
deliberately NOT in the base design to keep uptake honest; pre-registered as
the first amendment if the A/B shows tool-silence.

**R2 — do we lose wave-B-style guaranteed steering?** Yes, by design: v2
trades guaranteed exposure of 4 frame-chosen cards for agent-chosen coverage
of all 29. The bet is that an index line (name + hero + applies-when) is
enough scent for a competent agent to open the right cards. Wave-B
counterfactual check on the real transcript: its ideation *did* read and
weave all four pushed cards — an agent that engaged that deeply with pushed
text plausibly opens them from a scented index; unproven until §R4 runs.

**R3 — context cost.** v1 pushed ~4 full bodies (~3-5k tokens) always. v2
pushes a ~12-line intro always + index pages (~2-3 lines per card per call)
+ only the bodies actually opened. Strictly cheaper unless a session reads
>4 cards — which is then presumably worth it.

**R4 — certification.** The redesign is itself a serving-generation change,
so it takes the same gate as a bank generation: an A/B wave pair —
candidate = intro+tools (v2), incumbent = current push brief (v1), same
bank head, same additive-notes base — before v2 becomes the default. KPIs:
score delta, tool-call counts (did pull come alive), exposure→uptake→payoff
chain in the exam, per the TIME-based efficiency contract.

**R5 — open questions for iteration:**
1. Should `bank_get_card` on a probe-carrying card *require* an explicit
   accept/decline line in the spec (cheap probe-settlement signal), or is
   that an output obligation §5.3 forbids? (Leaning: forbidden; settlement
   stays learning-side.)
2. `cards_load_bearing` judge field: fix its input path (route living-doc
   citations into the judge's view) or delete the field (Rule 10)? Needs the
   wave-B inertness root-cause first.
3. Does the lens planner get full tools or index-only? (Draft says full;
   cheapest to trim later.)
4. Dataset dossier (wave-C gap) — orthogonal card *species*; if adopted, it
   is just more index lines here, no tool changes. Tracked separately.
5. Procedure execution: `bank_get_card` hands out code paths; do we also
   want a gated executor tool, or is copy-and-adapt the right contract?
   (Leaning: copy-and-adapt; an executor is an obligation-shaped coupling.)

## 10. Implementation map (for the build, later)

`retriever.py`: `compile_brief` → `compile_intro` (intro text + gaps);
`pull_shortlist` → `render_index` (applies-when lines + gaps footer; delete
`PITFALL_TAG` routing and the visited/discount machinery);
`pull_projections` → `render_card` / `render_card_with_evidence`
(procedures attach code/replay paths; probe offers ride card render under
`probe_budget`). `serving_launch.py`: returns intro instead of brief;
record v2 skeleton. `gated_mcp/presets.py`: three tool entries replace two.
`benchmarks/relbench/context.py`: intro injection. Prompt files:
ideation/implementation/lens-planner paragraphs (§5). Learning side:
grading prompts + update-frame vocabulary + card-writer scope_conditions
floor + tests (§7). Config: delete `k_insights`/`k_procedures`/`k_pitfalls`/
`unvisited_discount`, keep `probe_budget`.
