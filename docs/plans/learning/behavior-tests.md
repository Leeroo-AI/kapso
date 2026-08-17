# The behavior suite — semantic production tests with agentic review

Mechanical tests (Rule 9, per phase) prove the machinery holds its
invariants. This suite proves the **behavior is semantically right**: that a
merge really unified one mechanism, an abstraction really transfers, a
replay really certified faithfulness — judged by an **agentic reviewer**
against fixtures whose correct outcome we authored and therefore know.
It generalizes the gauntlet's trick (planted input, known answer) from
integrity arithmetic to semantic judgment.

**Standing contract:** every scenario = a versioned fixture (engineered mined
views / banks / archived-run slices with the ground truth written down in
the fixture's `truth.md`) + a run of the *real* machinery (real crews, real
frames — never mocks of the thing under test) + a read-only **reviewer
session** with the scenario's rubric, emitting `{verdict: PASS|FAIL,
rationale}` per the no-naked-tags rule. **Verdicts are gates**: a semantic
FAIL blocks learner-version promotion exactly as a gauntlet FAIL does
(gates dominate scores). The reviewer runs on a different model than the
crew under test (cross-model, per the D4 split). Config:
`learning.behavior_tests: {reviewer: {cli, model, effort}, scenarios: […]}`.

**Cadence:** the full suite runs per learner-version candidate (development
regime, beside gauntlet + hindcast) and before any promotion; individual
scenarios on demand (`kapso learn behave <scenario|--all>`).

## Scenarios

| # | Behavior under test | Fixture + known truth | Correct behavior (what the reviewer verifies semantically) | Lands with |
|---|---|---|---|---|
| B1 | **Routing** | a batch with four engineered observations: same-mechanism-as-existing-card; same-*vocabulary*-different-mechanism (the lexical trap); genuine first sighting; infra junk | ATTACH / SPAWN / SIGHTING / PASS respectively — and the SPAWN rationale must *name* the mechanism difference, not just hedge | P4 |
| B2 | **Abstraction** | 2–3 engineered sibling observations sharing one causal mechanism under different surface bindings | the induced card states the mechanism at the right generality: no bound identifiers in the fact, MDL-compact, scope matching exactly the evidence — a transferable abstraction, not a memory | P4 |
| B3 | **Merge** | a bank with two true twins (one mechanism, disjoint wording, split ledgers) and one decoy pair (shared vocabulary, different mechanisms) | twins merged — successor's unified fact faithful to *both* parents, references correct; decoy pair declined with the distinguishing argument | P4 |
| B4 | **Generalization** | three family-scoped cards, same mechanism, agreeing ledgers | domain successor born candidate; its fact is the true common mechanism (not a vague union); its probe asks exactly the unseen-family question | P4 |
| B5 | **Tension resolution** | pair A: ledgers genuinely support a region split; pair B: engineered undecidable | A → SPLIT with a mechanism-vocabulary boundary citing both sides (not an ad-hoc quarantine); B → CONTESTED with a probe that would *actually discriminate* the two claims | P4 |
| B6 | **Reliability & lifecycle** | three engineered ledger histories: steady confirms; one refute among confirms; contested pair | scores + rationales read as calibrated (the refute moves boundary/validity sensibly, not catastrophically); transitions match the algebra's spirit, not just its bounds | P4 |
| B7 | **Hindcast honesty** | a held-out fixture trajectory whose discoveries we planted (so HIT / MISS-UNCARDED / MISS-NOVEL ground truth is known), against a fixture bank | the report classifies every planted discovery correctly, the NOVEL attestation survives the verifier's re-search, and the rationale states the binding factor truthfully — the exam itself is being examined | P3 (fixture bank) → re-run P4 |
| B8 | **Serving relevance** | a task descriptor + fixture bank containing relevant cards, irrelevant same-vocabulary cards, and a planted contradicts pair | the brief carries the semantically relevant cards, the noise stays out, the gap analysis names the true gaps, the co-serving guard names the planted tension | P5 |
| B9 | **Codify faithfulness** | a fixture procedure + archived implementation where a *plausible shortcut* reproduces the number without the method | the feedback judge rejects the shortcut implementation (faithfulness), accepts the faithful one; the final replay/ asserts the card's `expected_outcome`, not something weaker | P7 |
| B10 | **End-to-end learning effect** | a mini-corpus of 3–4 fixture trajectories engineered to contain one transferable lesson + noise | after save → mine → update → grade: the lesson exists in the bank as an abstraction (B2 standard), and hindcast foresight on a planted 4th sibling beats the empty-bank baseline — the whole loop demonstrably *learned* | P4 (mini), extended P5 |

## Fixture sourcing — real by default

Realism comes from the relbench corpus; ground truth comes from a minimal,
recorded intervention. Three modes, every scenario in exactly one:

- **Selected-real** — untouched corpus material whose truth we already know
  from the documented forensics (the wave-4 deep analysis; the D1 churn
  cluster's known recurring lessons): B2, B7 (real trajectory + assembled
  fixture bank), B10, B5-preferred (a natural cross-dataset contradiction).
- **Doctored-real** — real material with one localized, recorded edit that
  implants the truth (the implant/duplicate-trap lineage): B3 (a real card
  reworded as its twin; a real shared-vocabulary pair as the decoy), B5
  fallback (sign-flip in a copied mined view), B9 (the authored shortcut
  beside the real implementation), B1's lexical trap.
- **Frozen-real** — early artifacts of the system itself, frozen: founding
  bank cards (B1 targets, B3/B4 material), first-crew outputs where a
  scenario needs a bank in a known state (B6 ledgers, B8's bank).

Two rules keep this honest: **every fixture cites its source refs**
(trajectory / run / card — the standard referencing discipline), and its
`truth.md` records source + edits + expected outcome, **with corpus
citations for any truth-by-our-reading** — selected-real truth is only as
good as the forensics behind it, so truth files get human sign-off at
creation. Fixture construction is a per-phase work item (mined material
exists from P2, cards from P4), matching the scenarios' phase mapping.

### Concrete candidates known today

Pinned now: **B7** = the wave-4 rel-amazon/user-churn bundle (line-by-line
forensics already documented in the mining-prompts doc §0). Candidates named:
**B2/B10** lessons — the wave-4-documented re-derivation case (the resolution
diagnostic rebuilt per lane) and the 9 practices as truth sources; exact
instance refs pinned at P2 from the mined views. **B9** — a
forward-gate-class procedure, chosen at P7 from what P4 carded. **B5** — the
P2 fixture-candidate hunt (below) searches for a natural cross-dataset
contradiction; sign-flip doctoring is the fallback. **B1/B3/B4/B6/B8** are
recipe-only by necessity (their material — founding-bank cards, first-crew
outputs — exists from P4). The P2 mining reports carry a **fixture-candidate
hunt**: recurring lessons with ≥2 instance refs, natural opposing-sign
findings, and procedure recurrences are flagged as they surface, so fixture
selection is a pick from a documented list, not a search.

## Reviewer contract

One rubric per scenario, shipped beside its fixture. The reviewer: reads the
fixture's `truth.md`, the run's artifacts (bank diff, journal, report,
brief — whatever the scenario touches), answers the rubric's semantic
questions, and returns per-question findings plus the scenario
`{verdict, rationale}`. The reviewer never fixes anything and never sees
other scenarios' outcomes. A FAIL's rationale must name the offending
artifact — it is a bug report, and it feeds the next crew version exactly as
critic findings do.

## What this suite is not

Not a replacement for Rule 9 tests (mechanics stay code-asserted), not a
score (no partial credit — semantic correctness of a merge is not 70%
achievable), and not run in production operation (it is a development-regime
and pre-promotion instrument; live behavior is measured by the graders).
