# P4 — Bank + update crew, developed against the graders

**Goal:** the founding bank exists; update-crew v1 runs end to end; crew
versions iterate against the scorecard with keep-best banking (MD§4.4,
MD§8.4). **Design sources:** MD§3.1–3.3 (bank home, card schema, reliability
algebra), MD§4.3, UC in full (run dir + worksheet/journal/findings schemas,
lead prompt, card-writer + five-specialist + critic + assessor defs, frame
contract, docket), MD§5.2 (evidence admission), GS§4 (settlement lift).
**Depends on:** P3. **Doubts:** D3 (repo host/timing), D4, D5 (none here —
no framework-core), D6, D7.

## Deliverables

`bank.py` (bundle model, card schema validate, diff invariants, indexes,
`lr_` tags, sightings); the founding bank (local repo per D3); update-crew
instruction material (`src/kapso/learning/crews/update/` — everything lifted
from UC); `learner.py` complete (staging incl. seeding, validation, commit,
report assembly); development driver; `kapso learn update`;
`learning.update_crew:` config (UC§0/§1/§7: models per role,
`repair_rounds` 1, `sightings_expiry_batches`, `dup_threshold`, batch
trigger; `learning.codify.min_recurrence` read by the seeder — the codify
*run* itself is P7 and its rows ship config-gated off until then).

## Work items

1. **Bank model** (MD§3.1–3.2): OKF bundle load/validate (frontmatter schema
   for both card types incl. procedure fields; reserved OKF fields; indexes;
   `log.md`; `sightings.md`); path-as-identity; link/edge extraction
   zero-LLM; derived `index/` rebuild (edges; embeddings for the
   consolidation shortlist only — MD§3.1/§5.1); `lr_<id>` post-commit tags.
2. **Diff invariants validator** (UC§7 item 2 + MD§4.3): append-only
   evidence/log; version ⇔ log-entry; `contradicts` both sides; retirement
   is a move; decoys untouched; sightings appends/expiry-removals only;
   MERGE shape (≥2 parents retired + linked, evidence by reference, never
   copied); GENERALIZE shape (candidate state, unseen-family probe,
   coverage = seen families); representation-flip-requires-green-codify-run
   (validator present now, exercised at P7).
3. **Evidence admission** (MD§5.2): source resolves + re-grep; usage vs
   serving record (probe claims via settlement-matching per the read-only
   substrate rule); verdict earnable; independence (same run ids = one
   measurement). **Reference transparency** as the one shared function
   (assessor scoring + codify seeder counting through merge/generalize
   founding refs — CD§1/§5).
4. **Founding bank** (MD§8.4): repo init (D3); the 9 practices back-filled
   as cards with founding evidence; ~5 pitfall insights + ~3 procedures from
   the wave-4 trace as seed; OKF conformance check; decoy cards planted
   (quarantined id range, MD§6).
5. **Update-crew frame** (UC§1/§7): run dir + `inputs.yaml`; worksheet
   seeding — batch classes (lift / card-candidate / serving-feedback /
   probe-check) from hindcast reports + docket classes (dup-merge / tension /
   generalize / expiry / codify) from the pre-run bank; `--split` twin
   checks (batch ∩ held_out = ∅; batch-own reports only); launch lead;
   six-step validation; one repair bounce; commit + tag + report assembly
   (MD§4.3 report spec: frontmatter + health block + the six body sections).
6. **Crew instruction set** (UC§2–§6 + CD§6 pointer): lead prompt;
   card-writer; card-merger, card-generalizer, tension-resolver,
   expiry-sweeper (procedure-codifier staged but its rows gated off until
   P7); critic; reliability-assessor. All content lifted from UC/CD — the
   docs are the prompt source of truth.
7. **Development driver** (MD§4.4): chronological learn-set batching under
   `--split`; disposable candidate banks; scorecard per crew version via
   `kapso learn grade`; **keep-best banking of learner versions** (a ledger
   of crew versions + their scorecards; replacement only on `accept`).
8. **First iteration cycle**: run crew v1 over the learn-set → candidate
   bank → full P3 grading (first real hindcasts + first real duplicate and
   stability trap executions) → human review of the first bank commits →
   crew v2 from findings. This item is the phase's substance, not a
   formality.

## Tests

- Invariant fixtures: each violation class trips exactly (append-only edit,
  missing log entry, one-sided contradicts, decoy touch, copied merge
  evidence, GENERALIZE without probe).
- Coverage arithmetic: worksheet row without journal verdict fails; diff
  hunk with no journal trace fails; PASS without rebuttal fails.
- Reference-closure recurrence: the CD§1 union example (parents sharing a
  source campaign → union 3, not sum 4).
- Evidence admission: quoted-number re-grep miss rejects; same-run-ids
  double-count rejects; settlement lift preserves delta + refs.
- Split twins: held-out id in a `--split` batch fails before staging.

## Done gate

Founding bank conformant; crew v1→v2 cycle completed with scorecards; human
review signed; traps executed; keep-best ledger live.
