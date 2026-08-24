# Plan — `Kapso.learn()` facade integration + reviewer-gated E2E

**Design source:** docs/research/learn-api-design.md (§§1-8, user-approved).
**Definition of done (user, 2026-08-24):** the full production loop —
research → learn_knowledge → evolve → learn → evolve — passes on a real
example, and a fresh-context REVIEWER AGENT reads the complete execution
logs and ACCEPTS; a rejection routes feedback to the implementor, the
code is fixed, and the loop reruns until acceptance.

## Phase 1 — implementation (design §7 + §8.4)

1. `src/kapso/kapso.py`:
   - rename `learn` → `learn_knowledge` (pure rename; call sites: cli.py
     has none left for the KG path — grep to confirm none elsewhere).
   - constructor: `bank: Optional[str] = None` param; one-time memory
     resolution (`self._bank_home` = arg | config
     `learning.bank.local_path`); `self.memory` property returning
     `MemoryStatus`.
   - new `learn()` per the §2 contract (dispatch SolutionResult | store
     id | dir; import→mine→exam→lesson; head bookkeeping; push per
     config/kwarg) — composing the exact `cmd_learn ingest` calls.
   - S1 fix: `learn_knowledge()` refreshes `self.knowledge_search` from
     the config preset after a merged run (mirrors `index_kg`'s tail).
   - evolve serving hook (§4): when `learning.serving.enabled` and the
     bank home exists → `prepare_campaign_serving(config, serving_scope
     or {"family": mode or "generic"}, workspace)`; intro appended to
     the problem context; `bank_serving` into strategy params; stamps
     `metadata["bank_head_served"]` (+ `metadata["kg_index"]`) into
     SolutionResult. Serving off → byte-identical path.
2. `src/kapso/learning/lesson_result.py`: `LessonResult` (§3) with
   `admitted` + `explain()`; `MemoryStatus` with `explain()` (same file —
   the facade's learning-side result types).
3. Config: `learning.update_crew.default_version: crew_v4` (Rule 1 —
   the one new key).
4. Class docstring rewrite: the memory model (knowledge + experience),
   the closed loop, `learn_knowledge` naming.
5. Tests (facade-level, hermetic, provider-mocked):
   - dispatch: SolutionResult / dir / store-id; unknown → loud error.
   - idempotency: re-learn of a banked trajectory refused, commit named.
   - exam pin: head recorded before the (stubbed) lesson runs.
   - `LessonResult.admitted` + `explain()` shape.
   - evolve serving staging: enabled+bank → intro in problem context,
     bank_serving in params, stamp in metadata; disabled → byte-identical
     problem context.
   - S1 regression: null-search Kapso + learn_knowledge(merge stubbed)
     → knowledge_search is live afterwards.
   - `Kapso(bank=...)` override beats config.
6. Gate: full curated + learning suites green; commit + push BOTH refs.

## Phase 2 — the E2E production loop (examples/ml_model_development)

Isolated e2e home (production stores untouched): a scratch config
derived from the packaged one overriding `learning.bank.local_path`,
`learning.trajectory_store.local`, run roots → an e2e sandbox dir;
`learning.serving.enabled: true`. KG backends: the RUNNING local
weaviate/neo4j (infrastructure-* containers).

The driver script (examples-style, checked in as
`examples/ml_model_development/run_full_loop.py`):
1. `k = Kapso(config_path=<e2e config>)` — fresh bank via `init_bank`.
2. `findings = k.research("<tabular feature-engineering/model-selection
   question scoped to the example>")`.
3. `k.learn_knowledge(findings)` → KG merge; assert knowledge_search
   live (S1 in production).
4. `sol1 = k.evolve(goal=<example goal>, data_dir=..., initial_repo=...,
   time_budget_minutes=~25)` — consults KG; serving on the founding bank
   (honest gaps).
5. `lesson = k.learn(sol1)` — mine→exam→lesson on the real crews;
   assert `lesson.admitted` semantics hold (either way is legal; the
   REPORT must be coherent).
6. `sol2 = k.evolve(goal=..., time_budget_minutes=~25)` — consults KG +
   updated bank; stamps show both.
7. Persist EVERYTHING under the e2e sandbox: driver log, both campaign
   workspaces, lesson/exam reports, serving records + pull logs, memory
   explain() snapshots at each step.

## Phase 3 — the reviewer gate (accept-or-feedback loop)

- Reviewer = a FRESH-context agent (Agent tool), no implementation
  knowledge, given: the acceptance rubric + the e2e sandbox paths +
  read-only instructions. It writes `verdict.yaml`
  ({verdict: ACCEPT|REJECT, findings: [...]}).
- Rubric (the contract of "done"):
  R1 every stage ran to completion with exit 0 and produced its
     artifacts (findings, KG merge report, sol1, lesson, sol2);
  R2 knowledge path PROVEN: post-learn_knowledge search is live and
     evolve's ideation actually had the KG gates mounted;
  R3 experience path PROVEN: sol1's workspace carries a serving record
     (mode agentic, founding head) and sol2's serving record shows the
     POST-LESSON head; LessonResult heads match the bank repo's log;
  R4 exam-before-lesson: exam report exists and its bank_head ==
     bank_head_before (the pin);
  R5 provenance stamps present in both SolutionResults;
  R6 no swallowed errors: driver log free of tracebacks that were
     retried-over silently; every WARN accounted for;
  R7 the loop is genuinely closed: something the lesson banked is
     VISIBLE to sol2 (its index/pull log lists the new card(s)).
- REJECT → findings go to the implementor (me), code/plan fixed, Phase
  2 reruns (fresh sandbox), reviewer re-invoked with the new logs +
  its prior findings. Loop until ACCEPT. Verdicts + iterations all
  retained in the sandbox for the record.

## Budget & ops

Models per config (evolve GENERIC: opus-5 sessions; learning crews:
fable-5; costs waived per standing grant). Wall estimate per loop
iteration: research ~10m, learn_knowledge ~30m, evolve 2×25m, learn
~45m, review ~10m ⇒ ~2.5h. Monitors: driver runs in background with a
milestone monitor + the 15-min status habit; commits after each green
phase, pushed to BOTH refs (sync policy).
