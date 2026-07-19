# Cross-run knowledge — experiment memory + the expert repo

Synthesis of three judged designs. Base: C (minimal-delta), ranked first by all three judges. Grafts: A's signing discipline, bug-class screen, and measurement KPIs; B's alarm wording and maturity-stage lint suite. Everything judged fatal or theater is dropped (§6).

## 1. Research digest (only what shaped the design)

- **CoALA memory taxonomy** — episodic/semantic/procedural is the organizing frame; the design maps all three onto artifacts kapso already has rather than new subsystems.
- **ExpeL** — exemplars with bindings plus fresh abstraction by the consuming agent transfer; compiled rules don't. This is why lessons render as bound episodes ("run 18, arena×1.7B, FAILED: …"), never as distilled rules.
- **Add-all study (arXiv 2505.16067)** — 2,400 uncurated records: 13% vs 39% curated. Governs the mandatory sliding window: archive everything, render a bounded window.
- **ETH generated-context result** — LLM-generated context files *decreased* success 2–3% at +20% cost. Kills auto-generated KNOWLEDGE.md/directory maps; the delivery channel is the store itself.
- **TroVE compute-matched** — library gains largely vanish under compute matching; usage-frequency tiering needs episode volume that 2–3 iterations/run × 10–30 runs never produces. Kills candidate/core tiers, reuse counters, card-embedding dedup; motivates A's KPI discipline (transfer measured, not assumed).
- **Faulty-consolidation + token-rent results** — LLM consolidation overgeneralizes (the decision-tree minting step); skill modules often don't pay their token rent. Kills any LLM curation pass; commit ef0127c5 (difficulties ARE the lesson artifact) already made this call in-repo.
- **Voyager** — execution-verified admission is the honest core of "battle-tested": nothing enters the seed that didn't actually run and get judged.

## 2. The retro-simulation

Condensed per-run harvest, what a seeded successor would have received:

| Run | Harvestable | Priced counterfactual |
|---|---|---|
| 13 | RM-stage candidate (untried), early lessons | Plan died with the VM (~5h/$30); run 16 lost ~1.5h re-deriving lessons |
| 16 | 0.4815 record; FAILED DPO with diagnosed cause (static mismatched pairs); cwd-bugged wrapper | Raw score mis-signs as SUCCESS — actually PLATEAU, 36 under tier |
| 17 | 89.64 teacher-distill; teacher-choice/decoding-bake/template-parity derivations; promote.py with 53GB copy bug | Kit-worthy scripts AND a latent bug in a high-scoring run |
| 18/19 | Eval-wrapper cwd bugs (again); same-day re-derivation of the same three facts as 17 | ~1–2h/run rebuilt scripts; three parallel language-mix P1s from one unrun measurement |

Family-entry episodic+semantic transfer was priced at ≈+28 cell points; toolkit at ≈1–2h/run. The simulation surfaced three failure modes, each answered structurally:

1. **Wrong-sign retrieval.** An unscoped "greedy > sampling" invariant carried into a judge-eval run is anti-knowledge (temp-1.0: 0.081 vs 0.378; >20 points). Answer: no rules exist — only exemplars with mandatory bindings (run/task/model/sign/date); the render raises if a binding is missing, so an unscoped rule is *unrepresentable*. The duplicate-alarm variant (run 17's on-policy DPO cosine-matches run 16's FAILED static-pairs DPO; suppression kills the +40-point candidate) is answered by the sign-conditional advisory alarm (§3).
2. **Staleness.** Run-13-era self-censoring caution rendered post-relaxation misleads. Answer: sliding window (`seed_max_runs`) deletes without curation; harvest stamps `{date, kapso_commit}`; prior eval profiles are archived as exemplar context, never rendered as current fact — profiles are re-measured every run.
3. **Decision-tree drift.** "Qwen3-30B for arena×Qwen" would have shipped the wrong teacher for SmolLM3. Answer: no generalization step exists to mint the rule (harvest is copy+stamp), and at toolkit maturity the identity lint becomes executable tests (§3).

Plus two the judges added: **mis-signing** (mechanical score-vs-parent stamps run 16 SUCCESS) — answered by a *required* human verdict for scored runs; **defect amplification** (verbatim exemplars re-ship the cwd/53GB bug classes) — answered by the mechanical bug-class screen at harvest.

## 3. Design

**One durable prefix per domain** — `<knowledge.root>/<domain>/` (GCS for posttrain, local dir for relbench) — holding a merged `experiment_history.json` and a `starter_kit/` directory. Cross-run transfer is file placement, not new memory machinery.

### Three layers

| Layer | Content | Lives | Delivered by |
|---|---|---|---|
| Episodic | Full prior records: solution, feedback, difficulties, score, sign, embedding, candidate pool + selector reasoning (ideation-v2) | Merged domain store, seeded to `workspace/.kapso/experiment_history.json` | Existing gate tools; `search_similar` spans origins out of the box (embeddings persist on records) |
| Semantic | Signed `technical_difficulties` + feedback **on those same records** (ef0127c5: difficulties IS the lesson artifact — no second store) | Same file | Generated `PRIOR RUN LESSONS` block in problem context |
| Procedural | The ~8 scripts every run rebuilds, verbatim from each seeded run's best branch | `task_dir/starter_kit/run<id>_<task>_<score>/` + generated provenance README + that run's `eval_profile.md` | Path rendered into problem context; agent copies/adapts. **Not** `initial_repo` — workspace boots empty |

Skipping workspace seeding sidesteps four verified hazards at once: the never-benchmark-exercised `is_seeded` path, the RepoMemory bootstrap clobber (base.py:507–540 → manager.py:670–728), the resume+seed collision (experiment_workspace.py:264–267), and git-history contamination (a once-committed leak survives deletion in every judged clone). Seeded records are structurally retrieval-only: `node_history` is built in-run or checkpoint-restored in `benchmark_tree_search.py`, never from the store — seeded records can never become parents.

### Harvest — who runs it, when

Two stages. **Stage 1, automatic, already shipped:** the periodic rsync + self_destruct sync (run_startup.sh:147–149/:25–33) delivers `.kapso/{experiment_history.json, repo_memory.json, run_state.json}` for every run, including killed ones — run 13's dead-VM loss is covered with nothing to build. **Stage 2, mechanical, at fetch:** `benchmarks/shared/harvest.py`, invoked from `20_fetch_results.sh`, runs on fetched results — including stopped runs — with zero LLM curation and zero standing human session. A wave can never launch on knowledge frozen by a skipped meeting. Declared inputs: the run's store **and `run_state.json`** (parent linkage for mechanical signs — judge-verified present in synced results), the best branch working tree, eval data (local, for the sweep), and `verdict.json`.

### Admission + pruning = "battle-tested" operationalized

- **Execution-verified (Voyager):** only records with real feedback from the synced store enter; starter-kit exemplars come only from a run's best branch — code that ran end-to-end under judgment.
- **Human-verdicted:** `verdict.json` — one line per node id, written during the per-run review that already happens in `benchmarks/posttrain/reviews/` — is **required for scored runs; harvest raises (Rule 2) if a scored node lacks one**. This is the only reliable producer of PLATEAU-vs-tier and the fix for C's judged-fatal optional-verdict hole.
- **Bug-class screen (from A, mechanized):** a grep-class checklist over starter-kit files — non-weights-only checkpoint copies, cwd-relative eval paths, missing absolute-path guards — grown one entry per diagnosed bug class. A hit blocks that file's admission with a named finding. Kills the cwd×4 / 53GB×2 amplifier without admission machinery.
- **Pruning:** sliding window `seed_max_runs` keeps the last N runs per domain — deletion without curation (add-all result); the full archive stays in the per-run GCS results untouched.
- **Maturity trigger:** when harvest's free file-similarity check sees the same kit file copied-and-modified in ≥3 runs, that file has earned a one-time, human-reviewed promotion into a `toolkit/` repo whose own `tests/` are B's mechanical lints: identity lint over a per-domain ban-list data file, and the argparse fresh-task drill (model/data/template required args, no identity-named defaults, synthetic-tiny smokes). Standing repo-evolution machinery never ships (§6).

### Outcome-signed rendering

Mechanical default from store + run_state: `SUCCESS` (score > parent), `NO_GAIN`, `FAILED` (had_error/no score), `UNTRIED` (pool entries never implemented). `verdict.json` enriches with the judgment-grade enum: `PLATEAU(known_ceiling)`, `INCONCLUSIVE(confounder)`. Sign + human verdict line render as the header of **every** surface — gate emissions (experiment_history_gate.py:188–215), lessons block, duplicate alarm — with the verbatim full feedback beneath (Rule 6: k caps *selection*, never clips content). Sign completeness is enforced **at harvest push, not at load** — B's raise-at-load would let one stamping bug brick an entire wave at $65/run through the shared `_load_from_json` path (orchestrator boot + every MCP-gate store rebuild).

**Retrieval scoping.** Records carry `origin` (run id); **empty origin = current run** — the convention that survives the MCP env-transport boundary (the gate process has no run id) with zero plumbing. `get_top_experiments` filters to current-run only, so a foreign-scale 96.0 never occupies an authority render; `search_similar`/`get_recent` span origins unchanged. Gate headers render origin ("Experiment 3 [run17]"), defusing the verified node_id collision between seeded and live records.

**Lessons-block selection (pinned — the main token-rent knob):** same-task seeded records first, then recency, capped at `lessons_k`; every seeded run additionally contributes its verdict lines (one per node — cheap, high-signal). Selected records render in full.

### Anti-decision-tree enforcement — mechanism, not intention

1. **No generalization step exists.** Harvest is copy+stamp; the pass that mints rules was deleted, not regulated.
2. **Scoping-by-schema, fail-loud.** Every lessons block must carry `run_id/task/model/sign/date`; the render function raises on a missing field. Validation runs **at harvest** (bad blocks can't ship) and the render assert is a backstop, so a live run's first ideation call is not the discovery point.
3. **Exemplars, not rules.** "Greedy wins" arrives permanently framed as an exact-match observation; it cannot masquerade as an arena invariant.
4. **Provenance-named kit dirs** + README ("prior-run exemplars — adapt, don't invoke"); no merged framework exists for identity branches to accrete in. At maturity, B's identity lint becomes the promoted toolkit's own executable tests.

### Sanitation gate (the PTB judge reads the workspace)

- **Source prevention:** the feedback-generator leak-ban paragraph (feedback_generator.md:121–129) is copied into `implementation_claude_code.md`, `coding_agent_implement.md`, and `difficulties_fallback.md` — closing the verified gap where difficulties can quote eval items.
- **Hard mechanical gate:** any 12-token shingle from the task's eval set found in the merged store or kit files **aborts the push** — deterministic, testable with planted fixtures, eval data local at harvest.
- **Layered LLM sweep (from A/B, subordinated):** runs second on the surviving surface for paraphrase/memorized-answer leakage; it can only **escalate to the human keystroke**, never pass what the shingle gate would block. Vibes never decide.
- **Surface minimization:** kit takes working-tree code/md only — no git history, no PLAN.md/changes.log, no data/weights (size + path-pattern excludes); no workspace seeding means no carried branch history at all.

### Domain scoping

One config block (Rules 1/3): `knowledge: {root, domain, seed_max_runs, lessons_k}`. Harvest, seeding, signing, rendering, sweeping are domain-blind; the domain is a path segment plus two data files (ban-list terms, leak spec). Zero benchmark conditionals in `src/kapso/`; posttrain and relbench differ only in content and root (GCS vs local dir).

### PTB deployment flow + failure semantics

Boot: `run_startup.sh` (~:74, after cache mount) pulls `gs://…/knowledge/posttrain/` → `/mnt/hfcache/kapso_seed/`. `solve.sh` (:63–68): if the seed exists, JSON-validate once, untar the kit into `task_dir/starter_kit/`, append `--experiment-history-seed`; `runner.py` (:318–327) threads it; `orchestrator.py` (:327–336) copies it into `workspace/.kapso/` before store construction. Finalize: stage-1 rsync/self_destruct is unconditional (dead-VM safe); harvest runs at fetch.

**Failure semantics (explicit, with WHERE):**
- **Missing seed** (no prefix yet, or pull found nothing) = **documented default**: launch seedless AND write a loud `SEED_ABSENT` marker into the synced results dir — surfaces in `solve.sh`. Never a silent degradation.
- **Corrupt seed** = **raise**: primary gate at **harvest push** (JSON round-trip + schema + sign-completeness + checksum before upload; gsutil CRC covers the download), backstop at **orchestrator copy-in**, where malformed JSON raises and kills the run before any spend (Rule 2). The two-layer placement means the backstop should be unreachable in practice.
- **`--resume`** ignores seed flags — the populated workspace already contains its store; nothing to collide, since the git workspace is never seeded.

### Composition with ideation-v2

- **Duplicate alarm — sign-conditional, advisory-only, never suppressing** (all three designs converged on this spec; B's wording adopted): FAILED match → "similar attempt failed because ⟨cause⟩ — differentiate on that cause or justify" (*raises* interest — preserves run 17's tier-jumper); PLATEAU → "basin exhausted at X vs ceiling Y; pursue only with a mechanism escaping it"; SUCCESS → "exploit or differentiate"; UNTRIED → routes to EXPLORE.
- **Stance rule:** EXPLORE draws on carried UNTRIED pool entries (run 13's VM-lost RM-stage candidate re-enters run 16's ideation as vetted frontier capital); EXPLOIT anchors only on SUCCESS-signed records.
- **Gap slot:** best prior same-task score renders as a measured anchor ("best prior: 89.64, run 17"), turning the gap into gap-vs-known-tier — the signal whose absence let run 16 sit 36 under. Prior `eval_profile.md` ships in the kit as exemplar context; profiles are re-measured every run.

## 4. Why this fits

**Counterfactual walk.** *13→16:* stage-1 rsync preserves run 13's store past the dead VM; run 16 boots with its lessons (~1.5h saved) and the RM-stage UNTRIED candidate in its EXPLORE pool. *16→17:* the reviewer's verdict line signs 0.4815 `PLATEAU(36-under-tier)` — run 17 is never anchored into the basin, and the gap slot shows tier explicitly; run 16's FAILED DPO renders with its diagnosed cause, so run 17's on-policy candidate draws "differentiate on the cause" instead of suppression; `teacher_generate.py` is copied from the kit, not rewritten. *17→18/19:* same wave — honest per-wave granularity means 18/19 launch on pre-17 knowledge; but the teacher-choice/decoding-bake/template-parity facts each of 17/18/19 re-derived the same day were already in ≤16-era records and arrive in every lessons block. *Post-17 harvest:* the bug-class screen flags the 53GB non-weights-only copy before it enters the kit; the shingle sweep blocks any R10-P2-1-class quoted target from tainting future runs.

**Net-value verdict at 10–30 runs.** The mechanical pipeline captures the priced value — ≈+28 family-entry points episodic/semantic, most of the 1–2h/run procedural — at ~40 lines of core delta, one script, and one reviewer keystroke per node. Repo-evolution machinery is negative-EV at this scale (ETH/TroVE/token-rent + our own defect-amplifier evidence); the maturity trigger prices exactly when a human half-day starts paying. Transfer is measured, not assumed (from A): re-derivation events and time-to-first-successful-training become campaign KPIs, with an occasional no-seed control run.

**RelBench:** zero code changes — `knowledge.root` is a local durable dir on the dev box (no GCS round-trip), domain `relbench`, ban list carries `rel-amazon`-style task/dataset strings, leak spec targets test-split values and leaky columns. Its runner threads the same two paths.

## 5. Touchpoints and migration (6 atomic commits, Rule 8)

Files: `store.py`, `orchestrator.py`, `benchmark_tree_search.py`, `experiment_history_gate.py`, ideation-v2 templates, three difficulties prompts, `src/kapso/config.yaml`; benchmark-side: `benchmarks/shared/harvest.py` (new), `benchmarks/posttrain/{runner.py, gcp/run_startup.sh, gcp/20_fetch_results.sh, ptb_adapter/agents/kapso/solve.sh}`, relbench runner.

1. **[framework] Store: origin/sign/seed-load.** Fields (`origin` empty = current run), seed copy-in at orchestrator.py:327–336, origin-scoped `get_top`, embedding-role pinned in config (dimension mismatch already raises, store.py:36–39). *Tests:* fixture seed loads; corrupt JSON raises; seeded records excluded from top, included in similar/recent; mismatch raises.
2. **[framework] Leak-ban prompts.** Paragraph into the three difficulties prompts. *Tests:* contract pins. Campaign-urgent; independent of everything else.
3. **[framework] Renders.** Lessons block in problem-context assembly (`benchmark_tree_search.py`), sign+origin headers in gate renders, sign-aware alarm + UNTRIED surfacing + prior-tier anchor in ideation-v2 templates. *Tests:* schema validation raises on a missing binding; FAILED-match renders differentiate-on-cause; difficulties text verbatim (Rule 6); "Experiment 3 [run17]" header.
4. **[benchmark] Harvest.** Merge+window+mechanical signs (store + run_state.json)+required-verdict raise+shingle gate+LLM-escalation sweep+bug-class grep+kit build+pre-push validation/checksum. *Tests:* planted shingle aborts; scored node without verdict raises; window drops oldest; planted 53GB-pattern flagged; data/history excluded; idempotent re-harvest.
5. **[benchmark] PTB plumbing + config.** `knowledge:` block; run_startup pull; solve.sh validate/flags/`SEED_ABSENT` marker; runner threading; resume ignores seeds. *Tests:* config round-trip; threading unit test; absent-seed marker written; shell dry-run.
6. **[benchmark] RelBench wiring.** Same keys, local root, domain data files. *Tests:* runner threads paths; relbench leak fixture aborts harvest.

Not booked: shrinking `PRIOR_RUN_INSIGHTS` — verified already paper-facts-only.

## 6. Explicitly rejected

- **Workspace seeding via `initial_repo` (v1)** — activates an unexercised path in a live campaign, requires the clobber fix and resume precedence surgery, and clones git history (contamination hole no sweep covers). Revisit at the maturity trigger; first exercise on relbench/dev-box, never a live PTB wave, from a history-free snapshot.
- **Consolidation-as-kapso.evolve / LLM curation** — lints check shape, not truth; unsupervised consolidation re-admits mis-signing and self-verified admission (cwd×4, 53GB×2); re-adds what ef0127c5 deleted.
- **TroVE tiering, reuse counters, embedding dedup, ≥2-run provenance lints** — amortization needs volume 10–30 runs never produce; dead weight (Rule 10).
- **Separate LESSONS.md/KNOWLEDGE.md** — indicted side-doc pattern (`arena-best-baseline-traces.md` held the ⅓-non-English fact; all three runs drew P1s anyway); one channel, the store.
- **Unsigned-record-raises-at-load** — one stamping bug bricks a wave via the shared load path; enforce at harvest push.
- **Auto-suppressing duplicate alarm** — kills tier-jumping candidates; advisory + sign-conditional only.
- **LLM-only contamination sweep** — a nondeterministic gate is what the judge threat model forbids; LLM layer escalates only.
- **Store committed into any seeded repo** — tracked-file modification breaks `switch_branch` (verified).
- **Live intra-wave sync** — 17/18/19 launched simultaneously; per-wave granularity is honest.
- **Carrying eval profiles as facts** — fastest-staling knowledge; re-measure every run, archive as exemplars.