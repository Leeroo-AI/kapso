# Run #14 review log — bfcl × Qwen3-4B-Base, 10h (2026-07-17, upgraded stack)

`bfcl-qwen3-4b-base-07171548`, fresh re-run of the run-#11 cell from
`0985fa1f` (first-live: technical_difficulties contract, per-session
stream forensics, no insight extractor, full-content memory renders).

## Pass 1 (15:51Z–17:59Z, t+2h08)

Verdict: **0 framework majors, 0 agent majors, 2 minor, 6 info.**
Score: baseline 0.00 → SFT 0.71 (t+57m) → **RFT 0.86 promoted (t+80m)**;
combined-SFT main bet (25.4k verifier-clean rows) lands ~19:03Z.
Strongest first two hours on this cell; ~10-11 min total waste of 128.

**Special-token trap: planned around upfront, 0 min lost** (the #10/#12
pattern, not #9/#11): full FT chosen at ideation, token-id audit + TRL
eos inspection + finalize_model.py EOS patcher all BEFORE training.
Post-revert scorecard on the trap: avoided 3 of 5 runs (10, 12, 14),
re-derived twice (9: 45 min, 11: 70 min).

**Minors:** R14-P1-1 — vLLM 0.11 removed `prompt_token_ids=`; ~3.2 min +
one in-namespace orphan, textbook PID-only cleanup. R14-P1-2 — DPO round
OOM'd while not learning; discarded safely behind the promote gate,
root-caused (`precompute_ref_log_probs=True`) for a later retry. Both are
ideal first material for the difficulties tag.

**Infos:** sleep-guard fired once, steered to bounded until-loops
(first live sighting on this cell — working as designed); ideation
visibly consumed the handler's static PRIOR_RUN_INSIGHTS (template
lever) — grounded, closes run-11's pending item on that channel; small
tool blips absorbed (Write param retry, 4 short-file reads, git-root
slip); training data via public re-upload of the gated xLAM set
(`NobodyExistsOnTheInternet/...`) — legal, convention question recorded;
eval-dataset touch was count/columns only; trace ends mid-poll.

**Feature table:** contamination clean (all 18 write targets enumerated,
eval files untouched); ensemble 2/2+2/2 no 529s, selector re-read the
eval code before judging; env_strip clean; difficulties contract
confirmed shipped prompt-side (tag due at session end ~20:51Z);
stream-forensics weak positive (.kapso/ present in workspace listing);
full-content renders N/A until iteration 2. Contract discipline: pass on
all five (sizing 43% < 60%; ten nohup+PID jobs; atomic promotes with
VERIFY_OK; unprompted final_model integrity re-check).

## Pass 2 (17:39Z–21:06Z: session-1 close, boundary, iteration 2)

Verdict: **0 majors, 1 minor, 3 info.** Session 1 self-exited CLEAN 25
min early (15952s, $18.58, 158 tools) — no R9-I-1 linger here.

**Difficulties chain: ALL LINKS PASS (first live validation).**
Self-authored 7-item tag (DPO OOM+fix, save_total_limit loss, eval
variance, EOS, vLLM API change, orphan EngineCore, buffered logs);
extraction with all 5 tags, no fallback; judge honest (reported the
0.91 mean not the 0.94 max, "never promote on a single-run delta given
±4 noise"), invariants verbatim; store add; THREE full uncut renders;
iteration-2 ideation root-caused the sampling noise to
vllm/config/model.py:1344 and the selector re-verified claims against
installed source before judging. Downstream proof: the new implementor
CITED difficulty #5's save_total_limit gotcha and relaunched its retrain
because of it — the artifact changes behavior, not just decorates the
store.

**Iteration-2 result: 0.95 promoted 20:51Z** (`greedy_config_exploit`):
temperature 0.0 shipped in the artifact, determinism proven (two limit-20
runs at 1.0, two identical full evals at 0.95), judge pre-sanctioned as a
legitimate model-artifact setting. R14-P2-4 (info): campaign record
should attribute the +4 to decoding config; the 25.7k retrain (ETA
22:25Z) is the weights-side follow-up.

**R14-P2-1 (minor, framework, DOCUMENT-ONLY):** stream-forensics
artifact unverifiable — never exercised (self-authoring worked) and
`.kapso/sessions/**/stream.jsonl` is absent from the results sync; if a
session dies we learn only then whether the fallback's evidence exists.
Candidate fix (deferred): include it in the sync. **R14-P2-2 (info):**
pydantic serializer UserWarnings on each boundary luna call
(litellm-version skew noise; responses parse fine). **R14-P2-3 (info):**
Experiment-0-vs-Iteration-1 numbering off-by-one across surfaces
(display only).

## Run outcome (official)

**95.0 official, contamination-clean** ("no contamination detected",
"only allowed use detected") — ties human post-training exactly on the
cell. The deterministic greedy artifact config meant the official
harness run reproduced the internal 0.95 precisely — the first
zero-variance official score of the campaign. Path: SFT 0.71 → RFT 0.86
→ 0.91 (mean-of-4 under sampling) → greedy config 0.95 deterministic;
the 25.7k retrain did not clearly beat it and was correctly not
promoted.
