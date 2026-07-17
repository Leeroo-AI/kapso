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
