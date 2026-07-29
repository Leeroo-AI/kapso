# Kaggle benchmark — run findings

Issues found running the URL→score loop on IOAI practice Task 1. Each entry:
what broke, the evidence, the fix (or the open proposal). Newest run first.

---

## Run 3 — 2026-07-29 15:29 UTC, spot 8×L4, k=8, 2h, fable-5, 5 candidates/member

### I-1 (FIXED, verified under failure) — `k=8` silently collapsed to one lane
**Symptom (run 2):** a k=8 run executed a single implementation lane.
**Evidence (run 2):**
```
member codex:gpt-5.6-sol       candidates=0/2, 662s, timed_out=True, last_message_empty
member claude_code:claude-opus-5  candidates=1/2, ok
Ensemble ideation pooled 1 candidates from 2 members
Single candidate — selector skipped
```
**Root cause:** K-way expansion needs the selector to emit K ranked solutions,
which needs a pooled candidate count ≥ K. `ENSEMBLE_CANDIDATES_PER_MEMBER` was a
hard-coded `2`, so one timed-out member left a pool of 1 → selector skipped →
single lane. No error was raised; the run looked normal.
**Fix:** the count is now config-driven —
`search_strategy.params.ideation_candidates_per_member` (commit `63eda5d7`),
set to `5` for this benchmark (commit `10f1b4dd`). Pool is now up to 10 ≥ K=8.
**Verified in run 3, under the same failure:** fable-5 under-delivered (1 of 5)
*and* the selector timed out, yet codex delivered 5/5 → pool 6 → **6 lanes
expanded** instead of collapsing to 1.

### I-2 (OPEN) — ideation time split starves members and the selector
**Symptom:** at a 2h budget both the claude member and the selector hit their caps.
**Evidence:**
```
Ensemble member claude_code:claude-fable-5 failed: Claude Code CLI exceeded its 756.0s
member claude_code:claude-fable-5 under-delivered: 1 of 5 candidates
Selector failed (selector session timed out); falling back to the pooled candidates
member codex:gpt-5.6-sol: candidates=5/5 (dropped 0), 628s, timed_out=False, ok
```
**Arithmetic:** 2h → `ideation_timeout` = min(1500, max(300, 7200×0.15)) = 1080s,
split `ENSEMBLE_MEMBER_TIME_FRACTION=0.7` → members 756s, selector 324s. Codex
finished in 628s; the claude member needed more than 756s; the selector needed
more than 324s.
**Consequences:** fewer candidates than configured, and — when the selector dies —
no ranking/synthesis at all, just the raw pool (complementarity across lanes is
then unmanaged).
**Proposed fix (core, needs approval):** raise `ideation_fraction` for long runs
and/or rebalance `ENSEMBLE_MEMBER_TIME_FRACTION`; both are module constants in
`generic/strategy.py` and, being tunables, belong in config per Rule 1 — the same
treatment `ideation_candidates_per_member` just received.

### I-3 (OPEN) — pool size caps expansion below configured K
`node_expansion_value=8` with 8 lane pins configured, but the pool was 6 → 6
lanes. Expansion can never exceed the pool, so K is an upper bound, not a
guarantee. Either accept this, or have the strategy report the shortfall as a
first-class signal (it is currently only inferable from the lane count).

---

## Run 2 — 2026-07-29 11:26 UTC, spot 8×L4, k=8, 1.75h, opus-5

### I-4 (FIXED) — `setup_box.sh` cloned `main`, not the worktree branch
**Symptom:** box had no `benchmarks/kaggle/` work; `preflight.py` absent.
**Root cause:** a plain `git clone` lands on the default branch, and the
follow-up `git -c "http.extraheader=AUTHORIZATION: basic …" fetch` does **not**
authenticate against this repo — git falls through to an interactive prompt and
dies `fatal: could not read Username`. The clone had already "succeeded" on main,
so the failure looked like a fetch hiccup.
**Fix:** `git clone --branch "$KAPSO_BRANCH"` (one authenticated op, PAT in URL,
stripped right after) plus an assertion on the resulting branch; same treatment
in `bootstrap.sh` (commit `493c77c0`).

### I-5 (FIXED) — `kapso/.env` missing → auth failure mid-campaign
**Symptom:** `ValueError: Claude Code OAuth credentials not found` during
ideation, after the box was provisioned and the preflight had run.
**Root cause:** `bootstrap.sh` writes `~/kapso/.env`; `setup_box.sh` does not.
Running setup alone leaves the runner with no token — and the failure only
surfaces once the campaign reaches its first Claude session.
**Fix:** `run_competition.sh` now checks for `CLAUDE_CODE_OAUTH_TOKEN` in
`~/kapso/.env` and fails before spending compute (commit, run-3 prep).

### I-6 (BY DESIGN, documented) — stale run checkpoint blocks relaunch
`RunCheckpointIncompatibleError: This workspace already contains a run
checkpoint` after a crashed launch. Correct fail-loud behaviour. Recovery: delete
`<root>/task/kapso_campaign` (and `run_meta.json`, `.kapso_runtime`) for a clean
restart, or pass `--resume` to continue the existing one.

### I-7 (OPEN) — end-to-end submission budgeting
**Symptom:** run 2 pushed its first kernels at 12:50/12:53 against a 12:59 end —
no margin to submit and score. Result: **0 submissions** despite ~93 min of work.
**Root cause:** the handler's END-TO-END paragraph is advice, not a mechanism;
nothing forces an early first submission.
**Proposed fix:** a hard "first scored submission by T+50% of budget" rule in the
handler prompt, or a strategy-level reserve that triggers a submit attempt.

---

## Run 1 — 2026-07-29 ~09:50 UTC, spot 8×L4

### I-8 (FIXED) — `setup_box.sh` assumed an image that does not exist
Written against a `/opt/conda` DLVM. The real
`pytorch-2-9-cu129-ubuntu-2204-nvidia-580` image has:
- **no** `/opt/conda`; system `/usr/bin/python3` (3.10) **already ships torch
  2.9.1+cu129 seeing all 8 GPUs** → inherit it, `pip install --user`, never
  rebuild the torch triple (the slow, ABI-fragile path);
- `python3 -m venv` fails (no `ensurepip`) and is unnecessary for the same reason;
- `pip install -e .` fails (old pip lacks the PEP 660 `build_editable` hook) →
  use `PYTHONPATH=~/kapso/src:~/kapso`;
- kaggle CLI 2.x needs py≥3.11 — py3.10 resolves the pre-KGAT `1.7` which crashes
  at import → `uv tool install kaggle --python 3.11` (uv defaults to 3.10 unless
  forced);
- **both** agent CLIs are required (codex *and* claude — the ideation ensemble and
  lens planner are `claude_code`), not just codex;
- `mcp` is an undeclared runtime import of kapso;
- bare `python` does not exist — use `python3`.
**Fix:** `setup_box.sh` rewritten against these facts, self-verifying at each step
(commit `3f5d4149`).

### I-9 (FIXED) — runner's `--coding-agent` default clobbered the config
`--coding-agent` defaulted to `"claude_code"` (a claude-primary-era default).
`orchestrator._create_search_strategy` treats an explicit value as an override, so
that default silently replaced the codex `coding_agent` block (codex/gpt-5.6-sol/
xhigh) with a bare default-model claude agent. Default is now `None` so the config
wins (commit `3ddb282f`).

---

## Environment notes (not bugs, but they shape every run)

- **8×L4 capacity is scarce.** `g2-standard-96` on-demand was `ZONE_RESOURCE_POOL_
  EXHAUSTED` in **all 17 zones** tried across US/EU/Asia on 2026-07-29; runs 2 and
  3 both landed on **spot** in `europe-west4-a`. Neither was preempted, but a
  ~2h campaign on spot is a real risk.
- **Per-model rate limits are separate from the unified 5h/7d buckets.** A token
  can show `5h=0% / 7d=55% allowed` and still return **429 for a specific model**
  (`claude-fable-5`), which killed a run in the lens planner. Probe the *actual
  model* before launching, not just the unified headers.
- **Never ship the Bedrock trio** (`AWS_BEARER_TOKEN_BEDROCK`,
  `CLAUDE_CODE_USE_BEDROCK`, `ANTHROPIC_MODEL`) to a box: `auth_mode: auto`
  resolves Bedrock *first* and would silently hijack which model runs. The dev
  box's `.env` contains them; the box `.env` is curated to
  `OPENAI_API_KEY` + `CLAUDE_CODE_OAUTH_TOKEN` + `HF_TOKEN`.
