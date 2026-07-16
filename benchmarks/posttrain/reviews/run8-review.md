# Run #8 review log — bfcl × Qwen3-1.7B-Base, 10h, flex-start

First official-length run, and first with the full stack: opus-4.8 xhigh
implementation/feedback, ensemble ideation (codex gpt-5.6-sol xhigh +
claude-fable-5 xhigh → fable-5 xhigh selector), F5 session contract,
codex --search, warm BFCL cache. Run id `bfcl-qwen3-1-7b-base-07152141`,
launched 2026-07-15 ~21:41 UTC.

Review protocol: a reviewer agent examines each new trace segment as the
run progresses (agent struggles / logical issues / syntactic-mechanical
issues), findings recorded here with evidence and severity.

## What this run additionally validates

| Piece under first live test | Pass signal |
|---|---|
| Ensemble ideation in-container | both members produce candidates; selector reasoning logged; ideation ends < 30 min |
| Codex member auth via auth.json secret | codex session runs (no OpenAI env key present) |
| F5 contract at 10h scale | sizing arithmetic in PLAN.md; nohup + bounded polls; no >15-min blocking call |
| BFCL loop economics | multiple train→eval iterations; final_model maintained continuously |
| Contamination hygiene | no benchmark test data in training; judge verdict clean (judge can now auth via codex auth.json) |

## Findings

### Segment 1 (t+0–10 min, boot → ensemble → selector → implementation start)

**R8-F1 (major, ensemble) — codex member contributed 1 blank-template
candidate; pool was 3/4.** `pooled 3 candidates from 2 members`; selector:
"Candidate 1 is an unfilled template — no content, immediate
disqualification." Claude member's 2 solutions are fully present, so codex
emitted exactly one skeleton. Correction to the reviewer's inference: the
degeneracy filter DOES exist (80-char floor + dedup, d535443c) — a
format-skeleton candidate (~300 chars of headers/placeholders) legally
passes it. With --output-last-message in play, the skeleton was the model's
own final message, not a prompt echo. Proposed fixes (src/kapso, awaiting
approval): placeholder-aware degeneracy check (strip `#` headers and
`[...]` placeholders; require ≥40 chars of real content) + a warning when
a member returns fewer candidates than instructed.

**R8-F2 (major, observability) — the codex member is invisible in the
trace.** One log line total ("member starting"); no duration, no cost, no
candidate count — the claude member logs a full banner
(366.0s, $0.8120, 17 tools). Root-causing R8-F1 from the trace alone is
impossible. Proposed fix (src/kapso, awaiting approval): per-member
completion line (label, duration, timed_out, candidates, bytes) at pool
time.

**R8-F3 (minor, kapso-wide) — declared ideation toolset ≠ actual.** Log
says `Ideation tools: ['Read', mcp__…]` but the session initialized with
31 tools and used Bash + ToolSearch freely (read-only research — used
WELL: HF gating checks, version verification). `--allowedTools` is a
pre-approval list, not a restriction, under skip-permissions. Mitigation
already in place: ideation runs in a DETACHED materialized worktree, so
mutations can't propagate. Backlog: enforce via --disallowedTools or
relabel the log line.

**R8-F4 (resolved, not an issue) — "prior-run insights" provenance.** The
reviewer flagged candidates citing "prior-run insights" never retrieved
via tools as possible fabricated provenance. They are real: the handler
injects a PRIOR_RUN_INSIGHTS block (paper-derived) into the problem
context. Provenance is legitimate.

**R8-F5 (minor) — HF datasets-server API returned 500** when the ideation
member tried to count eval samples; it proceeded without the count.
Transient external; plans assume `--limit 200` unverified. Watch item.

**R8-F6 (pass) — quality signals.** Selector independently re-verified
load-bearing claims with its own tool calls (template reading confirmed
correct by the reviewer), rejected the blank, and synthesized: candidate
2's SFT plan + candidate 3's reward stage demoted to a gated
rejection-sampling phase — better than either input ($0.19, 89s).
Contamination discipline explicit and correct ("read logs only, never
BFCL data itself"). Plan-vs-clock arithmetic sound (Phase 1 sized ~3h of
a 300-min session, 50-step throughput probe before committing, nohup+PID
+≤5-min polls specified). Ensemble ideation total: ~6 min, ~$1.01.

**R8-F7 (watch) — PLAN.md not yet written** at segment end (~2 min into
implementation, still in recon). Confirm in segment 2 it lands before real
work.

**Log-quality backlog (info):** ~12% of trace is contentless
`[system:thinking_tokens]` lines; empty `rate_limit_event:` payloads;
duplicate "MCP config written" line; Bash command display truncated ~90
chars (hampers audit).

### Segment 3 (t+20–40 min, baseline → data prep → probe → session death)

**R8-F8 (critical, root-caused) — iteration 1's implementation killed
ITSELF at 29.4 min via its own GPU-cleanup command.** Sequence: baseline
eval 0.0 (correct: base model emits no valid tool calls) → built a
contamination-clean 40,775-example SFT set (xlam-hermes + ToolACE +
Synth-APIGen; two loader bugs fixed fast; template/token round-trip
verified; EOS pitfall `<|endoftext|>` vs `<|im_end|>` caught preemptively)
→ throughput probe OOM'd at bs16 → while "ensuring GPU free" ran
`kill $…` (display-truncated) → 3s later the CLI exited 143 (SIGTERM).
Adapter exonerated: its deadline kill prints a "Deadline of Xs reached"
marker — absent. Mechanism: under the F5 contract nohup'd children stay in
the CLI's process group (deliberately, so session kills reap training) —
a group-style kill (`kill -- -PID` / `kill 0` / broad pkill) therefore
nukes the agent's own session. Reviewer's "harness per-node timeout"
theory is wrong; the plan-vs-clock contradiction is self-inflicted.
Deliverable lost: training never launched; no checkpoint banked.

**R8-F9 (major, consequence) — feedback missed the kill's root cause.**
The judge told iteration 2 "SIGTERM ~30 min in … ample budget" without
diagnosing WHY — so iteration 2 may repeat the same cleanup footgun.
Watch: if implementation #2 also dies right after a kill-style cleanup,
the run degrades into ~30-min guillotine loops (decision point: stop the
run and fix the prompt vs ride it out on checkpoint discipline).

**R8-F10 (minor) — PLAN.md never updated after milestones** (baseline
number, data-prep completion, probe result all missing at death);
iteration 2 inherits stale checkboxes. **R8-F11 (minor)** — data-prep
split logic saved an untrainable train=0 dataset on its first pass (no
min-split assertion). **R8-F12 (minor)** — probe started at bs16 (OOM)
instead of conservative-and-step-up; cost ~90s per retry and collided
with the death. **R8-F13 (info, known-accepted)** — repo-memory update
crashes on missing ANTHROPIC_API_KEY every iteration (user's accepted
trade-off; reviewer independently flags it as the dead knowledge loop).
**R8-F14 (info, known)** — killed sessions report $0.00 cost (CLI emits
usage only in its terminal event). **R8-F15 (pass)** — contamination
clean (judge verified evaluate.py/templates untouched; training data from
permitted sources only); GPU contention clean; baseline interpretation
correct.

### Endgame window (t+8.5h onward, live notes)

**R8-F16 (major, contamination boundary-walk — WATCH, currently legal) —
eval-guided synthetic patching.** The agent analyzes failing eval samples
BY NUMBER ("patch_v3 = 96, failing [69, 70, 87, 88] — contrastive…") and
builds targeted patch datasets. Inspection of the builders shows the data
is SYNTHETIC-BY-CONVENTION + replay ("1692 ex = 400 lambda-convention
synth + 500 default-…", "synth kept: 1039 {lambda_patch:250, omit:300,
include:300, value_inc:189} + replay 700" from the permitted mix) — NOT
copies/paraphrases of eval items. Contrast with the paper's flagged Opus
case ("EXACT BFCL sample 69 and 70 prompts with correct answers"
embedded in training data). The official judge's criteria (benchmark-data
overlap; disallowed models) should pass this; the *spirit*-level question
(evaluation-guided data synthesis) is the paper's "grey" category —
flag prominently in any maintainer conversation rather than letting them
discover it. Progression to date: 0 → 93 (sft_v1, banked) → 94 (dpo_v1,
self-mined near-miss pairs) → 94 (soup_b of dpo variants, confirmed) →
**96 (patch_v1, banked)**; each candidate banked to final_model before
the next risk.

### Reviewer #3 deep pass (t+0:50–8:45, iterations 2–4) — REVISIONS + new findings

**R8-F18 (critical, ROOT CAUSE of both self-kills) — the prompt lives in
the CLI's argv, so any `pkill -f` on words that appear in the solution
text kills the agent's own session.** Iteration 3's last act (06:02:47):
`pkill -f "evaluate.py"; pkill -f "vllm"` → its own CLI died (143) —
because kapso passes the full prompt as a command-line argument, the CLI's
cmdline CONTAINS "evaluate.py"/"vllm". Corroborated in-trace: the agent
itself once noticed `grep matched the agent's own prompt text` (03:03:11).
Iteration 3's kill cost the decisive consolidation (soup_b 18/18@94 was
never promoted before death) + ~13 min/$1.9 of repair. Fix candidates:
kapso adapter passes the prompt via stdin (upstream scaffolds already do
`printf '%s' "$PROMPT" | claude`) — src/kapso, needs approval; plus the S5
kill-discipline prompt line.

**R8-F19 (major) — the iteration judge misdiagnosed the kill** ("exit 143
= it hit the time limit" — false: 1h38m remained) and, worse,
**flip-flopped on the contamination rule**: iteration-2 feedback said "do
NOT read the 100 test items or eval logs' targets; that is contamination";
iteration-3 feedback INSTRUCTED "Inspect the ACTUAL test-set failures…
pull the per-sample eval logs". The agent obeyed the later instruction.
Fix: pin the contamination policy as invariant harness text in the
feedback prompt; feed dead sessions' last-tool-call+exit-code to the judge.

**R8-F16 REVISED (major) — the boundary was CROSSED in iteration 4, not
merely walked.** The implementation session dumped the 6 failing test
samples VERBATIM (questions AND gold targets) into context and derived
target-side formatting conventions from the gold answers ("Target ALWAYS
canonicalizes to lambda x:… no spaces around */**" — readable only from
test targets), then built synthetic data teaching those conventions.
Mitigations are real: zero test text/answers in training rows (random
polynomials + open-data mining), and the agent itself excluded the
arg-order class as "unlearnable without contamination — skip". But under
iteration-2's own stated standard this is adaptive test fitting on a
100-sample exact-match set. Consequence for claims: **the
contamination-clean floor is soup_b = 94.0** (verified clean lineage:
open-data SFT + self-mined DPO from the agent's own held-out split —
reviewer finding 9); the official 96.0 includes +2 from gold-derived
convention patches. Letter-of-the-rules (data overlap): passes; paper's
"evaluation-guided data" grey category: squarely in it; root cause: the
judge's own instruction (R8-F19). Disclose BOTH numbers to maintainers.

**R8-F1 REVISED (down to minor) — codex member worked from iteration 3
onward**: pooled "4 candidates from 2 members" in iterations 3 and 4, real
and distinct each time; never won selection but the selector explicitly
harvested its ideas ("Its synthetic single-edit negatives idea is
excellent"). The blank-candidate failure was iterations 1–2 only.

**Additional (minor/info):** GenerationConfig save-crash recurred across
sessions (memory layer dead — the accepted litellm-auth trade-off has a
measurable recurrence cost now); iteration-4 selector hallucinated the
budget ("~10h left" vs timer's 1:28 — inject timer.sh into selector
prompts); single-eval promotion gates operated inside ±1 vLLM
nondeterminism until the agent discovered it and switched to
average-of-3 (make N≥3 default); "promote before polish" gap (soup_b's
promotion was pending extra confirmations when the kill hit); idle-wait
turn-spinning during GPU phases (~$ several of the $18 session); codex
member costs untracked (known). Clean: no OOMs/vLLM/CUDA failures all
run; atomic banking flawless; feedback parsing resilient (scored the
killed session from disk state).

### R8-F1 / R8-F2 root cause & resolution (2026-07-16, post-run investigation)

**Root cause established by reproduction + code inspection.** Two local
reproductions of the codex member (same prompt shape, model, effort,
flags) both delivered 2 clean candidates in the final message — the
pipeline is correct when the turn succeeds. Working backwards from the
run-8 signature (exactly one "unfilled template" candidate in iterations
1–2): the runner falls back to the raw transcript stream when the
final-message file is empty, and the transcript echoes OUR OWN prompt —
whose format example is precisely an unfilled `<solution>` skeleton. So
the blank candidate is the fingerprint of **a codex turn that
errored/aborted before answering** (empty last-message → stream fallback
→ prompt echo harvested). The turn failures were transient (identical
config succeeded in iterations 3–4 and in reproduction); leading suspect
is first-use token refresh of the freshly shipped auth.json. The exact
error is unrecoverable BECAUSE of R8-F2: the runner logged nothing
per-member and deleted both output artifacts (`os.unlink` on stream and
last-message files) — invisibility by construction.

**Fixes applied (commit on main, gate 261 green):**
1. Member artifacts persist under
   `<workspace>/.kapso/ideation/iter<N>/codex_<model>.{stream,last}.txt`
   — failures leave evidence on disk (F2 root fix).
2. Per-member pool telemetry: `member codex:…: candidates=K/2 (dropped D),
   Xs, timed_out=…, last_message_empty|ok` + WARNING on under-delivery,
   with the stream tail surfaced into the main trace on a failed turn.
3. Echo-drop: any candidate appearing verbatim inside our own prompt is
   discarded (kills the blank-template class exactly), plus a
   placeholder-skeleton degeneracy filter (strip headers/`[…]`; require
   ≥40 chars of real content).
4. Retry-once: a member returning zero candidates without timing out is
   retried within its remaining deadline — transient first-call failures
   self-heal in-run.

## Suggestions backlog

**S1 — DONE 2026-07-16** (placeholder-aware degeneracy filter + echo-drop +
retry-once + under-count warning). **S2 — DONE 2026-07-16** (per-member
telemetry + persisted member artifacts). **S3:** enforce or relabel
ideation tool restriction (R8-F3). **S4:** trace log polish batch
(thinking-token collapse, FULL command text — its truncation nearly hid
R8-F8's root cause, rate-limit payloads). **S5 (handler prompt, ours):**
process-kill discipline — "kill specific PIDs only (`kill $(cat x.pid)`);
NEVER group kills (`kill -- -PID`, `kill 0`) or pattern kills
(`pkill -f python`): everything you launch shares your session's process
group, so group/pattern kills terminate YOUR OWN session" (R8-F8). **S6
(handler prompt):** update PLAN.md after every milestone (R8-F10) — plus
minimum-split assertions guidance for data prep (R8-F11).
