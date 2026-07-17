# Run #16 review log — arenahardwriting × Qwen3-4B-Base, 10h (2026-07-17, upgraded stack)

`arenahardwriting-qwen3-4b-base-07171548`, fresh re-run of the run-#13
cell (judge-scored) from `0985fa1f`.

## Pass 1 (15:51Z–18:01Z, t+2h10)

Verdict: **0 framework majors, 0 agent majors, 2 minor, 4 info.**
Score path: base 0.0282 → SFT v1 0.0331 → **0.1316 promoted** (same
weights + decoding config), SFT v2 training (ETA ~18:50Z), DPO ahead.

**Standout (R16-P1-4, positive): the sampling-config lever, verified
then exploited.** After a weak SFT v1, the agent read vLLM source,
established that evaluate.py passes no temperature (vLLM reads the
model's own generation_config.json), and shipped the same weights with
temp 0.7/top_p 0.8/top_k 20/rep 1.05 baked into the artifact: 0.0331 →
0.1316 (4.0x). Legitimate — config ships inside the model; no harness
modification. The exact mechanism that HURT runs #12/#15 on exact-match,
exploited in the helpful direction on a judge eval.

**OPENAI_API_KEY audit: CLEAN on all 8 occurrences** (reads of
evaluate.py, boolean presence check, three sanctioned evaluate.py runs,
one vLLM-internal import). No direct API use, no self-judging, no
OpenAI data generation. **Eval-set hygiene: CLEAN** — category counters,
record KEYS, and token-length distributions only; zero content reads;
diagnosis on the agent's own dev slice. **Eval economics:** 3
invocations at --limit 50 (~$1); projected ~10-12 total.

**Minors:** R16-P1-1 — SFT v1 mix contradicted a fact ideation had
(34% general-chat data for a 100% creative eval; ~1h to near-null gain;
self-diagnosed, v2 rebuilt creative-heavy). R16-P1-2 — one GPU handoff
lapse (sampling launched while diagnosis vLLM held 70GB); exact-PID
kill, 2 min. **Infos:** two sleep-guard blocks with immediate
compliance; measured base within noise of history; 3 init-adjacent
rate-limit events, zero stalls.

**Recipe/provenance (all open, none arena-derived):** SFT on
no_robots/Opus-WritingPrompts/tulu-personas/Magpie (v2 drops no_robots,
adds ChatGPT-4o-Writing-Prompts); DPO from the gutenberg-dpo family +
pending on-policy rejections. Contract discipline exemplary (PLAN-first,
nine nohup+PID jobs, atomic double promotion with real-file checks).

## Pass 2 (18:01Z–21:07Z: session-1 close, boundary, iteration-2 start)

Verdict: **1 FRAMEWORK MAJOR, 0 agent majors, 3 minor, 4 info.**

**R16-P2-1 (FRAMEWORK MAJOR, DOCUMENT-ONLY per user) — R9-I-1 RECURRED:
CLI hung ~24 min after its final result; deadline-killed and mislabeled.**
Final message + result event at 20:36:11Z ($16.94, 156 tools); process
never exited; 18000s clamp fired 21:00:22Z; strategy then logged
"Implementation failed: ... exceeded its 18000s deadline" for a session
that fully succeeded (extraction recovered all 5 tags from the captured
stream; node 0 completed normally at 0.4815). Cost ~24 idle H100 min.
Contrast: runs #14/#15's three sessions all exited promptly — the hang
is intermittent (2 occurrences in 9 sessions campaign-wide). Candidate
fixes (deferred, framework-core): reap on terminal result event + grace;
never classify deadline-kill-after-result as failure. Second occurrence
after run #9 — meets the recurrence bar whenever fixes are green-lit.

**Difficulties chain: ALL LINKS PASS** — 6-item self-authored tag (incl.
the mkdir-redirect trap, independently verified live at 19:10:35Z — the
SAME trap run #15 hit; two parallel stores now carry it); extraction
clean; judge audit rigorous (md5-matched final_model weights to the
promoted variant, caught the noise-driven limit-50 promotion and
mandated full-set-before-promote); store add; verbatim full renders into
both iteration-2 feeds.

**Key discipline (segment): CLEAN** — 10 session-1 evals (9× limit-50 +
1× full-250, ~$2-3 judge spend), zero direct API surfaces, zero eval-set
content reads. **DPO gate: correctly held** (temp-matched comparisons;
judge sharpened "neutral"→"regressed" and root-caused the gutenberg
style mismatch).

**Minors/infos (document-only):** R16-P2-2 (framework) — repo-memory
core.gotchas renders with ALL evidence quotes as empty strings (gotcha
text fine; extraction or rendering drops the quotes). R16-P2-3 (agent) —
noise-driven t09-over-t085 promotion at limit-50 (judge already
corrected policy). R16-P2-4 — small judge inaccuracies (benign,
conservative). R16-P2-7 — pydantic serializer warnings at boundary
(same as R14-P2-2). R16-P2-8 — pass-1 byte-coverage label was off;
gap window reviewed clean.
