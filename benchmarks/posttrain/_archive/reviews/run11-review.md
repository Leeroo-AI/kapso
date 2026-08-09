# Run #11 review log — bfcl × Qwen3-4B-Base, 10h (2026-07-17)

`bfcl-qwen3-4b-base-07170824`, built from `46e6390f` (live luna memory,
judge leak ban, 1800s admission floor). Cell: base 1.0 measured, human
95.0, proven top gpt-5.4-h-rp/fable-5 100.0.

## Pass 1 (08:28Z–10:30Z, t+2h05)

Verdict: **1 FRAMEWORK MAJOR, 0 agent majors, 3 minor, 4 info.**

**R11-P1-1 (FRAMEWORK MAJOR) — run #9's special-token lesson reached
this run through NO channel; ~70 min re-derived (12% of budget).**
Ideation planned the stopping half (eos must include <|im_end|>) but the
selected plan used all-linear LoRA with frozen embeddings — run #9's
exact trap. Merged model emitted garbage control tokens; agent
diagnosed (untrained rows), pivoted to full FT (clean fix given
tie_word_embeddings). Why framework-level: the lesson exists only in
reviews/run9-review.md which agents cannot see; handler static insights
don't cover it (hint was reverted by user direction 2026-07-16); the
live memory layer worked mechanically but is per-campaign — fresh run =
empty store; get_insights never invoked by members. Recurrence evidence:
run #10 planned around the trap, run #11 did not — 1 of 2. Decision
pending (user): handler hint / cross-campaign insight seeding / accept.

**Minors:** R11-P1-2 — no final_model for first 2h09 (LoRA candidate
died unscored; base-model insurance promotion was available). R11-P1-3 —
killing the recorded eval PID leaves the vLLM server tree alive (~2 min
orphan dance, recurs across runs; contract could record process groups
for server-spawning evals). R11-P1-4 — claude member telemetry line
still lacks duration (known cosmetic, now filed).

**Infos:** selector rejected both codex candidates with evidence (they
trained categories the eval never scores); 4 rate-limit events, zero
stall correlation; no OOMs (full-FT 62/80GB and FASTER than LoRA:
2.48 vs 3.39 s/step); data = argilla/Synth-APIGen-v0.1 (49k → 14.5k
single-call curated), provenance clean; env_strip clean; contamination
clean (diagnosis on own dev split, "no per-sample reading" in plan).

State at review end: full-FT recovery training, first score ~11:00Z.
