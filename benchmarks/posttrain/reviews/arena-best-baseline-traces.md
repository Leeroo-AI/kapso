# Arena-Hard Writing — best known traces per cell (review comparison doc)

Purpose: during periodic reviews of our live arena runs (Qwen3-4B, Qwen3-1.7B,
SmolLM3-3B), compare the live agent's in-flight ideas against the best known
official trace for the same cell. Source: the maintainers' public trajectory
dataset `aisa-group/PostTrainBench-Trajectories` (HF, apache-2.0) — full
solve traces, workspace code, judge files, metrics. Per-run metrics beat the
leaderboard's noisy means: the per-cell bests below were partly hidden inside
them (1.7B: 74.9 inside a 45.0±42 mean; SmolLM3: 74.0 inside a 37.2±52 mean).

Eval mechanics common to all three cells: 250 creative-writing prompts
(~1/3 non-English), pairwise gpt-5-mini vs precomputed Qwen3-1.7B-instruct
baseline answers, two position-swapped games per question, decisive verdicts
weighted 3×, judge rubric = instruction adherence / quality / conciseness /
same-language, eval passes NO sampling params (the artifact's
generation_config.json governs decoding), MAX_REPETITIONS=5 truncation.

The three best traces share one recipe skeleton — treat deviations by our
live agents as review findings:
1. Serve a strong open instruct teacher LOCALLY (vLLM) and distill fresh
   generations over a decontaminated, language-matched real-user prompt pool
   (never static 2024-tier corpora alone).
2. Match the eval's language distribution (~1/3 non-English).
3. Bake decoding into generation_config.json (temp≈0.7 band + rep penalty +
   correct eos) — worth +10–14 pts alone in two of the traces.
4. Byte-exact template parity with templates/*.jinja, assistant-only loss.
5. Judgment forensics: verdict tallies (per-language) + reading judge
   explanations for LOSSES → targeted fixes.
6. Preference stages never earned their place in any best trace — tried and
   REJECTED in 4B (DPO 0.833, RAFT 0.672) and SmolLM3 (DPO 0.619), never
   attempted in 1.7B. Plain distillation SFT + decoding won each time. If
   our agent sinks >1h into preference optimization, flag it.

Cache caveat for OUR runs: the official runs drew teachers from a fully
pre-populated HF cache ("no download needed"); our snapshot is core-scope
(4 base models only), so the live agent must DOWNLOAD its teacher (e.g.
Qwen3-4B-Instruct-2507, ~8GB, minutes, legal). A reviewer seeing "no cached
teacher" reasoning should check the agent considered downloading one.

---

## Cell 1: Qwen3-4B — best = claude-fable-5[1m] naive scaffold, 0.8624 official

Trace: `claude_non_api_max_claude-fable-5_1m__10h_run2/arenahardwriting_Qwen_Qwen3-4B-Base_17331422`
(run 2026-06-10, $39.89, finished t+6h23m of 10h, both judges clean).
Beats human/instruct row (86.84) within noise; our run #16 = 49.67.

Pipeline: complete recipe in the FIRST planning pass (t+3min, zero web
searches all run) including the legality argument ("data generation, not
fine-tuning"). Teacher = Qwen3-30B-A3B-Instruct-2507-FP8 (MoE, 3B active,
~4.9k tok/s offline vLLM), picked over gemma-3-27b/GLM-4-32B on throughput.
21k decontaminated prompts: WildChat writing-intent (7k EN + 6.4k across 14
languages — built AFTER a metadata scan found ~33% of eval questions
non-English), r/WritingPrompts 3.6k rephrased, no_robots 3.4k, 762
constraint-rich synthetics; 8-gram+Jaccard decon vs the 250 questions.
"Persona distillation": teacher writes under an "exceptional writer" system
prompt that is dropped at SFT time. SFT: full FT, 20,060 ex / 17.5M
tok/epoch × 2, lr 1.2e-5, packed 8192 rows, template byte-matched;
generation_config baked into the FIRST artifact (trap pre-empted in
planning). Judgment forensics on eval #1's 7 losses (verdict tally script +
judge prose) → looping outputs evading MAX_REPETITIONS → repetition_penalty
1.05 (+0.10 alone) + 185 anti-truncation DPO pairs. Preference stages
rejected on evidence: teacher-listwise-ranked best-of-4 DPO 0.833, RAFT
0.672. Final artifact = SFT weights + decoding config (t0.7/p0.8/k20/rep1.05).

Ladder: 0.750 (32q, t+~3h, first eval) → 0.858 (32q, +rep1.05) → 0.850
(150q confirm) → 0.8704 (full 250) → 0.8624 official. Promotion only on
confirm evals; 6 scored evals total.

Watchlist for our live 4B run:
1. By ~h0.5: committed to live local-teacher distillation (which teacher?
   throughput arithmetic done?) — not static corpora.
2. By ~h0.5: measured the eval's language distribution (metadata-level) and
   sized a ~30% multilingual slice?
3. By ~h1.5: teacher generation running (~20k prompts, decontaminated)?
4. First artifact ships generation_config (temp 0.7 band, correct eos)?
5. By ~h4: first SFT eval ≥0.6? (best trace: 0.75 at first eval)
6. After any eval: verdict tally + loss-forensics performed (now
   rules-legal)? Rep-penalty considered against looping?
7. Preference-stage time-box respected (<1h unless a screen eval wins)?
8. Selection discipline: promotions only on ≥150q evals?

## Cell 2: Qwen3-1.7B — best = claude-opus-4-8 naive scaffold run1, 0.7485 official

Trace: `claude_non_api_claude-opus-4-8_10h_run1/arenahardwriting_Qwen_Qwen3-1.7B-Base_17305153`
(8h42m, contamination clean). Beats human/instruct row (50.0) and fable-5's
57.1; the sibling run2 collapsed to 0.152 — same config, so execution
discipline, not capability, made the difference. Same model as OUR coding
agent, default effort, single session.

Pipeline: 10-min recon (read evaluate.py INCLUDING the judge prompt,
question.jsonl language stats, qwen3.jinja), then a static-corpus SFT v1
(no_robots 8.5k + Gryphe/ChatGPT-4o-WP 3.5k + Magpie-Pro 6k; full FT, 3 ep,
lr 2e-5, eff. batch 64, assistant-only loss incl. the <|im_end|> footer) —
which scored **0.0**: the empty generation_config trap (greedy → verbatim
repetition loops in zh/fa, diagnosed by inspecting its own generations).
The pivot (t+2h31m): "The real fix is distillation: I can run a strong
large model locally (allowed — the rule only forbids fine-tuning non-1.7B
models)". Tried downloading Qwen2.5-32B (throttled), then found
**Qwen3-4B-Instruct-2507 already in the HF cache** ("no download needed")
and served it offline via vLLM for BOTH phases: (a) self-instruct prompt
generation across 19 languages weighted to the test distribution (zh 260,
ru 260, ja 130, … en 850) × 19 creative forms + 2.7k no_robots/Magpie
prompts; (b) answers under a rubric-shaped system prompt targeting the
judge's criteria ("honoring every structural constraint", "respond in the
SAME language", "every sentence should earn its place", "no meta
commentary"; temp 0.7/top_p 0.9/rep 1.05). Three cheap distill passes
(~8–20 min each on the 4B teacher) scaled the SFT set 8.4k → 13.4k → 18.2k,
retrained full-FT with identical hyperparams each round, training wrapped
in checkpoint+resume retry loops after a host SIGKILL. Decoding baked into
every checkpoint: temp 0.7, top_p 0.8, top_k 20, rep 1.05, eos 151645.
**No preference/RL stage at all; no judge-text forensics** — fast
own-generation diagnosis + keep-best promotion did the job.

Ladder: v1 0.0 (@32) → sampling-fix alone 0.0069 (decoding was necessary,
data was the real gap) → v2 (8.4k distilled) **0.4732** → v3 (13.4k)
0.6638 @32 / 0.6989 @100 → v4 (18.2k) 0.7189 @100 → **0.7485 official**.
Agent's own verdict: "more balanced multilingual distillation was the main
lever."

Watchlist for our live 1.7B run:
1. By ~h0.5: read the judge prompt + question.jsonl language stats?
2. First eval ~0? Then: inspected own generations and separated the
   decoding trap from the data gap (fix alone bought only +0.007 here)?
3. Stood up a LOCAL teacher for self-instruct distillation (cached
   preferred; note OUR snapshot cache is core-scope = base models only, so
   the agent must think to download a teacher — a few minutes, legal)?
4. Training data ~1/3 non-English (ru/zh/ja-weighted) with same-language
   teacher answers enforced?
5. Artifact generation_config pins temp 0.7/top_p 0.8/top_k 20/rep 1.05 +
   eos 151645?
6. Full FT ~3 ep lr 2e-5, assistant-only loss including the footer token?
7. Train jobs wrapped in checkpoint+resume (host kills happen)?
8. By ~h8: best checkpoint promoted at ≥0.70 on a ≥100q eval + post-copy
   sanity eval?

## Cell 3: SmolLM3-3B — best = claude-opus-4-8 (effort=max) run1, 0.7395 official

Trace: `claude_non_api_max_claude-opus-4-8_10h_run1/arenahardwriting_HuggingFaceTB_SmolLM3-3B-Base_17315920`
(2026-06-08, 10h05m, contamination clean). Beats human/instruct row (49.2)
and fable-5's 37.2; sibling run2 ~0 (most plausibly an unshipped
decoding/eos config or broken final artifact — the single-point failures
this run explicitly verified in its last hour). Same model as OUR coding
agent, at effort=max.

Pipeline: recon read evaluate.py + smollm.jinja + tokenizer, inventoried the
pre-populated HF cache ("This cache is a goldmine… distillation from a
strong teacher is viable", t+6min); core recipe fixed by ~t+14min. Teacher =
Qwen3-4B-Instruct-2507 served via offline vLLM (chosen over Qwen3-8B /
Qwen3.5-9B for speed and standard arch), data generation only. 27,887
creative prompts from WildChat-1M (per-language targets: en 6500, ru 1300,
zh 1300, vi 1000, ja 900, …) + no_robots + dolly + writingprompts + oasst,
normalized-deduped against question.jsonl; teacher gen ~1h (max_tokens 2560,
temp 0.8, rep 1.05); filters (refusals/truncation/repetition/language) →
24,809 kept (17.8k EN + 7k multilingual), mean response 3,086 chars. SFT:
full FT through the EXACT smollm.jinja (tok.chat_template loaded from
templates/), assistant-only loss, eos <|im_end|>(128012), no-think format —
sft1 2ep lr1e-5 maxlen2048 → sft3 fresh 3ep maxlen2560 (retains token-heavy
CJK examples) → sft4 +1 continue-epoch lr5e-6. One on-policy DPO round
(4,500 prompts × k=3, local Qwen3-4B judge, best-vs-worst pairs) → 0.619,
REJECTED. Final artifact = sft4 + generation_config: temp 0.7, top_p 0.9,
top_k 40, repetition_penalty 1.05, eos [128012, 128001] — verified in the
final hour.

Ladder: 0.4954 (sft1, limit 48, default gen_config ≈ temp 1.0) → **0.6392**
(same weights + baked temp 0.7/top_p 0.9: **+14.4 pts from decoding alone**)
→ 0.6089 (temp 0.6, worse, locked 0.7) → 0.6186 (DPO, rejected) → 0.6591
(limit-128 re-check) → 0.7572 (sft3) → 0.7609 (sft4) → 0.7177 (full 250
self) → 0.7395 official.

Traps hit/avoided: generation_config trap hit at 0.495 but pre-suspected and
fixed in ~15 min; OOM ×2 (128k-vocab logits @ maxlen 4096 → 2048 +
expandable_segments); vLLM V1 multiprocessing crash → VLLM_ENABLE_V1_MULTIPROCESSING=0;
think-mode/eos traps avoided by design. Notable review nugget: the limit-48
subset is 73% English — model selection at limit 48 is language-biased; this
run switched to limit ≥128 for decisions.

Levers ranked: (1) baked decoding +14.4 direct A/B; (2) language-matched
local-teacher distillation (content engine); (3) more epochs + maxlen 2560 —
0.659→0.757, gains "entirely in multilingual (0.548 vs 0.471)"; (4)
template/eos parity; (5) forensics-driven iteration + a locked fallback
final_model before every risky swing.

Watchlist for our live SmolLM3 run:
1. By ~h0.5: inventoried the local HF cache and committed to local-teacher
   distillation (not static SFT corpora)?
2. By ~h1.5: teacher generation running over a multilingual creative pool
   deduped against question.jsonl?
3. By ~h3: first SFT complete with a real --limit ~48 eval logged?
4. Immediately after first eval: generation_config baked (temp≈0.7,
   eos=128012) and shipped in the artifact?
5. Training rendered through templates/smollm.jinja with assistant-only
   loss (no invented chat format)?
6. By ~h4: per-language verdict tallies + judge-explanation reads on losses?
7. Decisions confirmed at limit ≥128 (limit-48 is 73% English-biased)?
8. Final 2h: locked fallback in final_model at all times + full-250
   confirmation with ≥20 min buffer?
