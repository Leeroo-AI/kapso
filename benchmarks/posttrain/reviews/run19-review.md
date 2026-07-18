# Run #19 review — arenahardwriting × SmolLM3-3B-Base (0718134)

Rebuilt stack (580a74a3 + 3ba24c4d). Dual-mandate reviews per
`arena-best-baseline-traces.md`; best-known trace for this cell:
opus-4.8-max 73.95 (local Qwen3-4B-Instruct-2507 teacher, 24.8k
multilingual through exact smollm.jinja). Launched 2026-07-18 13:43Z.

## P1 (t+0 → t+86min)

Headline: plan matches the 73.95 skeleton on 4/6 pillars ahead of the best
trace's clock — teacher (Qwen3-30B-A3B-Instruct-2507, 57GB downloaded in
2.6 min without hesitation), byte-parity smollm.jinja rendering with
empty-think targets, eos 128012 verified, generation_config baked
pre-first-eval, baseline 0.0 @32 confirmed, teacher gen 10,620 prompts in
~24 min @5.1k tok/s → 9,988 kept, SFT running (loss 1.82→1.52) by segment
end. Relaxed rules semantically effective ("Ranking with a locally-served
open model is explicitly legal per the harness rules").

- **R19-P1-1 — P1 major (recipe), 13:56→14:20.** Multilingual slice
  undersized AND wrong language set; no language-distribution scan
  (recon was `head -c 2000` → "multilingual prompts present (es, etc.)").
  Adopted ≈10% multilingual restricted to es/fr/de/it/pt ("SmolLM3's
  languages"); implementation Counter shows 1,120/10,620 (10.5%). The eval
  is ~1/3 non-English INCLUDING zh/ru/ja/vi; the 73.95 trace used
  7k/24.8k with ru/zh/vi/ja targets and attributed its late gains
  "entirely in multilingual". Worst of the three runs on this axis; risk
  masked at limit-48 (73% English). Watch P2 for self-correction via
  judge forensics.
- **R19-P1-2 — P2 (recipe watch), 14:00:44.** On-policy DPO adopted as
  phase 2 (~2.5h/round, second round contemplated), downside framed as
  "bounded to zero" — ignores opportunity cost; the best SmolLM3 trace
  REJECTED DPO (0.619). Mitigated: gated behind a promoted SFT anchor;
  zero DPO time spent in P1. Enforce the >1h flag in P2.
- **R19-P1-3 — P2 (framework/stream), 13:58:38.** Codex member output not
  streamed; candidates auditable only via selector critique. RECURRING
  (3/3 runs — systemic, = R17-P1-2/R18-P1-2).
- **R19-P1-4 — OBS (framework, positive), 14:24/14:29/15:00.** env_defaults
  clock policy active: three foreground sleeps blocked with Monitor
  guidance, agent adapted immediately each time; no auto-background
  surprises.
- **R19-P1-5 — OBS, 13:51:24.** repo-memory core.architecture
  placeholder-only (fresh campaign, expected); R15-P2-1/R16-P2-2 not
  exercised (no feedback pass yet).
- **R19-P1-6 — OBS, 13:51/15:03.** Micro-waste: two empty ToolSearch calls
  at member boot (RECURRING 3/3), one wasted Read, duplicate "MCP config
  written" log line. Cosmetic.
- **R19-P1-7 — OBS (positive).** Boot clean; both members 2/2; selector
  verified evaluate.py claims line-by-line and corrected C3's OOM-risky
  batch; implementation start clean; vLLM generation_config="auto" premise
  re-verified in-session.

Verdict: **concern** — framework health excellent and execution ahead of
schedule, but the 10% wrong-language multilingual slice is the known #2
lever left on the table for this cell; watch whether P2's judge forensics
self-corrects it.
