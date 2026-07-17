# Run #12 review log — bfcl × gemma-3-4b-pt, 10h (2026-07-17)

`bfcl-gemma-3-4b-pt-07170825`, first gemma run (gated model, fresh cache
bake), built from `46e6390f`. Cell: base 6.0 published (0.03 measured),
human 67.0, proven top three all 100.0.

## Pass 1 (08:28Z–10:32Z, t+2h04)

Verdict: **0 majors, 2 minor, 6 info — "cleanest first two hours of any
run in the campaign."**

All three gemma-specific risks explicitly de-risked BEFORE training:
cache bake worked (model loaded locally, no gating errors, token file
present); template handling exemplary (read gemma3_tool_calling.jinja,
verified turn-marker token ids, byte-exact render checks "ALL PREFIX
CHECKS PASSED", untrained-stop-token trap pre-fixed via eos_token_id
[1,106]); vision tower confirmed and frozen (LoRA on language_model
only, 2.7% trainable). Overlapped baseline eval with data prep — near
zero idle GPU.

**Minors:** R12-P1-1 — agent printed $HF_HOME/token contents; 20 of 37
chars of the HF token now sit in solve_out.txt in GCS (not usable alone;
backlog: runner-side hf_[A-Za-z0-9]+ redaction). R12-P1-2 — merged model
ships base gemma sampling config (do_sample true, top_k 64): official
eval decodes with sampling while the 92% smoke was greedy; watch.

**Infos:** baseline 0.03 vs published 6.0 (conservative floor); ensemble
2/2+2/2, selector rejected over-scoped candidates ("teaches auto-fail
behaviors"); 4 rate-limit events zero stalls; ~6 min total waste, no
OOMs (batch-16 probe near-OOM consciously rejected); data =
minpeter/xlam-function-calling-60k-parsed mirror (gated original
correctly refused), clean provenance; luna memory mechanically fine
(empty store, iteration 1); contract discipline exemplary (sizing 35%
of session, all jobs PID-filed, polls ≤300s, kills none).

State at review end: LoRA trained (85 min, loss 0.019), merged, smoke
92/100 held-out with 100/100 parseability, first official eval in
flight (PID 7146).
