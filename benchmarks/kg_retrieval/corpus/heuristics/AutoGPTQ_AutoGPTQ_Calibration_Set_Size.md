# Heuristic: Calibration_Set_Size

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers|https://arxiv.org/abs/2210.17323]]
* [[source::Repo|AutoGPTQ|https://github.com/AutoGPTQ/AutoGPTQ]]
|-
! Domains
| [[domain::LLMs]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Guidance on how many calibration sequences and tokens a post-training compression pass needs before more data stops improving the result.

=== Description ===

Calibration data only has to expose typical activation statistics, so returns diminish quickly. Too little data overfits the rounding decisions to a few examples; too much slows compression without a measurable gain.

=== Usage ===

Consult it when preparing the sample inputs for a compression run.

== The Insight (Rule of Thumb) ==

Use about 128 sequences of 512 to 2048 tokens, drawn from text that resembles the deployment domain. Doubling the amount rarely changes perplexity by more than a few hundredths, while mismatched data can cost far more.

== Reasoning ==

Statistics are averaged over all tokens, so the token count matters more than the number of sequences, and diversity matters more than either.
