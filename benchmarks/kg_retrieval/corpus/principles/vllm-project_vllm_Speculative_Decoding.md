# Principle: Speculative_Decoding

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Fast Inference from Transformers via Speculative Decoding|https://arxiv.org/abs/2211.17192]]
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::LLMs]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Generation method in which a cheap draft model proposes several tokens that the target model verifies in a single forward pass, preserving the target distribution.

=== Description ===

Decoding is memory-bound, so verifying k proposed tokens costs almost the same as generating one. Accepted tokens are kept, the first rejected one is resampled from a corrected distribution, and the output distribution is exactly the target model's. Speed-up depends on how often the draft agrees with the target.

=== Usage ===

Use it when latency per request matters and a well-aligned smaller model exists.

== Theoretical Basis ==

A proposed token with draft probability q and target probability p is accepted with probability min(1, p/q). With acceptance rate a and k proposals, the expected number of tokens per target pass is (1 - a^(k+1)) / (1 - a).

== Related Pages ==

* [[implemented_by::Implementation:vllm-project_vllm_SpeculativeConfig]]
