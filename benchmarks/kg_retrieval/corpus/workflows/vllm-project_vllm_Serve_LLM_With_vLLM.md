# Workflow: Serve_LLM_With_vLLM

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM documentation|https://docs.vllm.ai]]
* [[source::Paper|Efficient Memory Management for Large Language Model Serving with PagedAttention|https://arxiv.org/abs/2309.06180]]
|-
! Domains
| [[domain::LLMs]], [[domain::Inference]], [[domain::Serving]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Recipe for exposing a pretrained language model behind an HTTP endpoint that handles many concurrent generation requests with high throughput.

=== Description ===

A production generation endpoint must keep the accelerator busy across requests of very different lengths. The recipe selects an engine with a block-based memory manager and an iteration-level scheduler, sizes its memory pool, and verifies latency under load.

=== Usage ===

Follow it when moving a model from interactive scripts to a service used by many clients.

== Execution Steps ==

=== Step 1: Choose weights and precision ===

Pick the checkpoint and the numeric format that fits the target card with room left over for request state.

=== Step 2: Start the engine ===

Launch the OpenAI-compatible server with the model path, maximum sequence length and the share of device memory the engine may use.

=== Step 3: Load-test ===

Replay a representative mix of prompt and output lengths at increasing concurrency and record throughput, time to first token and inter-token latency.

=== Step 4: Tune ===

Adjust the concurrency limit and memory share, and enable optional accelerators such as draft-model generation if latency matters more than throughput.

== GitHub URL ==

[https://github.com/vllm-project/vllm vllm-project/vllm]
