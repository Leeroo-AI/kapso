# Principle: Continuous_Batching

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Paper|Orca: A Distributed Serving System for Transformer-Based Generative Models|https://www.usenix.org/conference/osdi22/presentation/yu]]
|-
! Domains
| [[domain::LLMs]], [[domain::Inference]], [[domain::Serving]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Scheduling policy that admits new requests and retires finished ones at every decoding iteration instead of waiting for a whole group of sequences to complete.

=== Description ===

With group-level scheduling, a batch lasts as long as its longest response and shorter ones leave idle slots. Iteration-level scheduling lets a finished sequence's slot be filled by a waiting request on the very next step, keeping the accelerator saturated and cutting queueing delay.

=== Usage ===

It is the basis of high-throughput generation servers and is combined with a block-based cache.

== Theoretical Basis ==

Each iteration runs one forward step for all active sequences, whatever their ages. The scheduler decides before each step which waiting requests fit in the available cache blocks, and preempts or swaps sequences when memory runs short.

== Related Pages ==

* [[implemented_by::Implementation:vllm-project_vllm_LLMEngine]]
