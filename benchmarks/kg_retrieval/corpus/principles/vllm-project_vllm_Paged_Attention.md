# Principle: Paged_Attention

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Efficient Memory Management for Large Language Model Serving with PagedAttention|https://arxiv.org/abs/2309.06180]]
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::LLMs]], [[domain::Inference]], [[domain::Serving]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Attention kernel and memory manager that keeps the key-value cache in fixed-size blocks located through a block table, avoiding fragmentation and unused reservations.

=== Description ===

Contiguous per-request allocation wastes memory because the final length of a response is unknown: space is reserved for the maximum and much of it stays empty, and gaps appear between requests. Treating the cache like virtual memory, with small blocks assigned on demand and a table mapping logical positions to physical blocks, removes nearly all of this waste and allows sharing of blocks between sequences with a common prefix.

=== Usage ===

It matters when many sequences of unpredictable length share one accelerator.

== Theoretical Basis ==

With block size b tokens, the average internal waste is under b tokens per sequence instead of up to the full maximum length. The attention kernel reads each block through the table, so a sequence's blocks need not be adjacent.

== Related Pages ==

* [[implemented_by::Implementation:vllm-project_vllm_LLMEngine]]
