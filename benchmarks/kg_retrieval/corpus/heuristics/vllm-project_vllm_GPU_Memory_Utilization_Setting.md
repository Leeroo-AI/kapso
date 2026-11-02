# Heuristic: GPU_Memory_Utilization_Setting

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|vLLM documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::LLMs]], [[domain::Inference]], [[domain::Serving]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Guidance for choosing the gpu_memory_utilization fraction, which caps the share of device memory the engine claims for weights, activations and cache blocks.

=== Description ===

The engine profiles a forward pass at start-up and gives everything that remains within its share to cache blocks. A larger share allows more concurrent sequences; a share that is too large leaves no room for other processes or allocator spikes.

=== Usage ===

Consult it when tuning concurrency or when start-up fails with an out-of-memory error.

== The Insight (Rule of Thumb) ==

Begin at 0.90 on a dedicated card. Lower it to 0.80 or less when another process uses the device or when long prompts cause sporadic failures; raise it toward 0.95 only after confirming stable behaviour under peak load.

== Reasoning ==

Activation peaks are only seen when real traffic arrives, so unused headroom absorbs them. Cache blocks beyond what concurrency needs bring no benefit.
