# Heuristic: Gradient_Checkpointing_Tip

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|loader.py|https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py]]
|-
! Domains
| [[domain::Optimization]], [[domain::Memory_Management]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2026-01-09 12:00 GMT]]
|}

== Overview ==
Use `use_gradient_checkpointing="unsloth"` for optimized memory-compute trade-off during training.

=== Description ===
Unsloth provides a custom gradient checkpointing implementation that is more efficient than the standard HuggingFace/PyTorch implementation. The "unsloth" mode intelligently checkpoints activations to reduce VRAM usage by ~50-60% while minimizing the compute overhead (typically only ~20% slower training).

Standard gradient checkpointing stores fewer intermediate activations during the forward pass and recomputes them during the backward pass. Unsloth's implementation optimizes which layers are checkpointed based on the model architecture.

=== Usage ===
Use this heuristic when:
- **VRAM constrained:** Getting CUDA OOM errors during training
- **Training large models:** Fine-tuning 7B+ parameter models on consumer GPUs (RTX 3090/4090)
- **Maximizing batch size:** Need to fit larger batches in memory

== The Insight (Rule of Thumb) ==
* **Action:** Set `use_gradient_checkpointing="unsloth"` in `FastLanguageModel.from_pretrained()` or `FastVisionModel.from_pretrained()`
* **Value:** String `"unsloth"` (not boolean `True`)
* **Trade-off:** Reduces VRAM usage by ~50-60% at the cost of ~20% slower training speed
* **Compatibility:** Requires `use_cache=False` during training (automatically set)
* **Default:** Enabled by default in Unsloth

== Reasoning ==
Deep Transformers have massive activation maps (Batch × SeqLen × Hidden). Storing all activations for backpropagation is the primary VRAM bottleneck during training. By selectively recomputing activations during the backward pass:

1. Peak VRAM is significantly reduced
2. Larger batch sizes become possible
3. Training throughput may actually improve due to better memory efficiency

Unsloth's implementation is optimized specifically for the LLM architectures it supports, resulting in better trade-offs than generic implementations.

**Benchmarks:** On Llama-2-7B, VRAM typically drops from ~22GB to ~11GB with this flag enabled.

== Code Evidence ==

From `loader.py:562-563`:
<syntaxhighlight lang="python">
if use_gradient_checkpointing == "unsloth":
    patch_unsloth_smart_gradient_checkpointing(dtype = dtype)
</syntaxhighlight>

From `FastLanguageModel.from_pretrained` signature in `loader.py:136`:
<syntaxhighlight lang="python">
use_gradient_checkpointing = "unsloth",  # Default value
</syntaxhighlight>

== Related Pages ==
* [[used_by::Implementation:Unslothai_Unsloth_FastLanguageModel_from_pretrained]]
* [[used_by::Implementation:Unslothai_Unsloth_FastVisionModel_from_pretrained]]
