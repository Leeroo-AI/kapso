{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|FlashAttention|https://arxiv.org/abs/2205.14135]]
* [[source::Paper|FlashAttention-2|https://arxiv.org/abs/2307.08691]]
* [[source::Repo|Unsloth GitHub|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Optimization]], [[domain::Memory_Management]], [[domain::Attention]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Techniques for reducing attention mechanism memory from O(n²) to O(n) using FlashAttention and Unsloth's optimizations.

=== Description ===
Standard attention requires O(n²) memory for the attention matrix, limiting sequence length. FlashAttention and related techniques compute attention in a memory-efficient manner by tiling the computation and avoiding materialization of the full attention matrix. Unsloth integrates these optimizations automatically.

=== Usage ===
Use this heuristic when training with **long sequences** (4K+ tokens) or when hitting OOM errors. Unsloth automatically enables FlashAttention when available, but understanding the mechanism helps optimize for your specific use case.

== The Insight (Rule of Thumb) ==
* **Action:** Unsloth automatically uses FlashAttention when available. Ensure xformers is installed for older GPUs.
* **Value:** Automatic - no manual configuration needed with Unsloth.
* **Manual Check:** Verify FlashAttention is active in logs.

<syntaxhighlight lang="python">
# Unsloth handles this automatically, but you can verify:
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = 8192,  # Long context enabled
    # FlashAttention automatically used if available
)
</syntaxhighlight>

* **Context Length Impact (with Unsloth optimizations):**

{| class="wikitable"
! GPU VRAM !! Max Context (Standard) !! Max Context (Unsloth)
|-
|| 8GB || OOM || 2,972
|-
|| 16GB || 2,551 || 40,724
|-
|| 24GB || 5,789 || 78,475
|-
|| 80GB || 28,454 || 342,733
|}

* **Trade-off:** Negligible - FlashAttention is faster AND more memory efficient.

== Reasoning ==
FlashAttention achieves memory efficiency by:
1. Computing attention in blocks (tiling)
2. Never materializing the full NxN attention matrix
3. Using hardware-optimized fused kernels

Unsloth further optimizes this with custom CUDA kernels that achieve 2x speedup over standard FlashAttention implementations while maintaining the memory benefits. This enables 12x longer context lengths compared to HuggingFace + FlashAttention-2.

== Related Pages ==
=== Used By ===
* [[uses_heuristic::Workflow:QLoRA_Finetuning]]
* [[uses_heuristic::Implementation:Unsloth_FastLanguageModel]]
* [[uses_heuristic::Principle:Flash_Attention]]
* [[uses_heuristic::Principle:Self_Attention]]

