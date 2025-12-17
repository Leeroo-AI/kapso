# Heuristic: unslothai_unsloth_Gradient_Checkpointing

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Blog|Unsloth Wiki|https://docs.unsloth.ai]]
|-
! Domains
| [[domain::Optimization]], [[domain::Memory_Management]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

## Overview

Memory optimization heuristic: Use `use_gradient_checkpointing="unsloth"` to reduce VRAM usage by ~30% compared to standard gradient checkpointing.

### Description

Unsloth provides an enhanced gradient checkpointing implementation that intelligently selects which layers to checkpoint. Instead of checkpointing every transformer layer (standard approach), Unsloth's smart checkpointing:

1. Analyzes memory usage patterns
2. Checkpoints strategically placed layers
3. Reduces recomputation overhead while maintaining memory savings

This allows fitting larger batch sizes or longer sequence lengths within the same VRAM budget.

### Usage

Use this heuristic when:
- Training 7B+ parameter models on consumer GPUs (RTX 3090/4090)
- Getting CUDA OOM errors during training
- Wanting to maximize batch size for faster throughput
- Fine-tuning with QLoRA or full LoRA

## The Insight (Rule of Thumb)

* **Action:** Set `use_gradient_checkpointing="unsloth"` in `FastLanguageModel.get_peft_model()` or `FastLanguageModel.from_pretrained()`
* **Value:** `"unsloth"` (string literal) - not `True` or `False`
* **Trade-off:** Reduces VRAM by ~30% with minimal (~5-10%) training speed reduction

**Recommended usage:**
<syntaxhighlight lang="python">
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    use_gradient_checkpointing = "unsloth",  # 30% less VRAM
)
</syntaxhighlight>

**Alternative values:**
- `"unsloth"`: Unsloth's optimized smart checkpointing (recommended)
- `True`: Standard HuggingFace gradient checkpointing
- `False`: Disable gradient checkpointing (fastest but uses most VRAM)

## Reasoning

Standard gradient checkpointing saves all intermediate activations at every layer boundary, then recomputes them during the backward pass. This trades compute for memory uniformly across all layers.

Unsloth's smart checkpointing observes that:
1. Not all layers have equal memory footprints
2. Some layers benefit more from checkpointing than others
3. Strategic placement reduces both memory AND recomputation

The implementation patches gradient checkpointing at the model level:

<syntaxhighlight lang="python">
# From _utils.py - smart gradient checkpointing setup
if use_gradient_checkpointing == "unsloth":
    patch_unsloth_smart_gradient_checkpointing(dtype = dtype)
</syntaxhighlight>

**Memory comparison (Llama-3.1-8B, 4-bit QLoRA, batch_size=4):**

{| class="wikitable"
! Setting !! VRAM Usage !! Training Speed
|-
| `use_gradient_checkpointing=False` || ~18GB || Fastest
|-
| `use_gradient_checkpointing=True` || ~14GB || ~20% slower
|-
| `use_gradient_checkpointing="unsloth"` || ~12GB || ~10% slower
|}

## Related Pages

* [[uses_heuristic::Implementation:unslothai_unsloth_get_peft_model]]
* [[uses_heuristic::Implementation:unslothai_unsloth_FastLanguageModel_from_pretrained]]
* [[uses_heuristic::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
* [[uses_heuristic::Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning]]
* [[uses_heuristic::Principle:unslothai_unsloth_LoRA_Configuration]]
