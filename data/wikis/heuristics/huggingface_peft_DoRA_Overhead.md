# Heuristic: huggingface_peft_DoRA_Overhead

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|DoRA Paper|https://huggingface.co/papers/2402.09353]]
|-
! Domains
| [[domain::LLMs]], [[domain::Optimization]], [[domain::DoRA]]
|-
! Last Updated
| [[last_updated::2024-12-18 00:00 GMT]]
|}

== Overview ==
DoRA improves low-rank adaptation quality but adds computational overhead; merge weights for efficient inference.

=== Description ===
DoRA (Weight-Decomposed Low-Rank Adaptation) decomposes weight updates into magnitude and direction components. This improves LoRA quality, especially at low ranks, but introduces additional overhead during training and inference. Understanding this trade-off is crucial for deciding when to use DoRA.

=== Usage ===
Use this heuristic when:
- Deciding whether to enable `use_dora=True` in LoraConfig
- Training at low ranks (r <= 8) where DoRA shows most benefit
- Planning inference deployment of DoRA-trained adapters

== The Insight (Rule of Thumb) ==

* **When to Use DoRA:**
  * Low-rank scenarios (r <= 8) where quality matters
  * Tasks where standard LoRA underperforms
  * When you can afford the training overhead

* **When to Avoid DoRA:**
  * High-rank scenarios (r >= 32) where LoRA is sufficient
  * Megatron parallelism (not supported)
  * When `lora_bias=True` (not compatible)

* **Overhead:**
  * Training: ~10-20% slower per step
  * Inference (unmerged): Additional magnitude computation
  * Inference (merged): No overhead after merging

* **Recommendation:**
  * For production: Merge DoRA weights after training (`merge_and_unload()`)
  * For experimentation: Keep unmerged for easy adapter switching

* **Layer Support:**
  * Currently only supports Linear and Conv2D layers
  * Does not support Megatron parallel layers

* **Ephemeral GPU Offload:**
  * DoRA initialization can be slow on CPU-offloaded models
  * Use `ephemeral_gpu_offload=True` to speed up DoRA initialization

== Reasoning ==

### How DoRA Works
DoRA decomposes weight updates as:
- **Direction:** Handled by standard LoRA (low-rank matrices)
- **Magnitude:** Handled by a separate learnable parameter

This decomposition better mimics full fine-tuning behavior and typically achieves better results at the same rank.

### Overhead Source
The additional magnitude parameter requires:
1. Extra forward pass computation for magnitude normalization
2. Extra backward pass computation for magnitude gradients
3. Additional memory for magnitude parameters

### Merging Benefit
Once merged, DoRA weights become regular dense weights with no inference overhead. This makes DoRA attractive for production:
- Train with DoRA for better quality
- Merge for deployment with no overhead

== Code Evidence ==

DoRA configuration from `config.py:634-645`:
<syntaxhighlight lang="python">
use_dora: bool = field(
    default=False,
    metadata={
        "help": (
            "Enable 'Weight-Decomposed Low-Rank Adaptation' (DoRA). This technique decomposes the updates of the "
            "weights into two parts, magnitude and direction. Direction is handled by normal LoRA, whereas the "
            "magnitude is handled by a separate learnable parameter. This can improve the performance of LoRA, "
            "especially at low ranks. Right now, DoRA only supports linear and Conv2D layers. DoRA introduces a bigger"
            "overhead than pure LoRA, so it is recommended to merge weights for inference."
        )
    },
)
</syntaxhighlight>

Megatron incompatibility from `config.py:795-796`:
<syntaxhighlight lang="python">
if self.use_dora and self.megatron_config:
    raise ValueError("DoRA does not support megatron_core, please set `use_dora=False`.")
</syntaxhighlight>

lora_bias incompatibility from `config.py:833-834`:
<syntaxhighlight lang="python">
if self.use_dora:
    raise ValueError("The argument lora_bias=True is not supported for DoRA, please pass use_dora=False")
</syntaxhighlight>

Ephemeral GPU offload for DoRA from `config.py:37-50`:
<syntaxhighlight lang="python">
ephemeral_gpu_offload: bool = field(
    default=False,
    metadata={
        "help": (
            "Whether to use ephemeral GPU offloading for models partially kept in CPU memory. "
            "...Currently only affects DoRA initialization."
        )
    },
)
</syntaxhighlight>

== Related Pages ==
* [[uses_heuristic::Implementation:huggingface_peft_LoraConfig_init]]
* [[uses_heuristic::Principle:huggingface_peft_LoRA_Configuration]]
* [[uses_heuristic::Workflow:huggingface_peft_LoRA_Fine_Tuning]]
