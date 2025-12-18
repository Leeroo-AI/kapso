{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|QLoRA|https://arxiv.org/abs/2305.14314]]
|-
! Domains
| [[domain::Fine_Tuning]], [[domain::Quantization]], [[domain::Memory_Optimization]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Principle for configuring LoRA parameters specifically optimized for quantized (QLoRA) training.

=== Description ===

QLoRA Configuration creates a LoraConfig tuned for training on 4-bit quantized models. Key differences from standard LoRA config:

* **target_modules="all-linear"**: QLoRA benefits from adapting all linear layers since quantization compresses the base model
* **Task type specification**: Required for proper adapter injection
* **Typical ranks**: r=16-64 common as quantization adds implicit regularization

The adapter weights remain in full precision (float32) despite the base model being quantized.

=== Usage ===

Apply when setting up LoRA configuration for a quantized model:
* Use "all-linear" for comprehensive adaptation
* Consider slightly higher ranks than standard LoRA
* Always specify task_type for proper layer targeting

== Theoretical Basis ==

'''QLoRA Architecture:'''

<syntaxhighlight lang="text">
Base Model (4-bit quantized, frozen)
         |
         v
    Dequantize
         |
         v
    W_0 * x (full precision compute)
         |
         v
    + B*A*x (LoRA, float32)
         |
         v
       Output
</syntaxhighlight>

The LoRA matrices B and A remain in full precision, ensuring stable gradient flow despite quantized base weights.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_peft_LoraConfig_for_qlora]]
