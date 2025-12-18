{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|QLoRA|https://arxiv.org/abs/2305.14314]]
|-
! Domains
| [[domain::Quantization]], [[domain::Training]], [[domain::Memory_Efficiency]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Concrete tool for preparing quantized models for k-bit training by enabling gradient checkpointing and proper dtype handling.

=== Description ===

`prepare_model_for_kbit_training` prepares a quantized model for fine-tuning by:
1. Freezing all base model parameters
2. Casting layer norms to float32 for stability
3. Enabling gradient checkpointing for memory efficiency
4. Setting up input embeddings to require gradients

This function is essential for stable QLoRA training.

=== Usage ===

Call this immediately after loading a quantized model, before applying PEFT. Set `use_gradient_checkpointing=True` (default) to reduce VRAM usage at the cost of ~20% slower training.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft peft]
* '''File:''' src/peft/utils/other.py
* '''Lines:''' L130-215

=== Signature ===
<syntaxhighlight lang="python">
def prepare_model_for_kbit_training(
    model: PreTrainedModel,
    use_gradient_checkpointing: bool = True,
    gradient_checkpointing_kwargs: Optional[dict] = None
) -> PreTrainedModel:
    """
    Prepare a quantized model for training.

    Args:
        model: Quantized model from transformers
        use_gradient_checkpointing: Enable gradient checkpointing. Default: True
        gradient_checkpointing_kwargs: Additional args for checkpointing

    Returns:
        Model prepared for k-bit training
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import prepare_model_for_kbit_training
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel || Yes || Quantized model (4-bit or 8-bit)
|-
| use_gradient_checkpointing || bool || No || Enable gradient checkpointing. Default: True
|-
| gradient_checkpointing_kwargs || dict || No || Args like {"use_reentrant": False}
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PreTrainedModel || Model prepared for k-bit training (modified in-place, also returned)
|}

== Usage Examples ==

=== Standard QLoRA Preparation ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
import torch

# 1. Load quantized model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# 2. Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# 3. Now apply PEFT
# model = get_peft_model(model, lora_config)
</syntaxhighlight>

=== With Custom Checkpointing ===
<syntaxhighlight lang="python">
from peft import prepare_model_for_kbit_training

# Use non-reentrant checkpointing (recommended for newer PyTorch)
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_peft_Kbit_Training_Preparation]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_peft_Quantization_Environment]]
