{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA|https://arxiv.org/abs/2106.09685]]
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::Adapter]], [[domain::Model_Loading]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Principle for loading pre-trained adapter weights onto a base model for inference or continued training.

=== Description ===

Adapter Loading reconstructs a trained PEFT model by:
1. Loading the adapter configuration (JSON)
2. Injecting adapter layers into the base model
3. Loading and placing adapter weights
4. Setting appropriate mode (inference vs. training)

This enables deployment of task-specific adaptations without redistributing the full model.

=== Usage ===

Apply this when you have a trained adapter checkpoint and need to use it:
* **Inference:** Set `is_trainable=False` to freeze adapters
* **Continued training:** Set `is_trainable=True` for gradient updates
* **Hub loading:** Use HuggingFace model ID directly
* **Local loading:** Use path to saved adapter directory

== Theoretical Basis ==

'''Adapter Reconstruction:'''

Loading restores the adapted weights:
<math>W_{adapted} = W_0 + BA</math>

Where:
* <math>W_0</math> is loaded from the base model
* <math>B, A</math> are loaded from the adapter checkpoint

'''Weight Placement:'''

The loading process:
<syntaxhighlight lang="python">
# Pseudo-code for adapter loading
config = LoraConfig.from_pretrained(model_id)  # Load JSON config

# Inject layers based on config
for target_name in config.target_modules:
    module = get_module(model, target_name)
    inject_lora_layer(module, config.r, config.lora_alpha)

# Load trained weights
state_dict = load_safetensors(f"{model_id}/adapter_model.safetensors")
model.load_state_dict(state_dict, strict=False)
</syntaxhighlight>

'''Inference Mode:'''

For inference, adapters are frozen:
<syntaxhighlight lang="python">
if not is_trainable:
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = False
    model.eval()  # Set to eval mode
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_peft_PeftModel_from_pretrained]]
