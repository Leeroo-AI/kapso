{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|GPTQ|https://arxiv.org/abs/2210.17323]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

AdaLoRA implementation for GPTQ-quantized linear layers, enabling SVD-based adaptive rank adaptation on GPTQ models.

=== Description ===

SVDQuantLinear implements AdaLoRA for GPTQ-quantized layers. It wraps a GPTQ quantized linear module and applies SVD-based adaptation with trainable lora_A, lora_B, and lora_E matrices. The forward pass computes: result + ((dropout(x) @ (lora_A * lora_E).T @ lora_B.T) * scaling / ranknum). Input is cast to float32 when autocast is disabled.

=== Usage ===

Use SVDQuantLinear when applying AdaLoRA to GPTQ-quantized models. The layer is automatically dispatched when the base layer is a GPTQ QuantLinear. Works with models loaded via AutoGPTQ or transformers GPTQ support.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/adalora/gptq.py src/peft/tuners/adalora/gptq.py]
* '''Lines:''' 1-72

=== Signature ===
<syntaxhighlight lang="python">
class SVDQuantLinear(torch.nn.Module, AdaLoraLayer):
    """AdaLoRA for GPTQ-quantized layers."""
    def __init__(
        self,
        base_layer,
        adapter_name,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        **kwargs,
    ) -> None:
        """Initialize AdaLoRA for GPTQ layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SVD adaptation to GPTQ layer."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.adalora.gptq import SVDQuantLinear
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| base_layer || QuantLinear || Yes || GPTQ quantized layer
|-
| adapter_name || str || Yes || Name for the adapter
|-
| r || int || Yes || Initial rank
|-
| lora_alpha || int || No || Scaling factor
|-
| lora_dropout || float || No || Dropout probability
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || torch.Tensor || GPTQ output + SVD adaptation
|}

== Usage Examples ==

=== AdaLoRA with GPTQ Model ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import AdaLoraConfig, get_peft_model

# Load GPTQ-quantized model
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-GPTQ",
    device_map="auto",
)

# Apply AdaLoRA
config = AdaLoraConfig(
    init_r=12,
    target_r=4,
    total_step=10000,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, config)
# Automatically uses SVDQuantLinear for GPTQ layers
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_GPTQ_Environment]]
