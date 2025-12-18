{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Repo|bitsandbytes|https://github.com/bitsandbytes-foundation/bitsandbytes]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Quantized AdaLoRA layer implementations for 8-bit and 4-bit bitsandbytes linear layers, enabling SVD-based adaptive rank adaptation on quantized models.

=== Description ===

SVDLinear8bitLt and SVDLinear4bit implement AdaLoRA for bitsandbytes quantized layers. The layers use SVD decomposition with trainable lora_A, lora_B, and lora_E (importance scores) matrices. The forward pass computes: result + ((dropout(x) @ (lora_A * lora_E).T @ lora_B.T) * scaling / ranknum). For 4-bit, result is defensively cloned for backprop compatibility.

=== Usage ===

Use AdaLoRA quantized layers when fine-tuning large quantized models with adaptive rank allocation. Layers are automatically dispatched when base layers are bitsandbytes Linear8bitLt or Linear4bit. Note: merging is not supported for quantized AdaLoRA.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/adalora/bnb.py src/peft/tuners/adalora/bnb.py]
* '''Lines:''' 1-144

=== Signature ===
<syntaxhighlight lang="python">
class SVDLinear8bitLt(torch.nn.Module, AdaLoraLayer):
    """AdaLoRA for 8-bit quantized layers."""
    def __init__(
        self,
        base_layer: torch.nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        **kwargs,
    ) -> None:
        """Initialize AdaLoRA for 8-bit layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SVD-based adaptation to quantized layer."""

class SVDLinear4bit(torch.nn.Module, AdaLoraLayer):
    """AdaLoRA for 4-bit quantized layers."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.adalora.bnb import SVDLinear8bitLt, SVDLinear4bit
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| base_layer || bnb.nn.Linear8bitLt/Linear4bit || Yes || Quantized base layer
|-
| adapter_name || str || Yes || Name for the adapter
|-
| r || int || Yes || Initial rank
|-
| lora_alpha || int || No || Scaling factor (default: 1)
|-
| lora_dropout || float || No || Dropout probability
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || torch.Tensor || Base output + SVD adaptation
|}

== Usage Examples ==

=== AdaLoRA with 4-bit Quantization ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import AdaLoraConfig, get_peft_model

# Load model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
)

# Apply AdaLoRA with rank scheduling
config = AdaLoraConfig(
    init_r=12,
    target_r=4,
    total_step=10000,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, config)
# Automatically uses SVDLinear4bit class
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Quantization_Environment]]
