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

Quantized VeRA layer implementations for 8-bit and 4-bit bitsandbytes linear layers, enabling vector-based random matrix adaptation on quantized models.

=== Description ===

Linear8bitLt and Linear4bit implement VeRA for bitsandbytes quantized layers. The layers use shared frozen random matrices (vera_A, vera_B) with per-layer trainable scaling vectors (lambda_d, lambda_b). The forward pass computes: result + lambda_b * linear(lambda_d * linear(x, sliced_A), sliced_B). During merge, the delta weight is added to dequantized weights then requantized.

=== Usage ===

Use VeRA quantized layers for extreme parameter efficiency on quantized models. VeRA adapters are much smaller than LoRA since only scaling vectors are trained. Automatic dispatch creates these layers when base layers are bitsandbytes Linear8bitLt or Linear4bit.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/vera/bnb.py src/peft/tuners/vera/bnb.py]
* '''Lines:''' 1-412

=== Signature ===
<syntaxhighlight lang="python">
class Linear8bitLt(torch.nn.Module, VeraLayer):
    """VeRA for 8-bit quantized layers."""
    def __init__(
        self,
        base_layer: torch.nn.Module,
        adapter_name: str,
        vera_A,
        vera_B,
        r: int = 0,
        vera_dropout: float = 0.0,
        d_initial: float = 0.1,
        **kwargs,
    ) -> None:
        """Initialize VeRA for 8-bit layer."""

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """Compute (lambda_b * B) @ (lambda_d * A)."""

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Apply VeRA adaptation to quantized layer."""

class Linear4bit(torch.nn.Module, VeraLayer):
    """VeRA for 4-bit quantized layers."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.vera.bnb import Linear8bitLt, Linear4bit
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
| vera_A || BufferDict || Yes || Shared frozen A matrices
|-
| vera_B || BufferDict || Yes || Shared frozen B matrices
|-
| r || int || Yes || Rank of shared matrices
|-
| d_initial || float || No || Initial value for lambda_d
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || torch.Tensor || Base output + VeRA adaptation
|-
| get_delta_weight() || torch.Tensor || (lambda_b * B) @ (lambda_d * A)
|}

== Usage Examples ==

=== VeRA with 4-bit Quantization ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import VeraConfig, get_peft_model

# Load model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
)

# Apply VeRA - extremely parameter efficient
config = VeraConfig(
    r=256,                      # Shared rank
    target_modules=["q_proj", "v_proj"],
    d_initial=0.1,
)

model = get_peft_model(model, config)
# Automatically uses Linear4bit class
</syntaxhighlight>

=== VeRA with 8-bit Quantization ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import VeraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
)

config = VeraConfig(
    r=128,
    target_modules=["q_proj", "v_proj"],
    vera_dropout=0.1,
)

model = get_peft_model(model, config)
</syntaxhighlight>

=== VeRA Forward Computation ===
<syntaxhighlight lang="python">
# VeRA forward on quantized layers:
# 1. Slice shared matrices to layer dimensions
# 2. result = base_layer(x)
# 3. adapter_output = lambda_b * linear(lambda_d * linear(x, sliced_A), sliced_B)
# 4. return result + adapter_output

# Only lambda_b and lambda_d are trained per layer
# A and B matrices are shared and frozen
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Quantization_Environment]]
