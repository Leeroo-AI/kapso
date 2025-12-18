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

Quantized RandLoRA layer implementations for 8-bit and 4-bit bitsandbytes linear layers, enabling random projection adaptation on quantized models.

=== Description ===

Linear8bitLt and Linear4bit implement RandLoRA for bitsandbytes quantized layers. The layers use the same shared random projection matrices (randlora_A, randlora_B) with per-layer trainable lambda and gamma vectors. During forward pass, the scaled projection is applied to inputs before the quantized linear operation. Merging dequantizes weights, adds the delta, and requantizes.

=== Usage ===

Use RandLoRA quantized layers when fine-tuning quantized models (load_in_8bit or load_in_4bit). Layers are automatically dispatched when base layers are bitsandbytes Linear8bitLt or Linear4bit. The method combines the parameter efficiency of RandLoRA with memory savings of quantization.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/randlora/bnb.py src/peft/tuners/randlora/bnb.py]
* '''Lines:''' 1-457

=== Signature ===
<syntaxhighlight lang="python">
class Linear8bitLt(torch.nn.Module, RandLoraLayer):
    """RandLoRA for 8-bit quantized layers."""
    def __init__(
        self,
        base_layer: torch.nn.Module,
        adapter_name: str,
        randlora_A,
        randlora_B,
        r: int = 0,
        randlora_alpha: int = 0,
        randlora_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        """Initialize RandLoRA for 8-bit layer."""

    def get_scaled_bases(self, adapter, device=None):
        """Get scaled A and B with lambda/gamma applied."""

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Apply RandLoRA then 8-bit linear."""

class Linear4bit(torch.nn.Module, RandLoraLayer):
    """RandLoRA for 4-bit quantized layers."""

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None):
        """Merge RandLoRA into dequantized weights."""

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Apply RandLoRA then 4-bit linear."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.randlora.bnb import Linear8bitLt, Linear4bit
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
| randlora_A || BufferDict || Yes || Shared frozen A matrices
|-
| randlora_B || BufferDict || Yes || Shared frozen B matrices
|-
| r || int || Yes || Rank for adaptation
|-
| randlora_alpha || int || No || Scaling factor
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || torch.Tensor || Base quantized output + RandLoRA delta
|-
| get_delta_weight() || torch.Tensor || Scaled random projection delta
|}

== Usage Examples ==

=== RandLoRA with 4-bit Quantization ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import RandLoraConfig, get_peft_model

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

# Apply RandLoRA
config = RandLoraConfig(
    r=32,
    randlora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    projection_prng_key=42,
)

model = get_peft_model(model, config)
# Automatically uses Linear4bit class
</syntaxhighlight>

=== RandLoRA with 8-bit Quantization ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import RandLoraConfig, get_peft_model

# Load model in 8-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
)

config = RandLoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, config)
# Automatically uses Linear8bitLt class
</syntaxhighlight>

=== Forward Computation ===
<syntaxhighlight lang="python">
# RandLoRA forward on quantized layers:
# 1. Compute scaled bases: update_A, update_B from lambda * A * gamma
# 2. Apply dropout
# 3. result = base_layer(x) + linear(linear(x, update_B), update_A) * scaling

# Merging (with rounding errors warning):
model.merge_adapter()
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Quantization_Environment]]
