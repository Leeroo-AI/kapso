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

Quantized OFT layer implementations for 8-bit and 4-bit bitsandbytes linear layers, enabling orthogonal fine-tuning on quantized models.

=== Description ===

Linear8bitLt and Linear4bit implement OFT (Orthogonal Fine-Tuning) for bitsandbytes quantized layers. The layers apply orthogonal transformations to inputs before the quantized linear operation. During merge, weights are dequantized, transformed, and requantized. The implementation handles the conversion between quantized and float formats for both forward pass and weight merging operations.

=== Usage ===

Use OFT quantized layers when fine-tuning quantized models (loaded with load_in_8bit or load_in_4bit). The layers are automatically dispatched via dispatch_bnb_8bit and dispatch_bnb_4bit when target layers are bitsandbytes Linear8bitLt or Linear4bit. Note that merging may introduce rounding errors due to quantization.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/oft/bnb.py src/peft/tuners/oft/bnb.py]
* '''Lines:''' 1-389

=== Signature ===
<syntaxhighlight lang="python">
class Linear8bitLt(torch.nn.Module, OFTLayer):
    """OFT for 8-bit quantized layers."""
    def __init__(
        self,
        base_layer: torch.nn.Module,
        adapter_name: str,
        r: int = 8,
        oft_block_size: int = 0,
        module_dropout: float = 0.0,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        **kwargs,
    ) -> None:
        """Initialize OFT for 8-bit layer."""

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None):
        """Merge OFT into dequantized weights, then requantize."""

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Apply OFT rotation then 8-bit linear."""

class Linear4bit(torch.nn.Module, OFTLayer):
    """OFT for 4-bit quantized layers."""

def dispatch_bnb_8bit(target, adapter_name, **kwargs):
    """Dispatch OFT for 8-bit layers."""

def dispatch_bnb_4bit(target, adapter_name, **kwargs):
    """Dispatch OFT for 4-bit layers."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.oft.bnb import Linear8bitLt, Linear4bit
from peft.tuners.oft.bnb import dispatch_bnb_8bit, dispatch_bnb_4bit
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
| r || int || Yes || Number of OFT blocks
|-
| oft_block_size || int || No || Size of each orthogonal block
|-
| coft || bool || No || Use constrained OFT
|-
| block_share || bool || No || Share rotation across blocks
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || torch.Tensor || OFT-rotated input through quantized layer
|-
| get_delta_weight() || torch.Tensor || Orthogonal rotation matrix
|}

== Usage Examples ==

=== OFT with 4-bit Quantization ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import OFTConfig, get_peft_model

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

# Apply OFT to quantized model
config = OFTConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, config)
# OFT layers automatically use Linear4bit class
</syntaxhighlight>

=== OFT with 8-bit Quantization ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import OFTConfig, get_peft_model

# Load model in 8-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
)

config = OFTConfig(
    r=4,
    target_modules=["q_proj", "v_proj"],
    coft=True,
)

model = get_peft_model(model, config)
# OFT layers automatically use Linear8bitLt class
</syntaxhighlight>

=== Merging Quantized OFT ===
<syntaxhighlight lang="python">
# Warning: Merging may introduce rounding errors
model.merge_adapter()

# For inference without adapter overhead:
# OFT rotation is applied to dequantized weights
# then requantized back to 4-bit/8-bit
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Quantization_Environment]]
