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

Quantized RoAD layer implementations for 8-bit and 4-bit bitsandbytes linear layers, enabling rotation adaptation on quantized models.

=== Description ===

Linear8bitLt and Linear4bit implement RoAD for bitsandbytes quantized layers. The rotational transformation (theta and alpha parameters) is applied to the layer output. During merge, the full rotation matrix R is constructed and applied to dequantized weights as R @ W, then requantized. For unmerge, the inverse rotation R^-1 is computed and applied.

=== Usage ===

Use RoAD quantized layers when fine-tuning quantized models with load_in_8bit or load_in_4bit. Layers are automatically dispatched when base layers are bitsandbytes Linear8bitLt or Linear4bit. Merging involves matrix inversion for unmerge which may introduce numerical errors.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/road/bnb.py src/peft/tuners/road/bnb.py]
* '''Lines:''' 1-408

=== Signature ===
<syntaxhighlight lang="python">
class Linear8bitLt(torch.nn.Module, RoadLayer):
    """RoAD for 8-bit quantized layers."""
    def __init__(
        self,
        base_layer: torch.nn.Module,
        adapter_name: str,
        variant: RoadVariant = "road_1",
        group_size: int = 64,
        init_weights: bool = True,
        **kwargs,
    ) -> None:
        """Initialize RoAD for 8-bit layer."""

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None):
        """Merge rotation into dequantized weights."""

    def unmerge(self) -> None:
        """Apply inverse rotation to unmerge."""

class Linear4bit(torch.nn.Module, RoadLayer):
    """RoAD for 4-bit quantized layers."""

def dispatch_bnb_8bit(target, adapter_name, **kwargs):
    """Dispatch RoAD for 8-bit layers."""

def dispatch_bnb_4bit(target, adapter_name, **kwargs):
    """Dispatch RoAD for 4-bit layers."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.road.bnb import Linear8bitLt, Linear4bit
from peft.tuners.road.bnb import dispatch_bnb_8bit, dispatch_bnb_4bit
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
| variant || str || Yes || "road_1", "road_2", or "road_4"
|-
| group_size || int || Yes || Size of rotation groups
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || torch.Tensor || Rotated output from quantized layer
|-
| _get_delta_weight() || torch.Tensor || Rotation matrix R for merging
|}

== Usage Examples ==

=== RoAD with 4-bit Quantization ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import RoadConfig, get_peft_model

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

# Apply RoAD
config = RoadConfig(
    variant="road_1",
    group_size=64,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, config)
# Automatically uses Linear4bit class
</syntaxhighlight>

=== RoAD with 8-bit Quantization ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import RoadConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
)

config = RoadConfig(
    variant="road_2",
    group_size=64,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, config)
</syntaxhighlight>

=== Merge/Unmerge with Quantization ===
<syntaxhighlight lang="python">
# Merging applies R @ W to dequantized weights
model.merge_adapter()

# Unmerging applies R^-1 @ W
# Warning: Matrix inversion may introduce numerical errors
model.unmerge_adapter()
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Quantization_Environment]]
