{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|IA3|https://arxiv.org/abs/2205.05638]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Activation_Scaling]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Infused Adapter by Inhibiting and Amplifying Inner Activations layer that applies learned scaling vectors to activations for ultra-efficient fine-tuning.

=== Description ===

IA3Layer implements the (IA)^3 method which introduces learned vectors that rescale key, value, and feedforward activations. Unlike LoRA which adds low-rank matrices, IA3 simply multiplies activations by a learned vector, making it extremely parameter-efficient (typically 10x fewer parameters than LoRA). The method distinguishes between feedforward layers (scale inputs) and attention layers (scale outputs).

=== Usage ===

Use IA3 when you need the most parameter-efficient adaptation method. IA3 is ideal when memory is extremely constrained or when you want to store many adapters with minimal overhead. Note that IA3's unmerge operation is approximate due to the multiplicative nature of the adaptation.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/ia3/layer.py src/peft/tuners/ia3/layer.py]
* '''Lines:''' 1-331

=== Signature ===
<syntaxhighlight lang="python">
class IA3Layer(BaseTunerLayer):
    """
    (IA)^3 layer for activation scaling.

    Attributes:
        ia3_l: Learned scaling vectors (nn.ParameterDict)
        is_feedforward: Whether this is a feedforward layer (bool)
    """
    adapter_layer_names = ("ia3_l",)

    def __init__(
        self,
        base_layer: nn.Module,
        is_feedforward: bool,
        **kwargs
    ) -> None:
        """Initialize IA3 layer."""

    def update_layer(
        self,
        adapter_name: str,
        init_ia3_weights: bool,
        inference_mode: bool = False,
        **kwargs
    ):
        """Create IA3 scaling vector for adapter."""

class Linear(nn.Module, IA3Layer):
    """IA3 implemented in a dense layer."""
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        fan_in_fan_out: bool = False,
        is_feedforward: bool = False,
        is_target_conv_1d_layer: bool = False,
        init_ia3_weights: bool = True,
        **kwargs,
    ) -> None:
        """Initialize IA3 Linear layer."""

class Conv2d(IA3Layer):
    """IA3 implemented in a 2D convolutional layer."""

class Conv3d(IA3Layer):
    """IA3 implemented in a 3D convolutional layer."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.ia3 import IA3Layer, IA3Config, IA3Model
from peft import IA3Config, get_peft_model
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| base_layer || nn.Module || Yes || The pretrained layer to adapt
|-
| adapter_name || str || Yes || Name identifier for the adapter
|-
| is_feedforward || bool || Yes || True for FFN layers (scale input), False for attention (scale output)
|-
| fan_in_fan_out || bool || No || True if layer stores weights as (fan_in, fan_out)
|-
| init_ia3_weights || bool || No || Initialize vectors to ones (default: True)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || torch.Tensor || Activation scaled by learned vector
|-
| ia3_l || nn.Parameter || Shape (1, in_features) for FFN or (out_features, 1) for attention
|}

== Usage Examples ==

=== Basic IA3 Configuration ===
<syntaxhighlight lang="python">
from peft import IA3Config, get_peft_model
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure IA3
config = IA3Config(
    target_modules=["k_proj", "v_proj", "down_proj"],
    feedforward_modules=["down_proj"],  # Specify feedforward layers
    init_ia3_weights=True,
)

# Create PEFT model
model = get_peft_model(model, config)
print(f"Trainable parameters: {model.print_trainable_parameters()}")
# IA3 typically has ~10x fewer parameters than LoRA
</syntaxhighlight>

=== IA3 for T5 ===
<syntaxhighlight lang="python">
from peft import IA3Config, get_peft_model
from transformers import AutoModelForSeq2SeqLM

# IA3 was originally designed for T5-style models
model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")

config = IA3Config(
    target_modules=["k", "v", "wo"],
    feedforward_modules=["wo"],
)

model = get_peft_model(model, config)
</syntaxhighlight>

=== Merging IA3 (Approximate) ===
<syntaxhighlight lang="python">
# Note: IA3 unmerge is approximate due to multiplicative scaling
model.merge_adapter()

# For exact reproduction, keep adapters separate
# IA3 is best used without merging
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
