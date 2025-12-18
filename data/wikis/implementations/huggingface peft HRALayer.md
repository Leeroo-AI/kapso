{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Householder_Adaptation]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Householder Reflection Adaptation layer that applies orthogonal transformations using Householder reflections with optional Gram-Schmidt orthogonalization.

=== Description ===

HRALayer implements adaptation using Householder reflections, which are a numerically stable way to parameterize orthogonal transformations. The layer learns a set of vectors that define Householder reflections, and the product of these reflections forms an orthogonal matrix that transforms the base weights. Optionally, Gram-Schmidt orthogonalization can be applied to ensure the reflection vectors remain orthogonal during training.

=== Usage ===

Use HRA when you want orthogonal fine-tuning with Householder parameterization. This method is particularly stable numerically and guarantees orthogonality. Enable Gram-Schmidt (apply_GS=True) when you want the reflection vectors to be explicitly orthogonalized, which can improve stability but adds computational cost.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/hra/layer.py src/peft/tuners/hra/layer.py]
* '''Lines:''' 1-462

=== Signature ===
<syntaxhighlight lang="python">
class HRALayer(BaseTunerLayer):
    """
    Householder Reflection Adaptation layer.

    Attributes:
        hra_u: Householder vectors (nn.ParameterDict)
        hra_r: Number of reflections per adapter (dict)
        hra_apply_GS: Whether to apply Gram-Schmidt (dict)
    """
    adapter_layer_names = ("hra_u",)
    other_param_names = ("hra_r", "hra_apply_GS")

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        apply_GS: bool,
        init_weights: bool,
        inference_mode: bool = False,
        **kwargs,
    ) -> None:
        """Create HRA adapter with r Householder reflections."""

    def get_delta_weight(self, adapter_name: str, reverse: bool = False) -> torch.Tensor:
        """Compute orthogonal transformation from Householder vectors."""

class HRALinear(nn.Module, HRALayer):
    """HRA implemented in a dense layer."""
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        apply_GS: bool = False,
        init_weights: bool = True,
        **kwargs,
    ) -> None:
        """Initialize HRA Linear layer."""

class HRAConv2d(nn.Module, HRALayer):
    """HRA implemented in Conv2d layer."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.hra import HRALayer, HRAConfig, HRAModel
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| base_layer || nn.Module || Yes || The pretrained layer to adapt (Linear or Conv2d)
|-
| adapter_name || str || Yes || Name identifier for the adapter
|-
| r || int || Yes || Number of Householder reflections
|-
| apply_GS || bool || No || Apply Gram-Schmidt orthogonalization (default: False)
|-
| init_weights || bool || No || Use symmetric initialization if r is even (default: True)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || torch.Tensor || Input transformed by orthogonal Householder transformation
|-
| get_delta_weight() || torch.Tensor || Orthogonal matrix from product of Householder reflections
|}

== Usage Examples ==

=== Basic HRA Configuration ===
<syntaxhighlight lang="python">
from peft import HRAConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure HRA
config = HRAConfig(
    r=8,                   # Number of Householder reflections
    target_modules=["q_proj", "v_proj"],
    apply_GS=False,        # No Gram-Schmidt
    init_weights=True,     # Symmetric initialization
)

# Create PEFT model
model = get_peft_model(model, config)
</syntaxhighlight>

=== HRA with Gram-Schmidt ===
<syntaxhighlight lang="python">
from peft import HRAConfig, get_peft_model

# Use Gram-Schmidt for guaranteed orthogonality
config = HRAConfig(
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    apply_GS=True,         # Apply Gram-Schmidt orthogonalization
)

model = get_peft_model(model, config)
# More stable but computationally more expensive
</syntaxhighlight>

=== HRA for Vision Models ===
<syntaxhighlight lang="python">
from peft import HRAConfig, get_peft_model
from transformers import AutoModelForImageClassification

# HRA works well for vision models via Conv2d support
model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224"
)

config = HRAConfig(
    r=4,                   # Fewer reflections for efficiency
    target_modules=["query", "value"],
)

model = get_peft_model(model, config)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
