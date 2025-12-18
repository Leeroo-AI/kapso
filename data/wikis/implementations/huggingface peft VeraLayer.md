{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|VeRA|https://arxiv.org/abs/2310.11454]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Vector_Adaptation]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Vector-based Random matrix Adaptation layer that uses shared frozen random matrices with per-layer trainable scaling vectors for extreme parameter efficiency.

=== Description ===

VeraLayer implements VeRA, which shares a single pair of random matrices (A and B) across all adapted layers. Only two small vectors per layer are trained: lambda_d (scales the rank dimension) and lambda_b (scales the output dimension). The forward pass computes: lambda_b * (B @ (lambda_d * (A @ x))). This achieves 10x fewer parameters than LoRA while maintaining similar performance, as the random matrices capture sufficient expressivity.

=== Usage ===

Use VeRA when you need extreme parameter efficiency, especially for deploying many adapters. Since the A/B matrices are shared and frozen, VeRA adapters are much smaller than LoRA. VeRA works best when the shared matrices are initialized with sufficient rank to cover all target layers' dimensions.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/vera/layer.py src/peft/tuners/vera/layer.py]
* '''Lines:''' 1-292

=== Signature ===
<syntaxhighlight lang="python">
class VeraLayer(BaseTunerLayer):
    """
    Vector-based Random matrix Adaptation layer.

    Attributes:
        vera_lambda_b: ParameterDict of output scaling vectors
        vera_lambda_d: ParameterDict of rank scaling vectors
        vera_A: BufferDict reference to shared frozen A matrices
        vera_B: BufferDict reference to shared frozen B matrices
        vera_dropout: ModuleDict of dropout layers
    """
    adapter_layer_names = ("vera_lambda_b", "vera_lambda_d")
    other_param_names = ("vera_A", "vera_B")

    def update_layer(
        self,
        adapter_name,
        vera_A: BufferDict,
        vera_B: BufferDict,
        r,
        vera_dropout,
        init_weights,
        d_initial: float = 0.1,
        inference_mode: bool = False,
        **kwargs,
    ):
        """Create VeRA scaling vectors."""

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """Compute delta weight: (lambda_b * B) @ (lambda_d * A)."""

class Linear(nn.Linear, VeraLayer):
    """VeRA implemented in Linear layer."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.vera import VeraLayer, VeraConfig, VeraModel
from peft import VeraConfig, get_peft_model
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| base_layer || nn.Module || Yes || The pretrained Linear layer
|-
| adapter_name || str || Yes || Name for the adapter
|-
| vera_A || BufferDict || Yes || Shared frozen A matrices
|-
| vera_B || BufferDict || Yes || Shared frozen B matrices
|-
| r || int || Yes || Rank of the shared matrices
|-
| vera_dropout || float || No || Dropout probability
|-
| d_initial || float || No || Initial value for lambda_d (default: 0.1)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || torch.Tensor || Base output + scaled VeRA adaptation
|-
| get_delta_weight() || torch.Tensor || (lambda_b * B) @ (lambda_d * A)
|}

== Usage Examples ==

=== Basic VeRA Configuration ===
<syntaxhighlight lang="python">
from peft import VeraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# VeRA shares random matrices across layers
config = VeraConfig(
    r=256,                    # Shared rank (can be larger since matrices are shared)
    target_modules=["q_proj", "v_proj"],
    vera_dropout=0.0,
    d_initial=0.1,            # Initial scaling factor
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# Much fewer parameters than LoRA with same rank
</syntaxhighlight>

=== VeRA with Dropout ===
<syntaxhighlight lang="python">
from peft import VeraConfig, get_peft_model

config = VeraConfig(
    r=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    vera_dropout=0.1,
    d_initial=0.1,
)

model = get_peft_model(model, config)
</syntaxhighlight>

=== Comparing VeRA vs LoRA Parameters ===
<syntaxhighlight lang="python">
from peft import VeraConfig, LoraConfig, get_peft_model

# LoRA parameters: 2 * r * d per layer
lora_config = LoraConfig(r=16, target_modules=["q_proj", "v_proj"])

# VeRA parameters: (in_features + rank) per layer + shared matrices
vera_config = VeraConfig(r=256, target_modules=["q_proj", "v_proj"])

# VeRA achieves similar quality with ~10x fewer trainable parameters
# because A and B matrices are shared across all layers
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
