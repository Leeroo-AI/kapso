{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|RandLoRA|https://arxiv.org/abs/2502.00987]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Random_Projection]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Random projection-based LoRA layer that uses shared random bases with per-layer trainable scaling vectors for parameter-efficient adaptation.

=== Description ===

RandLoraLayer implements adaptation using shared random projection matrices (randlora_A and randlora_B) with per-layer trainable parameters (lambda and gamma). The random bases are initialized once and shared across all layers, with lambda scaling the rank dimension and gamma scaling the base dimension. A custom autograd function (UniqueBaseGrad) efficiently computes gradients for the unique shared base. This achieves fewer parameters than LoRA while maintaining expressivity.

=== Usage ===

Use RandLoRA when you want parameter efficiency beyond LoRA by sharing random projection matrices across layers. The method is particularly effective when the projection_prng_key ensures reproducible random matrices. RandLoRA supports sparse random projections for additional efficiency.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/randlora/layer.py src/peft/tuners/randlora/layer.py]
* '''Lines:''' 1-351

=== Signature ===
<syntaxhighlight lang="python">
class UniqueBaseGrad(torch.autograd.Function):
    """Memory-efficient gradient for shared random base."""
    @staticmethod
    def forward(ctx, randlora_A, randlora_lambda, randlora_gamma):
        """Compute lambda * A * gamma."""

    @staticmethod
    def backward(ctx, grad_output):
        """Gradient for lambda and gamma only (A is frozen)."""

class RandLoraLayer(BaseTunerLayer):
    """
    Random projection LoRA layer.

    Attributes:
        randlora_lambda: ParameterDict of rank scaling vectors [r, num_bases]
        randlora_gamma: ParameterDict of base scaling vectors [num_bases, min_dim]
        randlora_A: BufferDict reference to shared A matrices
        randlora_B: BufferDict reference to shared B matrices
        num_bases: Number of bases for full-rank coverage
    """
    adapter_layer_names = ("randlora_lambda", "randlora_gamma")
    other_param_names = ("randlora_A", "randlora_B")

    def update_layer(
        self,
        adapter_name,
        randlora_A: BufferDict,
        randlora_B: BufferDict,
        r,
        randlora_alpha,
        randlora_dropout,
        init_weights,
        **kwargs,
    ):
        """Create RandLoRA scaling parameters."""

    def get_scaled_bases(self, adapter, device=None):
        """Compute scaled A and B matrices for forward."""

class Linear(nn.Linear, RandLoraLayer):
    """RandLoRA implemented in Linear layer."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.randlora import RandLoraLayer, RandLoraConfig, RandLoraModel
from peft import RandLoraConfig, get_peft_model
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
| randlora_A || BufferDict || Yes || Shared frozen A matrices
|-
| randlora_B || BufferDict || Yes || Shared frozen B matrices
|-
| r || int || Yes || Rank for adaptation
|-
| randlora_alpha || int || No || Scaling factor
|-
| randlora_dropout || float || No || Dropout probability
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || torch.Tensor || Base output + scaled random projection
|-
| get_delta_weight() || torch.Tensor || (B @ (lambda * A * gamma)) * scaling
|}

== Usage Examples ==

=== Basic RandLoRA Configuration ===
<syntaxhighlight lang="python">
from peft import RandLoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# RandLoRA shares random matrices across layers
config = RandLoraConfig(
    r=32,
    randlora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    projection_prng_key=42,   # Reproducible random init
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# Fewer parameters than LoRA with same rank
</syntaxhighlight>

=== RandLoRA with Sparse Projections ===
<syntaxhighlight lang="python">
from peft import RandLoraConfig, get_peft_model

# Use sparse random projections for efficiency
config = RandLoraConfig(
    r=64,
    randlora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj"],
    sparse=True,              # Sparse random matrices
)

model = get_peft_model(model, config)
</syntaxhighlight>

=== Very Sparse RandLoRA ===
<syntaxhighlight lang="python">
from peft import RandLoraConfig, get_peft_model

# Maximum sparsity for lowest memory
config = RandLoraConfig(
    r=64,
    target_modules=["q_proj", "v_proj"],
    very_sparse=True,         # sqrt(min_dim) sparsity
    save_projection=False,    # Don't save random matrices
)

model = get_peft_model(model, config)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
