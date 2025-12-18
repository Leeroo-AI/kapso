{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|AdaLoRA|https://openreview.net/forum?id=lq62uWRJjiY]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Adaptive_Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

SVD-based adaptive low-rank adaptation layers that dynamically prune and allocate ranks during training based on importance scores.

=== Description ===

AdaLoraLayer implements an SVD-based parameterization of LoRA that enables adaptive rank allocation during training. Unlike standard LoRA which uses fixed ranks, AdaLoRA decomposes the weight update into three learnable matrices: lora_A (right singular vectors), lora_E (singular values), and lora_B (left singular vectors). During training, a RankAllocator dynamically prunes less important singular values based on gradient-weighted importance scores, redistributing the rank budget to more important layers.

=== Usage ===

Use AdaLoRA when you need to automatically determine optimal rank allocation across layers rather than using a fixed rank. This is particularly useful when you want to minimize trainable parameters while maximizing performance, as AdaLoRA will concentrate parameters in the most important layers.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/adalora/layer.py src/peft/tuners/adalora/layer.py]
* '''Lines:''' 1-361

=== Signature ===
<syntaxhighlight lang="python">
class AdaLoraLayer(LoraLayer):
    """
    AdaLoRA layer with SVD-based parameterization.

    Attributes:
        lora_A: Right singular vectors (nn.ParameterDict)
        lora_B: Left singular vectors (nn.ParameterDict)
        lora_E: Singular values (nn.ParameterDict)
        ranknum: Current rank tracker (nn.ParameterDict)
    """
    adapter_layer_names = ("lora_A", "lora_B", "lora_E", "lora_embedding_A", "lora_embedding_B")
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout", "ranknum")

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        init_lora_weights: bool,
        inference_mode: bool = False,
        **kwargs
    ) -> None:
        """Update layer with AdaLoRA adapter."""

class SVDLinear(nn.Module, AdaLoraLayer):
    """SVD-based adaptation by a dense layer."""
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool = True,
        **kwargs,
    ) -> None:
        """Initialize SVDLinear layer."""

class RankAllocator:
    """
    Rank allocation manager for AdaLoRA.

    Dynamically adjusts rank budget based on importance scores
    computed from gradient information during training.
    """
    def __init__(self, model, peft_config, adapter_name):
        """Initialize RankAllocator."""

    def update_and_allocate(self, model, global_step, force_mask=False):
        """Update importance scores and allocate budget."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.adalora import AdaLoraLayer, SVDLinear, RankAllocator
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
| r || int || Yes || Initial rank for SVD decomposition
|-
| lora_alpha || int || No || Scaling factor (default: 1)
|-
| lora_dropout || float || No || Dropout probability (default: 0.0)
|-
| init_lora_weights || bool || No || Whether to initialize weights (default: True)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || torch.Tensor || Input transformed by base layer + SVD low-rank adaptation
|-
| delta_weight || torch.Tensor || Computed as (B @ (A * E)) * scaling / ranknum
|}

== Usage Examples ==

=== Basic AdaLoRA Configuration ===
<syntaxhighlight lang="python">
from peft import AdaLoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure AdaLoRA with adaptive rank allocation
config = AdaLoraConfig(
    init_r=12,           # Initial rank
    target_r=4,          # Target final rank after pruning
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    tinit=200,           # Warmup steps before pruning
    tfinal=1000,         # Steps before final rank
    deltaT=10,           # Pruning frequency
    orth_reg_weight=0.5, # Orthogonal regularization
)

# Create PEFT model
model = get_peft_model(model, config)
</syntaxhighlight>

=== Training with Rank Allocation ===
<syntaxhighlight lang="python">
# Training loop with AdaLoRA rank updates
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for step, batch in enumerate(dataloader):
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    optimizer.step()

    # Critical: Update rank allocation after backward, before zero_grad
    model.base_model.update_and_allocate(step)

    optimizer.zero_grad()
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
