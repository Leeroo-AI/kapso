# Implementation: huggingface_peft_AdaLoraLayer

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_peft|https://github.com/huggingface/peft]]
* [[source::Paper|AdaLoRA|https://openreview.net/forum?id=lq62uWRJjiY]]
|-
! Domains
| [[domain::NLP]], [[domain::Parameter_Efficient_Training]], [[domain::Adaptive_Rank]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

SVD-based adaptive low-rank adaptation layers that dynamically allocate rank budget during training based on importance scores.

=== Description ===

The `AdaLoraLayer` and `SVDLinear` classes implement adaptive LoRA using SVD decomposition. Unlike standard LoRA with fixed rank, AdaLoRA maintains three trainable parameter sets: lora_A (right singular vectors), lora_B (left singular vectors), and lora_E (singular values). The `RankAllocator` dynamically prunes unimportant rank dimensions during training based on gradient-weighted importance scores with uncertainty quantification.

=== Usage ===

Use these classes when you need automatic rank allocation during fine-tuning. AdaLoRA is particularly useful when you don't know the optimal rank beforehand and want the model to learn which layers need more capacity. The rank budget decreases according to a cubic schedule from `init_r` to `target_r`.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft huggingface_peft]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/adalora/layer.py src/peft/tuners/adalora/layer.py]
* '''Lines:''' 1-361

=== Signature ===
<syntaxhighlight lang="python">
class AdaLoraLayer(LoraLayer):
    """Base AdaLoRA layer with SVD decomposition."""
    adapter_layer_names = ("lora_A", "lora_B", "lora_E", "lora_embedding_A", "lora_embedding_B")
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout", "ranknum")

    def __init__(self, base_layer: nn.Module) -> None: ...
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, inference_mode=False, **kwargs): ...
    def reset_lora_parameters(self, adapter_name): ...

class SVDLinear(nn.Module, AdaLoraLayer):
    """SVD-based adaptation for dense layers."""
    def __init__(self, base_layer, adapter_name, r=0, lora_alpha=1, lora_dropout=0.0,
                 fan_in_fan_out=False, init_lora_weights=True, **kwargs) -> None: ...
    def merge(self, safe_merge=False, adapter_names=None) -> None: ...
    def unmerge(self) -> None: ...
    def get_delta_weight(self, adapter) -> torch.Tensor: ...
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor: ...

class RankAllocator:
    """Dynamic rank allocation based on importance scores."""
    def __init__(self, model, peft_config, adapter_name): ...
    def update_and_allocate(self, model, global_step, force_mask=False): ...
    def mask_to_budget(self, model, budget): ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.adalora import AdaLoraLayer, SVDLinear, RankAllocator
</syntaxhighlight>

== I/O Contract ==

=== Inputs (SVDLinear.forward) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| x || torch.Tensor || Yes || Input tensor of shape (batch, seq_len, in_features)
|-
| *args || Any || No || Additional positional arguments passed to base layer
|-
| **kwargs || Any || No || Additional keyword arguments passed to base layer
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| result || torch.Tensor || Output tensor with SVD-based LoRA adaptation applied
|}

=== RankAllocator.update_and_allocate ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| budget || int || Current rank budget after allocation
|-
| rank_pattern || dict or None || Dictionary mapping layer names to boolean rank masks
|}

== Usage Examples ==

=== Basic AdaLoRA Setup ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import get_peft_model, AdaLoraConfig

# Load base model
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")

# Configure AdaLoRA with adaptive rank
config = AdaLoraConfig(
    init_r=12,           # Initial rank per layer
    target_r=4,          # Target rank after pruning
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    # Rank allocation schedule
    tinit=200,           # Warmup steps before pruning
    tfinal=1000,         # Steps to finalize rank
    deltaT=10,           # Pruning frequency
    beta1=0.85,          # Sensitivity smoothing
    beta2=0.85,          # Uncertainty smoothing
    orth_reg_weight=0.5, # Orthogonality regularization
)

model = get_peft_model(model, config)
</syntaxhighlight>

=== Training Loop with Rank Update ===
<syntaxhighlight lang="python">
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=1e-4)

for step, batch in enumerate(dataloader):
    # Forward pass
    outputs = model(**batch)
    loss = outputs.loss

    # Backward pass
    loss.backward()
    optimizer.step()

    # CRITICAL: Update rank allocation AFTER backward, BEFORE zero_grad
    model.base_model.update_and_allocate(step)

    optimizer.zero_grad()
</syntaxhighlight>

== Related Pages ==

* [[requires_env::Environment:huggingface_peft_CUDA_Training]]
