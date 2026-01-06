= ShiraLayer =

== Knowledge Sources ==
* [https://github.com/huggingface/peft HuggingFace PEFT Repository]
* Source: src/peft/tuners/shira/layer.py

== Domains ==
* [[Natural Language Processing (NLP)]]
* [[Parameter-Efficient Fine-Tuning (PEFT)]]
* [[Neural Network Layers]]
* [[Sparse Matrices]]
* [[Low-Rank Adaptation]]

== Overview ==

=== Description ===
ShiraLayer is the layer implementation for Sparse High Rank Adapter (SHiRA), which applies sparse high-rank adaptation to neural network layers. The layer stores sparse weight updates using a sparse COO tensor representation, where trainable parameters are stored as a flat vector with corresponding indices. This approach enables parameter-efficient fine-tuning with the same parameter count as LoRA but maintains high-rank adaptation through sparsity.

The implementation includes a base ShiraLayer class that manages adapter parameters and a Linear class that implements the SHiRA adapter for dense linear layers.

=== Usage ===
ShiraLayer is automatically instantiated when applying ShiraConfig to a model. It wraps base layers (currently only nn.Linear is supported) and adds sparse trainable parameters. The layer supports merging/unmerging adapters with base weights and can handle multiple adapters simultaneously.

== Code Reference ==

=== Source Location ===
File: src/peft/tuners/shira/layer.py
Lines: 26-218

=== Class Signatures ===
<syntaxhighlight lang="python">
class ShiraLayer(BaseTunerLayer):
    """Base class for SHiRA layers"""
    adapter_layer_names = ("shira_weight",)
    other_param_names = ("r", "scaling", "shira_indices")

class Linear(nn.Module, ShiraLayer):
    """SHiRA implementation for dense linear layers"""
    def __init__(
        self,
        base_layer,
        mask,
        adapter_name: str,
        r: int = 0,
        fan_in_fan_out: bool = False,
        init_weights: bool = True,
        **kwargs
    ) -> None
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from peft.tuners.shira.layer import ShiraLayer, Linear
</syntaxhighlight>

== I/O Contract ==

=== ShiraLayer Initialization Parameters ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| base_layer || nn.Module || The base layer to wrap with SHiRA adapter
|-
| kwargs || dict || Additional keyword arguments
|}

=== Linear Layer Initialization Parameters ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| base_layer || nn.Module || (required) || The base linear layer to adapt
|-
| mask || torch.Tensor || (required) || Binary mask defining sparse pattern
|-
| adapter_name || str || (required) || Name of the adapter
|-
| r || int || 0 || Rank parameter for SHiRA
|-
| fan_in_fan_out || bool || False || True if weights stored as (fan_in, fan_out)
|-
| init_weights || bool || True || Initialize weights to zeros
|}

=== Attributes ===
{| class="wikitable"
! Attribute !! Type !! Description
|-
| shira_weight || nn.ParameterDict || Dictionary of trainable sparse weight vectors per adapter
|-
| shira_indices || dict || Dictionary of indices for sparse COO tensor per adapter
|-
| r || dict || Dictionary of rank values per adapter
|-
| scaling || dict || Dictionary of scaling factors per adapter
|-
| merged_adapters || list || List of currently merged adapter names
|}

=== Key Methods ===
{| class="wikitable"
! Method !! Parameters !! Returns !! Description
|-
| update_layer || adapter_name, mask, r, init_weights, inference_mode || None || Update layer with new adapter
|-
| merge || safe_merge, adapter_names || None || Merge adapter weights into base weights
|-
| unmerge || None || None || Unmerge adapter weights from base weights
|-
| get_delta_weight || adapter || torch.Tensor || Compute delta weight for given adapter
|-
| forward || x, *args, **kwargs || torch.Tensor || Forward pass with adapter
|-
| set_scale || adapter, scale || None || Set scaling factor for adapter
|}

== Usage Examples ==

=== Forward Pass with SHiRA Layer ===
<syntaxhighlight lang="python">
import torch
from peft import ShiraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Create model with SHiRA
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
config = ShiraConfig(r=32, target_modules=["q_proj"])
model = get_peft_model(base_model, config)

# Forward pass
input_ids = torch.randint(0, 1000, (2, 10))
outputs = model(input_ids)
</syntaxhighlight>

=== Merging and Unmerging Adapters ===
<syntaxhighlight lang="python">
# Access a SHiRA layer
shira_layer = model.base_model.model.decoder.layers[0].self_attn.q_proj

# Merge adapter into base weights
shira_layer.merge(safe_merge=True)

# Check if merged
print(f"Is merged: {shira_layer.merged}")

# Unmerge adapter
shira_layer.unmerge()
</syntaxhighlight>

=== Setting Custom Scaling ===
<syntaxhighlight lang="python">
# Set custom scaling for an adapter
shira_layer.set_scale("default", scale=2.0)

# Forward pass with scaled adapter
output = shira_layer(input_tensor)
</syntaxhighlight>

=== Inspecting Sparse Pattern ===
<syntaxhighlight lang="python">
# Get delta weight as sparse COO tensor
delta = shira_layer.get_delta_weight("default")

# Inspect sparsity
num_nonzero = delta._nnz()
total_params = delta.shape[0] * delta.shape[1]
sparsity = 1.0 - (num_nonzero / total_params)
print(f"Sparsity: {sparsity:.2%}")
</syntaxhighlight>

=== Disabling Adapters Temporarily ===
<syntaxhighlight lang="python">
# Disable adapters for inference
with model.disable_adapter():
    output = model(input_ids)  # Uses only base model weights
</syntaxhighlight>

=== Accessing Layer Parameters ===
<syntaxhighlight lang="python">
# Access SHiRA parameters
for name, param in shira_layer.named_parameters():
    if "shira_weight" in name:
        print(f"Adapter: {name}, Shape: {param.shape}, Trainable: {param.requires_grad}")
</syntaxhighlight>

== Related Pages ==
* [[huggingface_peft_ShiraConfig|ShiraConfig]] - Configuration class for SHiRA
* [[huggingface_peft_ShiraModel|ShiraModel]] - Model class for SHiRA
* [[huggingface_peft_LoraLayer|LoraLayer]] - Layer implementation for LoRA
* [[Neural Network Layers]]
* [[Sparse Matrices]]

[[Category:Machine Learning]]
[[Category:PEFT]]
[[Category:Neural Network Layers]]
[[Category:HuggingFace]]
