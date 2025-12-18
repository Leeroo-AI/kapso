{{Implementation
|domain=NLP,PEFT,Parameter-Efficient Fine-Tuning,Layer Normalization Tuning,Neural Networks
|link=https://github.com/huggingface/peft
}}

== Overview ==

=== Description ===

The <code>LNTuningLayer</code> class is a PyTorch module that implements the Layer Normalization tuning adapter layer. It wraps a base layer from the model and creates deep copies of it for each adapter, allowing for efficient adapter switching and merging without modifying the original layer weights.

LNTuningLayer inherits from both <code>nn.Module</code> and <code>BaseTunerLayer</code>, providing the core functionality for managing multiple adapters, enabling/disabling adapters, and handling forward passes with adapter-specific logic.

Key features include:
* Deep copy mechanism for preserving original layer weights
* Support for adapter merging and unmerging (swapping base layer with adapter layer)
* Single active adapter constraint (only one adapter can be active at inference time)
* Adapter enable/disable functionality with gradient control

=== Usage ===

LNTuningLayer is typically not instantiated directly by users but is created internally by the LNTuningModel when wrapping target modules. It manages the lifecycle of adapter layers and handles the forward pass routing based on whether adapters are enabled, disabled, or merged.

== Code Reference ==

=== Source Location ===
* '''Repository:''' huggingface/peft
* '''File Path:''' <code>src/peft/tuners/ln_tuning/layer.py</code>
* '''Lines:''' 25-124

=== Signature ===
<syntaxhighlight lang="python">
class LNTuningLayer(nn.Module, BaseTunerLayer):
    def __init__(self, base_layer: nn.Module, adapter_name: str)

    def update_layer(self, layer: nn.Module, adapter_name: str, inference_mode: bool = False, **kwargs)

    def enable_adapters(self, enabled: bool) -> None

    def merge(self, adapter_names: Optional[list[str]] = None, safe_merge: bool = False)

    def unmerge(self)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.ln_tuning.layer import LNTuningLayer
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===

'''Constructor Parameters:'''
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| <code>base_layer</code> || <code>nn.Module</code> || Required || The original layer to wrap with LN tuning adapter
|-
| <code>adapter_name</code> || <code>str</code> || Required || Name identifier for the adapter
|}

'''Method: update_layer'''
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| <code>layer</code> || <code>nn.Module</code> || Required || The layer to add as a new adapter
|-
| <code>adapter_name</code> || <code>str</code> || Required || Name identifier for the new adapter
|-
| <code>inference_mode</code> || <code>bool</code> || <code>False</code> || Whether to set adapter in inference mode
|}

'''Method: enable_adapters'''
{| class="wikitable"
! Parameter !! Type !! Description
|-
| <code>enabled</code> || <code>bool</code> || True to enable adapters, False to disable adapters
|}

'''Method: merge'''
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| <code>adapter_names</code> || <code>Optional[list[str]]</code> || <code>None</code> || List of adapter names to merge (only one allowed)
|-
| <code>safe_merge</code> || <code>bool</code> || <code>False</code> || Safe merge flag (not used in LN tuning)
|}

'''Method: forward'''
{| class="wikitable"
! Parameter !! Type !! Description
|-
| <code>x</code> || <code>torch.Tensor</code> || Input tensor to the layer
|-
| <code>*args</code> || <code>Any</code> || Additional positional arguments passed to the layer
|-
| <code>**kwargs</code> || <code>Any</code> || Additional keyword arguments passed to the layer
|}

=== Outputs ===

'''Attributes:'''
{| class="wikitable"
! Attribute !! Type !! Description
|-
| <code>base_layer</code> || <code>nn.Module</code> || The wrapped base layer
|-
| <code>ln_tuning_layers</code> || <code>nn.ModuleDict</code> || Dictionary of adapter layers indexed by adapter name
|-
| <code>merged_adapters</code> || <code>list</code> || List of currently merged adapter names
|-
| <code>in_features</code> || <code>int</code> || Input feature dimension of the base layer
|-
| <code>out_features</code> || <code>int</code> || Output feature dimension of the base layer
|}

'''Method Returns:'''
{| class="wikitable"
! Method !! Return Type !! Description
|-
| <code>forward</code> || <code>torch.Tensor</code> || Output tensor from the layer (base or adapter)
|-
| <code>__repr__</code> || <code>str</code> || String representation prefixed with "ln_tuning."
|}

== Usage Examples ==

=== Example 1: Basic Layer Wrapping (Internal) ===
<syntaxhighlight lang="python">
import torch.nn as nn
from peft.tuners.ln_tuning.layer import LNTuningLayer

# Typically done internally by LNTuningModel
base_layer = nn.LayerNorm(768)
ln_layer = LNTuningLayer(base_layer, adapter_name="default")

# Forward pass with adapter
output = ln_layer(input_tensor)
</syntaxhighlight>

=== Example 2: Adapter Merging and Unmerging ===
<syntaxhighlight lang="python">
# Merge adapter (swap base layer with adapter layer)
ln_layer.merge(adapter_names=["default"])
print(ln_layer.merged)  # True

# Forward pass uses merged adapter as base layer
output = ln_layer(input_tensor)

# Unmerge adapter (restore original base layer)
ln_layer.unmerge()
print(ln_layer.merged)  # False
</syntaxhighlight>

=== Example 3: Enabling and Disabling Adapters ===
<syntaxhighlight lang="python">
# Enable adapters (default state)
ln_layer.enable_adapters(enabled=True)

# Forward pass with adapter
output_with_adapter = ln_layer(input_tensor)

# Disable adapters (use base layer only)
ln_layer.enable_adapters(enabled=False)

# Forward pass without adapter
output_without_adapter = ln_layer(input_tensor)
</syntaxhighlight>

=== Example 4: Multi-Adapter Management ===
<syntaxhighlight lang="python">
import torch.nn as nn
from copy import deepcopy

# Add second adapter
second_layer = deepcopy(base_layer)
ln_layer.update_layer(second_layer, adapter_name="adapter2")

# Set active adapter
ln_layer.set_adapter("adapter2")

# Forward pass uses "adapter2"
output = ln_layer(input_tensor)

# Note: Only one adapter can be active at inference time
# Attempting to use multiple active adapters will raise ValueError
</syntaxhighlight>

== Implementation Details ==

=== Adapter Merging Mechanism ===
LN Tuning uses a unique merging approach where the base layer and adapter layer are swapped:
* '''Merge:''' <code>base_layer</code> and <code>ln_tuning_layers[adapter_name]</code> swap positions
* '''Unmerge:''' They swap back to original positions
* This allows seamless switching without weight modifications

=== Constraints ===
* '''Single Adapter Limitation:''' Only one adapter can be merged at a time
* '''Single Active Adapter:''' Only one adapter can be active during inference
* These constraints ensure deterministic behavior and prevent conflicts

=== Forward Pass Logic ===
<syntaxhighlight lang="python">
if disable_adapters:
    if merged: unmerge()
    return base_layer(x)
elif merged or no_active_adapters:
    return base_layer(x)
else:
    return active_adapter_layer(x)
</syntaxhighlight>

== Related Pages ==

* [[huggingface_peft_LNTuningConfig|LNTuningConfig]] - Configuration for LN Tuning
* [[huggingface_peft_LNTuningModel|LNTuningModel]] - Model class that creates LNTuningLayer instances
* [[huggingface_peft_BaseTunerLayer|BaseTunerLayer]] - Base class for tuner layers
* [[huggingface_peft_LoraLayer|LoraLayer]] - Alternative adapter layer implementation
* [[huggingface_peft_AdapterLayer|AdapterLayer]] - Another adapter layer implementation

[[Category:PEFT]]
[[Category:Layer]]
[[Category:Layer Normalization]]
[[Category:Adapter]]
[[Category:PyTorch]]
