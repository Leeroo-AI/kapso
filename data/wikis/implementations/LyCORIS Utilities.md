= LyCORIS Utilities =

== Knowledge Sources ==
* '''Repository:''' [https://github.com/huggingface/peft HuggingFace PEFT]
* '''Source File:''' src/peft/tuners/lycoris_utils.py

== Domains ==
* [[Natural Language Processing]]
* [[Parameter-Efficient Fine-Tuning]]
* [[Low-Rank Adaptation]]
* [[Model Tuning]]

== Overview ==

=== Description ===
The LyCORIS (Lora beYond Conventional methods, Other Rank adaptation Implementations for Stable diffusion) utilities module provides base classes and utilities for implementing LyCORIS-style adapters in PEFT. LyCORIS extends the LoRA concept with additional techniques for parameter-efficient fine-tuning.

This module contains foundational components for LyCORIS-style adapters:
* '''LycorisConfig''': Configuration dataclass for LyCORIS adapters with rank and alpha patterns
* '''LycorisLayer''': Base layer class for LyCORIS-style adapter implementations
* '''LycorisTuner''': Base tuner class for creating and managing LyCORIS adapters

Key features:
* Pattern-based rank and alpha configuration for fine-grained control
* Support for rank and module dropout
* Merge and unmerge capabilities
* Scale operations for adapter weights
* Safe merge with NaN detection
* Abstract interface for implementing custom LyCORIS variants

The module provides a flexible framework that can be extended to implement various LyCORIS methods such as LoHa, LoKr, and other adaptations.

=== Usage ===
This module is primarily used as a base for implementing specific LyCORIS adapter types. It is not typically instantiated directly but rather extended by concrete implementations like LoHa or LoKr.

== Code Reference ==

=== Source Location ===
<code>src/peft/tuners/lycoris_utils.py</code>

=== LycorisConfig Signature ===
<syntaxhighlight lang="python">
@dataclass
class LycorisConfig(PeftConfig):
    rank_pattern: Optional[dict] = field(default_factory=dict)
    alpha_pattern: Optional[dict] = field(default_factory=dict)
</syntaxhighlight>

=== LycorisLayer Signature ===
<syntaxhighlight lang="python">
class LycorisLayer(BaseTunerLayer):
    other_param_names = ("r", "alpha", "scaling", "rank_dropout", "module_dropout")

    def __init__(self, base_layer: nn.Module) -> None
</syntaxhighlight>

=== LycorisTuner Signature ===
<syntaxhighlight lang="python">
class LycorisTuner(BaseTuner):
    prefix: str
    tuner_layer_cls = LycorisLayer
    layers_mapping: dict[type[torch.nn.Module], type[LycorisLayer]]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.lycoris_utils import LycorisConfig, LycorisLayer, LycorisTuner
</syntaxhighlight>

== I/O Contract ==

=== LycorisConfig Fields ===
{| class="wikitable"
! Field !! Type !! Default !! Description
|-
| rank_pattern || Optional[dict] || {} || Mapping from layer names/regex to ranks different from default
|-
| alpha_pattern || Optional[dict] || {} || Mapping from layer names/regex to alphas different from default
|}

=== LycorisLayer Constructor ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| base_layer || nn.Module || The base layer to wrap with LyCORIS adapter
|}

=== LycorisLayer Key Methods ===

==== merge ====
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| safe_merge || bool || False || Check for NaNs before merging
|-
| adapter_names || Optional[list[str]] || None || Adapters to merge; None merges all active
|}

==== unmerge ====
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| (none) || - || - || Unmerges all merged adapters
|}

==== set_scale ====
{| class="wikitable"
! Parameter !! Type !! Description
|-
| adapter || str || Name of the adapter
|-
| scale || float || Scale factor to apply
|}

==== scale_layer ====
{| class="wikitable"
! Parameter !! Type !! Description
|-
| scale || float || Scale factor to multiply existing scaling
|}

==== unscale_layer ====
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| scale || Optional[float] || None || Scale to divide by; None resets to default
|}

=== LycorisTuner Methods ===

==== _create_new_module (classmethod) ====
{| class="wikitable"
! Parameter !! Type !! Description
|-
| config || LycorisConfig || Configuration for the adapter
|-
| adapter_name || str || Name of the adapter
|-
| target || nn.Module || Target module to adapt
|-
| **kwargs || Any || Additional arguments
|}

{| class="wikitable"
! Return !! Type !! Description
|-
| new_module || LycorisLayer || New LyCORIS layer wrapping the target
|}

=== Abstract Methods (Must be Implemented by Subclasses) ===

==== LycorisLayer Abstract Methods ====
* <code>_available_adapters</code> (property): Returns set of available adapter names
* <code>create_adapter_parameters(adapter_name, r, **kwargs)</code>: Creates adapter parameter tensors
* <code>_get_delta_activations(adapter_name, x, *args, **kwargs)</code>: Computes activations to add to base output
* <code>get_delta_weight(adapter_name)</code>: Computes weight delta for merging
* <code>reset_adapter_parameters(adapter_name)</code>: Resets adapter parameters
* <code>update_layer(adapter_name, r, alpha, **kwargs)</code>: Updates or creates adapter in layer

==== LycorisTuner Abstract Methods ====
* <code>_create_and_replace(config, adapter_name, target, target_name, parent, current_key)</code>: Creates and replaces module with LyCORIS version

== Usage Examples ==

=== Defining Custom LyCORIS Config ===
<syntaxhighlight lang="python">
from peft.tuners.lycoris_utils import LycorisConfig
from peft.utils import PeftType

@dataclass
class CustomLycorisConfig(LycorisConfig):
    """Custom LyCORIS configuration"""
    custom_param: float = field(default=1.0, metadata={"help": "Custom parameter"})

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.CUSTOM_LYCORIS

# Use with pattern-based configuration
config = CustomLycorisConfig(
    r=8,
    alpha=16,
    rank_pattern={
        "^model.decoder.layers.0.encoder_attn.k_proj": 16,
        "^model.decoder.layers.0.encoder_attn.q_proj": 16,
    },
    alpha_pattern={
        "^model.decoder.layers.0": 32,
    },
    target_modules=["q_proj", "v_proj"],
)
</syntaxhighlight>

=== Implementing Custom LyCORIS Layer ===
<syntaxhighlight lang="python">
from peft.tuners.lycoris_utils import LycorisLayer
import torch.nn as nn
import torch

class CustomLycorisLinear(LycorisLayer):
    adapter_layer_names = ("custom_A", "custom_B")

    def __init__(self, base_layer: nn.Module, adapter_name: str, r: int = 8,
                 alpha: float = 16, **kwargs):
        super().__init__(base_layer)
        self.custom_A = nn.ModuleDict({})
        self.custom_B = nn.ModuleDict({})
        self.update_layer(adapter_name, r, alpha, **kwargs)

    @property
    def _available_adapters(self) -> set[str]:
        return set(self.custom_A.keys())

    def create_adapter_parameters(self, adapter_name: str, r: int, **kwargs):
        self.custom_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.custom_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)

    def _get_delta_activations(self, adapter_name: str, x: torch.Tensor,
                               *args, **kwargs) -> torch.Tensor:
        custom_A = self.custom_A[adapter_name]
        custom_B = self.custom_B[adapter_name]
        scaling = self.scaling[adapter_name]
        return custom_B(custom_A(x)) * scaling

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        custom_A = self.custom_A[adapter_name]
        custom_B = self.custom_B[adapter_name]
        return custom_B.weight @ custom_A.weight

    def reset_adapter_parameters(self, adapter_name: str):
        nn.init.kaiming_uniform_(self.custom_A[adapter_name].weight)
        nn.init.zeros_(self.custom_B[adapter_name].weight)

    def update_layer(self, adapter_name: str, r: int, alpha: float, **kwargs):
        if adapter_name not in self.custom_A:
            self.r[adapter_name] = r
            self.alpha[adapter_name] = alpha
            self.scaling[adapter_name] = alpha / r
            self.create_adapter_parameters(adapter_name, r, **kwargs)
            self.reset_adapter_parameters(adapter_name)
</syntaxhighlight>

=== Using Rank and Alpha Patterns ===
<syntaxhighlight lang="python">
# Configure different ranks for different layers
config = LycorisConfig(
    r=8,  # Default rank
    alpha=16,  # Default alpha
    rank_pattern={
        # Attention layers get higher rank
        ".*attn.*": 16,
        # MLP layers get lower rank
        ".*mlp.*": 4,
        # Specific layer gets custom rank
        "^model.layer.0.attention.self.query": 32,
    },
    alpha_pattern={
        # Scale attention layers more
        ".*attn.*": 32,
    },
)

# Pattern matching uses regex, so you can be very specific
# or very general based on your needs
</syntaxhighlight>

=== Merge and Unmerge Operations ===
<syntaxhighlight lang="python">
# Assuming lycoris_layer is a LycorisLayer instance

# Merge adapter into base weights
lycoris_layer.merge(safe_merge=False, adapter_names=["default"])

# Now the base layer includes adapter weights
# Forward pass will be faster but adapters can't be disabled

# Check if merged
if lycoris_layer.merged:
    print("Adapters are merged")

# Unmerge to separate adapters from base weights
lycoris_layer.unmerge()

# Adapters can now be enabled/disabled independently
</syntaxhighlight>

=== Safe Merge with NaN Detection ===
<syntaxhighlight lang="python">
try:
    # Attempt merge with safety check
    lycoris_layer.merge(safe_merge=True, adapter_names=["adapter1", "adapter2"])
    print("Merge successful!")
except ValueError as e:
    print(f"Merge failed: {e}")
    # One of the adapters has NaN values
    # Investigate the adapter or training process
</syntaxhighlight>

=== Scaling Operations ===
<syntaxhighlight lang="python">
# Set scale for specific adapter
lycoris_layer.set_scale("default", scale=2.0)
# New scaling = 2.0 * alpha / r

# Scale all active adapters
lycoris_layer.scale_layer(scale=1.5)
# Multiplies existing scaling by 1.5

# Reset scaling to default
lycoris_layer.unscale_layer(scale=None)
# Resets to alpha / r

# Unscale by specific factor
lycoris_layer.unscale_layer(scale=1.5)
# Divides existing scaling by 1.5
</syntaxhighlight>

=== Implementing Custom LyCORIS Tuner ===
<syntaxhighlight lang="python">
from peft.tuners.lycoris_utils import LycorisTuner
import torch.nn as nn

class CustomLycorisTuner(LycorisTuner):
    prefix: str = "custom_lycoris_"

    # Map layer types to custom implementations
    layers_mapping = {
        nn.Linear: CustomLycorisLinear,
        # Add more mappings as needed
    }

    def _create_and_replace(self, config, adapter_name, target,
                           target_name, parent, current_key):
        # Create new module using base class method
        new_module = self._create_new_module(
            config, adapter_name, target,
            r=config.r,
            alpha=config.alpha,
        )

        # Replace old module with new one
        if adapter_name not in self.active_adapters:
            new_module.requires_grad_(False)

        self._replace_module(parent, target_name, new_module, target)

# Use the custom tuner
model = CustomLycorisTuner(base_model, config, adapter_name="default")
</syntaxhighlight>

== Related Pages ==
* [[LoRA]]
* [[LoHa]]
* [[LoKr]]
* [[PEFT Configuration]]
* [[Parameter-Efficient Fine-Tuning]]
* [[Low-Rank Adaptation]]
* [[Adapter Methods]]
* [[Model Fine-Tuning]]

== Notes ==
* LyCORIS adapters can be more expressive than standard LoRA for certain tasks
* Pattern-based configuration allows fine-grained control over adapter parameters per layer
* The merge operation modifies base weights in-place and cannot be reversed without unmerge
* Safe merge checks for NaN values which can indicate training instability
* Rank and module dropout are supported for regularization
* The framework supports multiple active adapters simultaneously
* Subclasses must implement all abstract methods for the framework to function

== Advanced Features ==

=== Empty Weight Initialization ===
The <code>_init_empty_weights</code> method enables fast initialization:
<syntaxhighlight lang="python">
# Instead of initializing on CPU then moving to device
# Initialize directly on target device without materializing weights
layer._init_empty_weights(
    nn.Linear,
    in_features=768,
    out_features=768,
    device="cuda"
)
</syntaxhighlight>

=== Cast Input Dtype ===
The <code>cast_input_dtype_enabled</code> flag controls automatic dtype casting:
<syntaxhighlight lang="python">
# Enable/disable automatic input dtype casting
lycoris_layer.cast_input_dtype_enabled = True  # Default
# When enabled, inputs are cast to match weight dtype
</syntaxhighlight>

=== Rank and Module Dropout ===
<syntaxhighlight lang="python">
# Rank dropout: randomly drops rank dimensions during training
# Module dropout: randomly drops entire adapter modules
config = LycorisConfig(
    r=16,
    rank_dropout=0.1,  # Drop 10% of rank dimensions
    module_dropout=0.05,  # Drop adapter with 5% probability
)
</syntaxhighlight>

== References ==
* LyCORIS Repository: https://github.com/KohakuBlueleaf/LyCORIS
* LoRA Paper: https://arxiv.org/abs/2106.09685
* PEFT Documentation: https://huggingface.co/docs/peft
* Stable Diffusion Fine-tuning: https://huggingface.co/docs/diffusers
