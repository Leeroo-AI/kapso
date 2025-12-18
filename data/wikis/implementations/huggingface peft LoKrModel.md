{{Implementation
|domain=NLP,Computer Vision,PEFT,Parameter-Efficient Fine-Tuning,LoKr,Low-Rank Adaptation,Kronecker Product,Diffusion Models
|link=https://github.com/huggingface/peft
}}

== Overview ==

=== Description ===

The <code>LoKrModel</code> class creates Low-Rank Kronecker Product (LoKr) models from pretrained models. This class implements an advanced parameter-efficient fine-tuning method that uses Kronecker product decomposition to achieve extreme parameter efficiency while maintaining model performance.

LoKrModel inherits from <code>LycorisTuner</code> and supports various layer types including Linear, Conv1d, and Conv2d. The method is described in multiple papers including https://huggingface.co/papers/2108.06098 and https://huggingface.co/papers/2309.14859. The current implementation heavily borrows from the LyCORIS repository (https://github.com/KohakuBlueleaf/LyCORIS).

Key features include:
* Kronecker product-based decomposition for extreme compression
* Support for Linear, Conv1d, and Conv2d layers
* Optional decomposition of both Kronecker matrices for higher compression
* Automatic target module identification for common architectures
* Layer-specific rank and alpha configuration through patterns
* Rank dropout with optional scaling
* Compatible with language models, vision models, and diffusion models

=== Usage ===

LoKrModel is typically instantiated through the PEFT library's interface or directly for fine-grained control. It's particularly effective when extreme parameter efficiency is needed, often achieving better compression ratios than standard LoRA or LoHa while maintaining comparable performance.

== Code Reference ==

=== Source Location ===
* '''Repository:''' huggingface/peft
* '''File Path:''' <code>src/peft/tuners/lokr/model.py</code>
* '''Lines:''' 27-119

=== Signature ===
<syntaxhighlight lang="python">
class LoKrModel(LycorisTuner):
    prefix: str = "lokr_"
    tuner_layer_cls = LoKrLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LOKR_TARGET_MODULES_MAPPING
    layers_mapping: dict[type[torch.nn.Module], type[LoKrLayer]] = {
        torch.nn.Conv2d: Conv2d,
        torch.nn.Conv1d: Conv1d,
        torch.nn.Linear: Linear,
    }

    def _create_and_replace(
        self,
        config: LycorisConfig,
        adapter_name: str,
        target: Union[LoKrLayer, nn.Module],
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.lokr.model import LoKrModel
# or
from peft import LoKrModel
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===

'''Constructor:'''
{| class="wikitable"
! Parameter !! Type !! Description
|-
| <code>model</code> || <code>torch.nn.Module</code> || The model to which the adapter tuner layers will be attached
|-
| <code>config</code> || <code>LoKrConfig</code> || The configuration of the LoKr model
|-
| <code>adapter_name</code> || <code>str</code> || The name of the adapter (default: "default")
|-
| <code>low_cpu_mem_usage</code> || <code>bool</code> || Create empty adapter weights on meta device for faster loading (default: False)
|}

'''Method: _create_and_replace'''
{| class="wikitable"
! Parameter !! Type !! Description
|-
| <code>config</code> || <code>LycorisConfig</code> || Configuration for the adapter
|-
| <code>adapter_name</code> || <code>str</code> || Name of the adapter to create
|-
| <code>target</code> || <code>Union[LoKrLayer, nn.Module]</code> || Target module to replace or update
|-
| <code>target_name</code> || <code>str</code> || Name of the target module
|-
| <code>parent</code> || <code>nn.Module</code> || Parent module containing the target
|-
| <code>current_key</code> || <code>str</code> || Key path to the current module
|}

=== Outputs ===

'''Attributes:'''
{| class="wikitable"
! Attribute !! Type !! Description
|-
| <code>model</code> || <code>torch.nn.Module</code> || The adapted model with LoKr layers
|-
| <code>peft_config</code> || <code>LoKrConfig</code> || The configuration used for adaptation
|-
| <code>prefix</code> || <code>str</code> || Parameter prefix for LoKr adapters ("lokr_")
|-
| <code>tuner_layer_cls</code> || <code>type</code> || The base layer class (LoKrLayer)
|-
| <code>layers_mapping</code> || <code>dict</code> || Mapping from PyTorch layer types to LoKr layer types
|}

'''Returns:'''
{| class="wikitable"
! Type !! Description
|-
| <code>torch.nn.Module</code> || The LoKr-adapted model ready for training or inference
|}

== Usage Examples ==

=== Example 1: Basic LoKr Model for Language Models ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import LoKrModel, LoKrConfig

# Configure LoKr
config = LoKrConfig(
    r=8,
    alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    rank_dropout=0.0,
    module_dropout=0.0,
    init_weights=True,
)

# Load and adapt model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = LoKrModel(model, config, "default")

# Check trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable: {trainable_params} / {total_params} ({100*trainable_params/total_params:.2f}%)")
</syntaxhighlight>

=== Example 2: LoKr for Stable Diffusion with Maximum Compression ===
<syntaxhighlight lang="python">
from diffusers import StableDiffusionPipeline
from peft import LoKrModel, LoKrConfig

# Configuration for text encoder
config_te = LoKrConfig(
    r=8,
    alpha=32,
    target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
    rank_dropout=0.0,
    module_dropout=0.0,
    init_weights=True,
)

# Configuration for UNet with maximum compression
config_unet = LoKrConfig(
    r=8,
    alpha=32,
    target_modules=[
        "proj_in",
        "proj_out",
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    ],
    rank_dropout=0.0,
    module_dropout=0.0,
    init_weights=True,
    use_effective_conv2d=True,  # Efficient conv decomposition
    decompose_both=True,  # Decompose both Kronecker matrices
    decompose_factor=-1,  # Auto-select factor
)

# Load and adapt pipeline
model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
model.text_encoder = LoKrModel(model.text_encoder, config_te, "default")
model.unet = LoKrModel(model.unet, config_unet, "default")

print("Ultra-compressed LoKr model ready for fine-tuning")
</syntaxhighlight>

=== Example 3: LoKr with Layer-Specific Ranks and Scaling ===
<syntaxhighlight lang="python">
from transformers import AutoModel
from peft import LoKrModel, LoKrConfig

# Configure with rank patterns and dropout scaling
config = LoKrConfig(
    r=8,
    alpha=8,
    target_modules=["query", "key", "value"],
    rank_dropout=0.1,
    rank_dropout_scale=True,  # Scale dropout for stability
    # Layer-specific ranks
    rank_pattern={
        "^model.encoder.layer.[0-5].*": 16,  # Higher rank for early layers
        "^model.encoder.layer.[6-9].*": 8,   # Default rank for middle
        "^model.encoder.layer.1[0-1].*": 4,  # Lower rank for late layers
    },
    alpha_pattern={
        "^model.encoder.layer.[0-5].*": 32,
    },
    decompose_both=True,
    init_weights=True,
)

model = AutoModel.from_pretrained("bert-base-uncased")
model = LoKrModel(model, config, "default")
</syntaxhighlight>

=== Example 4: LoKr with LyCORIS Initialization ===
<syntaxhighlight lang="python">
from transformers import AutoModelForSeq2SeqLM
from peft import LoKrModel, LoKrConfig

# Use LyCORIS-style initialization for better convergence
config = LoKrConfig(
    r=16,
    alpha=32,
    target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    init_weights="lycoris",  # LyCORIS initialization
    decompose_both=True,
    decompose_factor=4,  # Specific decomposition factor
    rank_dropout=0.1,
    module_dropout=0.05,
    rank_dropout_scale=True,
)

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
model = LoKrModel(model, config, "default")

print("LoKr model with LyCORIS initialization ready")
</syntaxhighlight>

=== Example 5: LoKr for Vision Transformer ===
<syntaxhighlight lang="python">
from transformers import ViTForImageClassification
from peft import LoKrModel, LoKrConfig

# Configure LoKr for ViT
config = LoKrConfig(
    r=8,
    alpha=16,
    target_modules=["query", "key", "value"],
    rank_dropout=0.1,
    module_dropout=0.05,
    decompose_both=True,
    # Only adapt certain layers
    layers_to_transform=[0, 1, 2, 9, 10, 11],  # First and last few layers
    layers_pattern="encoder.layer",
    init_weights=True,
)

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=10
)
model = LoKrModel(model, config, "default")

# Train on custom dataset
# ... training loop ...
</syntaxhighlight>

=== Example 6: Multi-Task LoKr with Different Configs ===
<syntaxhighlight lang="python">
from peft import LoKrModel, LoKrConfig

# Create base LoKr model for task 1
config1 = LoKrConfig(
    r=8,
    alpha=16,
    target_modules=["q_proj", "v_proj"],
    decompose_both=False,
)
# Conceptual: First adapter
model = LoKrModel(base_model, config1, adapter_name="task1")

# Configuration for task 2 with different settings
config2 = LoKrConfig(
    r=16,
    alpha=32,
    target_modules=["q_proj", "v_proj", "o_proj"],
    decompose_both=True,  # More compression for task 2
    decompose_factor=4,
)

# In practice, multi-adapter management is done through PeftModel
# This shows the conceptual structure
print("Multi-task LoKr model configured")
</syntaxhighlight>

== Implementation Details ==

=== Kronecker Product Decomposition ===
LoKr decomposes weight matrices using Kronecker products:
<syntaxhighlight lang="python">
W ≈ A ⊗ B

Where:
- W: Original weight matrix (m × n)
- A: First Kronecker factor (m1 × n1)
- B: Second Kronecker factor (m2 × n2)
- m = m1 * m2, n = n1 * n2
</syntaxhighlight>

When <code>decompose_both=True</code>:
<syntaxhighlight lang="python">
A ≈ U_A @ V_A.T  (low-rank decomposition)
B ≈ U_B @ V_B.T  (low-rank decomposition)

W ≈ (U_A @ V_A.T) ⊗ (U_B @ V_B.T)
</syntaxhighlight>

This achieves extreme compression with minimal accuracy loss.

=== Layer Mapping System ===
LoKrModel uses a <code>layers_mapping</code> dictionary:
<syntaxhighlight lang="python">
layers_mapping = {
    torch.nn.Conv2d: Conv2d,  # LoKr Conv2d layer
    torch.nn.Conv1d: Conv1d,  # LoKr Conv1d layer
    torch.nn.Linear: Linear,  # LoKr Linear layer
}
</syntaxhighlight>

Each LoKr layer type implements the Kronecker product decomposition appropriate for its layer type.

=== Adapter Creation Process ===
The <code>_create_and_replace</code> method:
1. Extracts rank and alpha from patterns or uses defaults
2. Adds <code>rank_dropout_scale</code> to kwargs
3. Checks if target is already a LoKrLayer
   - If yes: updates the layer with new adapter
   - If no: creates new LoKrLayer wrapping the target
4. Replaces the original module in the parent

<syntaxhighlight lang="python">
# Simplified implementation:
r_key = get_pattern_key(config.rank_pattern.keys(), current_key)
alpha_key = get_pattern_key(config.alpha_pattern.keys(), current_key)
kwargs = config.to_dict()
kwargs["r"] = config.rank_pattern.get(r_key, config.r)
kwargs["alpha"] = config.alpha_pattern.get(alpha_key, config.alpha)
kwargs["rank_dropout_scale"] = config.rank_dropout_scale

if isinstance(target, LoKrLayer):
    target.update_layer(adapter_name, **kwargs)
else:
    new_module = self._create_new_module(config, adapter_name, target, **kwargs)
    self._replace_module(parent, target_name, new_module, target)
</syntaxhighlight>

=== Parameter Efficiency Comparison ===
For a matrix of size m×n with rank r:
* '''Standard LoRA:''' 2 * r * (m + n) parameters
* '''LoHa:''' 4 * r * sqrt(m) * sqrt(n) parameters (assuming square-like)
* '''LoKr (basic):''' 2 * sqrt(m*n) * (m1 + n1) parameters
* '''LoKr (decompose_both):''' Even fewer parameters with nested decomposition

LoKr often achieves 50-70% of LoRA's parameter count with similar performance.

=== Target Module Resolution ===
Uses <code>TRANSFORMERS_MODELS_TO_LOKR_TARGET_MODULES_MAPPING</code> for automatic module identification across different architectures.

=== Parameter Prefix ===
All LoKr parameters use the prefix "lokr_" for easy identification in state dicts.

== Related Pages ==

* [[huggingface_peft_LoKrConfig|LoKrConfig]] - Configuration for LoKr models
* [[huggingface_peft_LoKrLayer|LoKrLayer]] - Base layer class for LoKr adapters
* [[huggingface_peft_LycorisTuner|LycorisTuner]] - Base class for Lycoris-based tuners
* [[huggingface_peft_LoHaModel|LoHaModel]] - Related model using Hadamard products
* [[huggingface_peft_LoraModel|LoraModel]] - Standard LoRA implementation
* [[huggingface_peft_get_peft_model|get_peft_model]] - Standard function to create PEFT models

[[Category:PEFT]]
[[Category:Model]]
[[Category:LoKr]]
[[Category:Kronecker Product]]
[[Category:Low-Rank Adaptation]]
[[Category:Parameter-Efficient Fine-Tuning]]
[[Category:Diffusion Models]]
