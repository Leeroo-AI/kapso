= XLoraConfig =

== Knowledge Sources ==

* '''Repository''': [https://github.com/huggingface/peft HuggingFace PEFT]
* '''Paper''': X-LoRA: Mixture of Low-Rank Adapters
* '''Type''': Configuration Class
* '''Module''': peft.tuners.xlora.config

== Domains ==

[[Category:Natural_Language_Processing]]
[[Category:Parameter_Efficient_Fine_Tuning]]
[[Category:Mixture_of_Experts]]
[[Category:Low_Rank_Adaptation]]
[[Category:Configuration]]

== Overview ==

=== Description ===

XLoraConfig is the configuration class for X-LoRA (Mixture of LoRA Experts), a technique that dynamically combines multiple LoRA adapters using a learned classifier. Unlike standard LoRA which applies a single adapter, X-LoRA trains a gating network to predict mixing weights for multiple LoRA adapters based on the input, enabling dynamic expert selection.

The configuration controls:
* The classifier architecture (depth, size, dropout)
* The softmax behavior (temperature, top-k sparsity)
* Whether scalings are layer-specific or shared
* Which LoRA adapters to use as experts
* Training and inference parameters

=== Usage ===

XLoraConfig is used to initialize an X-LoRA model that manages multiple LoRA adapters. The adapters dictionary specifies which LoRA experts to load, and the classifier configuration determines how they are dynamically weighted.

== Code Reference ==

=== Source Location ===

<code>/tmp/praxium_repo_zyf9ywdz/src/peft/tuners/xlora/config.py</code>

=== Signature ===

<syntaxhighlight lang="python">
@dataclass
class XLoraConfig(PeftConfig):
    hidden_size: int = None
    adapters: dict[str, str] = None
    enable_softmax: bool = True
    enable_softmax_topk: bool = False
    layerwise_scalings: bool = False
    xlora_depth: int = 1
    xlora_size: int = 2048
    xlora_dropout_p: float = 0.2
    use_trainable_adapters: bool = False
    softmax_temperature: float = 1.0
    top_k_lora: Optional[int] = None
    scaling_pass_value: float = 0.0
    global_scaling_weight: float = 1.0
</syntaxhighlight>

=== Import ===

<syntaxhighlight lang="python">
from peft import XLoraConfig
</syntaxhighlight>

== I/O Contract ==

=== Configuration Parameters ===

{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| hidden_size || int || None (4096) || Hidden size of the base model
|-
| adapters || dict[str, str] || None ({}) || Mapping of adapter names to LoRA adapter IDs for loading
|-
| enable_softmax || bool || True || Enable softmax normalization for classifier output
|-
| enable_softmax_topk || bool || False || Enable softmax only for top-k adapters (requires top_k_lora)
|-
| softmax_temperature || float || 1.0 || Temperature for softmax (lower = sharper predictions)
|-
| layerwise_scalings || bool || False || Generate per-layer scalings vs broadcasting same scalings
|-
| top_k_lora || int || None || Sparse selection of top-k LoRA experts (None = dense)
|-
| xlora_depth || int || 1 || Number of layers in X-LoRA classifier network
|-
| xlora_size || int || 2048 || Hidden size of classifier (irrelevant if xlora_depth=1)
|-
| xlora_dropout_p || float || 0.2 || Dropout probability in classifier (irrelevant if xlora_depth=1)
|-
| use_trainable_adapters || bool || False || Whether to make the LoRA adapters trainable
|-
| scaling_pass_value || float || 0.0 || Value for dummy scalings in initial forward pass
|-
| global_scaling_weight || float || 1.0 || Global multiplier for all LoRA adapter outputs
|}

=== Validation Rules ===

* If <code>hidden_size</code> is None, defaults to 4096 with a warning
* If <code>adapters</code> is None, defaults to empty dict with a warning
* If <code>enable_softmax_topk=True</code> and <code>top_k_lora</code> is None, issues warning
* If both <code>enable_softmax_topk</code> and <code>enable_softmax</code> are True, warns about worse performance
* If <code>top_k_lora < 1</code>, issues warning about invalid value
* <code>peft_type</code> is automatically set to <code>PeftType.XLORA</code>

== Usage Examples ==

=== Basic X-LoRA Configuration ===

<syntaxhighlight lang="python">
from peft import XLoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Define X-LoRA configuration with multiple adapters
config = XLoraConfig(
    hidden_size=4096,
    adapters={
        "math": "path/to/math_lora",
        "code": "path/to/code_lora",
        "general": "path/to/general_lora"
    },
    enable_softmax=True,
    xlora_depth=2,
    xlora_size=2048,
    xlora_dropout_p=0.1
)

# Create X-LoRA model that dynamically mixes the three adapters
model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Shallow Classifier (Fast) ===

<syntaxhighlight lang="python">
from peft import XLoraConfig

# Single-layer classifier for fast inference
config = XLoraConfig(
    hidden_size=4096,
    adapters={
        "adapter1": "path/to/adapter1",
        "adapter2": "path/to/adapter2"
    },
    xlora_depth=1,  # Single linear layer
    enable_softmax=True,
    softmax_temperature=1.0
)
</syntaxhighlight>

=== Deep Classifier (More Capacity) ===

<syntaxhighlight lang="python">
from peft import XLoraConfig

# Multi-layer classifier with dropout for complex mixing
config = XLoraConfig(
    hidden_size=4096,
    adapters={
        "expert1": "path/to/expert1",
        "expert2": "path/to/expert2",
        "expert3": "path/to/expert3",
        "expert4": "path/to/expert4"
    },
    xlora_depth=4,  # Four layers with ReLU and dropout
    xlora_size=2048,
    xlora_dropout_p=0.2,
    enable_softmax=True
)
</syntaxhighlight>

=== Sparse Top-K Selection ===

<syntaxhighlight lang="python">
from peft import XLoraConfig

# Use sparse top-k instead of dense mixing
config = XLoraConfig(
    hidden_size=4096,
    adapters={
        "expert1": "path/to/expert1",
        "expert2": "path/to/expert2",
        "expert3": "path/to/expert3",
        "expert4": "path/to/expert4",
        "expert5": "path/to/expert5"
    },
    top_k_lora=2,  # Only use top 2 adapters per token
    enable_softmax_topk=True,  # Softmax only over selected adapters
    enable_softmax=False,  # Disable full softmax
    xlora_depth=2
)
</syntaxhighlight>

=== Temperature-Controlled Sharpness ===

<syntaxhighlight lang="python">
from peft import XLoraConfig

# Sharp predictions (favor single adapter)
sharp_config = XLoraConfig(
    hidden_size=4096,
    adapters={"a1": "p1", "a2": "p2", "a3": "p3"},
    softmax_temperature=0.5,  # Lower temperature = sharper
    enable_softmax=True
)

# Smooth predictions (more uniform mixing)
smooth_config = XLoraConfig(
    hidden_size=4096,
    adapters={"a1": "p1", "a2": "p2", "a3": "p3"},
    softmax_temperature=2.0,  # Higher temperature = smoother
    enable_softmax=True
)
</syntaxhighlight>

=== Layerwise vs Shared Scalings ===

<syntaxhighlight lang="python">
from peft import XLoraConfig

# Layerwise scalings (different adapter weights per layer)
layerwise_config = XLoraConfig(
    hidden_size=4096,
    adapters={"task1": "path1", "task2": "path2"},
    layerwise_scalings=True,  # Predict weights for each layer
    xlora_depth=2,
    enable_softmax=True
)

# Shared scalings (same weights broadcast to all layers)
shared_config = XLoraConfig(
    hidden_size=4096,
    adapters={"task1": "path1", "task2": "path2"},
    layerwise_scalings=False,  # Single set of weights for all layers
    xlora_depth=1,  # Can use simpler classifier
    enable_softmax=True
)
</syntaxhighlight>

=== Trainable Adapters Mode ===

<syntaxhighlight lang="python">
from peft import XLoraConfig

# Make adapters trainable (fine-tune both classifier and adapters)
config = XLoraConfig(
    hidden_size=4096,
    adapters={
        "base_math": "path/to/math_adapter",
        "base_code": "path/to/code_adapter"
    },
    use_trainable_adapters=True,  # Allow adapter parameters to update
    xlora_depth=2,
    xlora_dropout_p=0.1,
    enable_softmax=True
)
</syntaxhighlight>

=== Global Scaling Weight ===

<syntaxhighlight lang="python">
from peft import XLoraConfig

# Apply global scaling to all adapter outputs
config = XLoraConfig(
    hidden_size=4096,
    adapters={"a1": "p1", "a2": "p2"},
    global_scaling_weight=0.5,  # Scale all adapter outputs by 0.5
    enable_softmax=True,
    xlora_depth=1
)
</syntaxhighlight>

=== Custom Scaling Pass Value ===

<syntaxhighlight lang="python">
from peft import XLoraConfig

# Set custom value for dummy scalings in initial pass
config = XLoraConfig(
    hidden_size=4096,
    adapters={"a1": "p1", "a2": "p2", "a3": "p3"},
    scaling_pass_value=0.333,  # Use 1/n_adapters instead of 0
    enable_softmax=True
)
</syntaxhighlight>

=== Loading from Pretrained ===

<syntaxhighlight lang="python">
from peft import PeftModel, XLoraConfig

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("model_name")

# Load X-LoRA model with new adapters
# The saved adapter paths are replaced by new ones
model = PeftModel.from_pretrained(
    base_model,
    "path/to/xlora_checkpoint",
    adapters={
        "new_adapter1": "path/to/new1",
        "new_adapter2": "path/to/new2"
    }
)
</syntaxhighlight>

=== Multiple Domain Experts ===

<syntaxhighlight lang="python">
from peft import XLoraConfig, get_peft_model

# Configure X-LoRA with domain-specific experts
config = XLoraConfig(
    hidden_size=4096,
    adapters={
        "mathematics": "username/math_lora",
        "programming": "username/code_lora",
        "science": "username/science_lora",
        "literature": "username/literature_lora",
        "general": "username/general_lora"
    },
    xlora_depth=3,
    xlora_size=2048,
    xlora_dropout_p=0.15,
    softmax_temperature=0.8,
    layerwise_scalings=True,
    enable_softmax=True,
    global_scaling_weight=1.0
)

model = get_peft_model(base_model, config)

# The classifier will learn to route different inputs to appropriate experts
# Math questions -> mathematics adapter
# Code questions -> programming adapter
# etc.
</syntaxhighlight>

== Configuration Combinations ==

=== Dense vs Sparse ===

{| class="wikitable"
! Mode !! enable_softmax !! enable_softmax_topk !! top_k_lora !! Behavior
|-
| Dense || True || False || None || All adapters weighted with softmax
|-
| Sparse || False || True || k || Top-k adapters with softmax
|-
| Hard || False || False || k || Top-k adapters without normalization
|}

=== Classifier Complexity ===

{| class="wikitable"
! xlora_depth !! Parameters !! Use Case
|-
| 1 || hidden_size * n_adapters || Fast inference, simple mixing
|-
| 2-3 || Moderate || Balanced capacity and speed
|-
| 4+ || High || Complex routing, many adapters
|}

== Related Pages ==

* [[huggingface_peft_XLoraClassifier|XLoraClassifier]] - Classifier implementation
* [[huggingface_peft_XLoraModel|XLoraModel]] - Main X-LoRA model class
* [[huggingface_peft_XLoraLayer|XLoraLayer]] - Layer implementation
* [[huggingface_peft_LoraConfig|LoraConfig]] - Base LoRA configuration
* [[huggingface_peft_PeftConfig|PeftConfig]] - Base configuration class
* [[Mixture_of_Experts|MoE Architectures]]
* [[Parameter_Efficient_Fine_Tuning|PEFT Overview]]
