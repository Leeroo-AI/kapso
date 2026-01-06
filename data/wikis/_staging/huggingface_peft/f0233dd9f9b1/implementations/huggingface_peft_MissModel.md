= MiSS Model =

== Knowledge Sources ==
* '''Repository:''' [https://github.com/huggingface/peft HuggingFace PEFT]
* '''Source File:''' src/peft/tuners/miss/model.py
* '''Paper:''' [https://huggingface.co/papers/2409.15371 MiSS: Householder Reflection Adaptation]

== Domains ==
* [[Natural Language Processing]]
* [[Parameter-Efficient Fine-Tuning]]
* [[Model Tuning]]
* [[Householder Reflection]]
* [[Deep Learning]]

== Overview ==

=== Description ===
The MiSSModel class creates Householder reflection adaptation (MiSS) models from pretrained models. MiSS is a parameter-efficient fine-tuning method that uses Householder reflections to adapt model weights with minimal additional parameters. It provides an alternative to LoRA with different characteristics for memory efficiency and model expressiveness.

Key features:
* '''Householder reflection-based adaptation''': Uses mathematical properties of Householder matrices for efficient weight updates
* '''Dual-rank decomposition''': Separate control over in_features (r) and out_features (mini_r) dimensions
* '''Multiple initialization modes''': Supports balance, bat, and mini variants
* '''Linear layer support''': Currently supports torch.nn.Linear layers
* '''Model architecture mapping''': Automatic target module detection for known architectures
* '''Low memory overhead''': Efficient adapter creation with optional meta device initialization

The implementation follows the PEFT BaseTuner pattern:
* '''MissModel''': Main tuner class for creating and managing MiSS adapters
* '''MissLinear''': Layer implementation (imported from .layer module)
* '''MissLayer''': Base layer class (imported from .layer module)

=== Usage ===
MissModel is used to apply MiSS adapters to pretrained models for parameter-efficient fine-tuning. It's particularly useful for vision and language models where memory efficiency is important.

== Code Reference ==

=== Source Location ===
<code>src/peft/tuners/miss/model.py</code>

=== Class Signature ===
<syntaxhighlight lang="python">
class MissModel(BaseTuner):
    prefix: str = "miss_"
    tuner_layer_cls = MissLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_MISS_TARGET_MODULES_MAPPING

    def __init__(
        self,
        model: torch.nn.Module,
        config: MissConfig,
        adapter_name: str,
        low_cpu_mem_usage: bool = False,
    )
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import MissModel
from peft.tuners.miss import MissConfig
</syntaxhighlight>

== I/O Contract ==

=== Constructor Parameters ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| model || torch.nn.Module || Required || The model to which adapter tuner layers will be attached
|-
| config || MissConfig || Required || Configuration of the MiSS model
|-
| adapter_name || str || Required || Name of the adapter
|-
| low_cpu_mem_usage || bool || False || Create empty adapter weights on meta device for faster loading
|}

=== _create_and_replace Method ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| miss_config || MissConfig || Configuration for MiSS adapter
|-
| adapter_name || str || Name of the adapter
|-
| target || torch.nn.Module || Target module to adapt
|-
| target_name || str || Name of target module
|-
| parent || torch.nn.Module || Parent module containing target
|-
| current_key || str || Key identifying current module in model
|-
| **optional_kwargs || Any || Additional keyword arguments
|}

=== _create_new_module (static method) ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| miss_config || MissConfig || Configuration for MiSS adapter
|-
| adapter_name || str || Name of the adapter
|-
| target || torch.nn.Module || Target module to wrap
|-
| **kwargs || Any || Additional arguments (r, mini_r, miss_dropout, init_weights, bias)
|}

{| class="wikitable"
! Return !! Type !! Description
|-
| new_module || MissLinear || New MiSS layer wrapping the target
|}

== Usage Examples ==

=== Basic Usage with Language Model ===
<syntaxhighlight lang="python">
from peft import MissModel, MissConfig
from transformers import AutoModelForCausalLM
import torch

# Load pretrained model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Create MiSS configuration
config = MissConfig(
    r=8,
    mini_r=1,
    target_modules=["c_attn", "c_proj"],
    init_weights=True,
)

# Apply MiSS adapters
model = MissModel(model, config, adapter_name="default")

# Check trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")
</syntaxhighlight>

=== Usage with Stable Diffusion (Example from Docstring) ===
<syntaxhighlight lang="python">
from diffusers import StableDiffusionPipeline
from peft import MissModel, MissConfig

# Configuration for text encoder
config_te = MissConfig(
    r=8,
    target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
    init_weights=True,
)

# Configuration for UNet
config_unet = MissConfig(
    r=8,
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
    init_weights=True,
)

# Load Stable Diffusion pipeline
model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# Apply MiSS to text encoder and UNet
model.text_encoder = MissModel(model.text_encoder, config_te, "default")
model.unet = MissModel(model.unet, config_unet, "default")

# Now ready for fine-tuning
</syntaxhighlight>

=== Using with PEFT get_peft_model ===
<syntaxhighlight lang="python">
from peft import get_peft_model, MissConfig
from transformers import AutoModelForSequenceClassification

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)

# Create configuration
config = MissConfig(
    r=16,
    mini_r=2,
    target_modules=["query", "value"],
    modules_to_save=["classifier"],
    init_weights=True,
)

# Apply MiSS using PEFT helper
model = get_peft_model(model, config)

# Train the model
trainer.train(model)
</syntaxhighlight>

=== Low Memory Usage Mode ===
<syntaxhighlight lang="python">
from peft import MissModel, MissConfig

# For large models, use low_cpu_mem_usage to speed up loading
config = MissConfig(
    r=64,
    mini_r=4,
    target_modules="all-linear",
    init_weights=True,
)

# Weights created on meta device, then moved to target device
model = MissModel(
    base_model,
    config,
    adapter_name="default",
    low_cpu_mem_usage=True  # Faster loading for large models
)
</syntaxhighlight>

=== Adding Multiple Adapters ===
<syntaxhighlight lang="python">
from peft import MissModel, MissConfig

# Create base model with first adapter
config1 = MissConfig(r=8, target_modules=["q_proj", "v_proj"])
model = MissModel(base_model, config1, adapter_name="task1")

# Add second adapter
config2 = MissConfig(r=16, target_modules=["q_proj", "v_proj"])
model.add_adapter("task2", config2)

# Switch between adapters
model.set_adapter("task1")  # Use task1 adapter
output1 = model(**inputs)

model.set_adapter("task2")  # Use task2 adapter
output2 = model(**inputs)
</syntaxhighlight>

=== Using Different Initialization Modes ===
<syntaxhighlight lang="python">
# Balanced initialization (default)
config_balanced = MissConfig(
    r=64,
    mini_r=1,
    target_modules=["q_proj", "k_proj", "v_proj"],
    init_weights=True,  # Most efficient and general
)
model_balanced = MissModel(base_model, config_balanced, "balanced")

# Bat initialization (nonlinear updates)
config_bat = MissConfig(
    r=64,
    mini_r=1,
    target_modules=["q_proj", "k_proj", "v_proj"],
    init_weights="bat",  # Nonlinear updates across shards
)
model_bat = MissModel(base_model, config_bat, "bat")

# Mini initialization (smaller rank)
config_mini = MissConfig(
    r=32,
    mini_r=8,  # Ensure out_features % mini_r == 0
    target_modules=["q_proj", "k_proj", "v_proj"],
    init_weights="mini",  # Fewer trainable parameters
)
model_mini = MissModel(base_model, config_mini, "mini")
</syntaxhighlight>

=== Training with MiSS ===
<syntaxhighlight lang="python">
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, MissConfig

# Setup
model = AutoModelForCausalLM.from_pretrained("gpt2")
config = MissConfig(
    r=8,
    mini_r=1,
    miss_dropout=0.05,
    target_modules=["c_attn"],
    bias="none",
)
peft_model = get_peft_model(model, config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./miss_model",
    per_device_train_batch_size=4,
    learning_rate=3e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
)

# Train
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()

# Save adapter
peft_model.save_pretrained("./miss_adapter")
</syntaxhighlight>

=== Loading Saved Adapter ===
<syntaxhighlight lang="python">
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Load MiSS adapter
model = PeftModel.from_pretrained(
    base_model,
    "path/to/miss_adapter"
)

# Use for inference
output = model.generate(**inputs)
</syntaxhighlight>

=== Inspecting Adapted Layers ===
<syntaxhighlight lang="python">
from peft import MissModel, MissConfig

model = MissModel(base_model, config, "default")

# Check which layers were adapted
print("Adapted modules:")
for name, module in model.named_modules():
    if hasattr(module, 'miss_'):
        print(f"  {name}")

# Check specific layer
target_layer = model.model.layer[0].attention.self.query
if hasattr(target_layer, 'base_layer'):
    print(f"Layer has MiSS adapter")
    print(f"  r = {target_layer.r}")
    print(f"  mini_r = {target_layer.mini_r}")
</syntaxhighlight>

== Related Pages ==
* [[MiSS Configuration]]
* [[MiSS Layer]]
* [[PEFT Configuration]]
* [[Parameter-Efficient Fine-Tuning]]
* [[Householder Reflection]]
* [[LoRA Model]]
* [[Model Adapter Methods]]
* [[BaseTuner]]

== Notes ==

=== Supported Layer Types ===
* Currently only <code>torch.nn.Linear</code> layers are supported
* Attempting to adapt other layer types will raise a ValueError
* Future versions may add support for Conv2d and other layer types

=== Adapter Management ===
* Multiple adapters can be added to the same model
* Each adapter has its own name and configuration
* Only one adapter is active at a time (can be switched)
* Adapters can be merged into base weights or kept separate

=== Memory Efficiency ===
* <code>low_cpu_mem_usage=True</code> creates adapters on meta device first
* Useful for very large models to avoid CPU memory spikes
* Adapters are then materialized on the target device

=== Initialization Modes ===
* '''Balance''' (default): Most efficient, suitable for most tasks
* '''Bat''': Enables nonlinear updates, potentially more expressive
* '''Mini''': Fewest parameters, requires out_features % mini_r == 0

=== Target Module Mapping ===
The <code>TRANSFORMERS_MODELS_TO_MISS_TARGET_MODULES_MAPPING</code> provides default target modules for known architectures:
* Automatically selects appropriate layers for popular models
* Can be overridden by specifying <code>target_modules</code> in config
* Supports models like BERT, GPT-2, LLaMA, etc.

== Implementation Details ==

=== Module Creation Process ===
1. <code>_create_and_replace</code> is called for each target module
2. Checks if module is already a MissLayer
3. If not, calls <code>_create_new_module</code> to create MissLinear
4. Validates that target is torch.nn.Linear
5. Replaces original module with MiSS-adapted version
6. If adapter not active, sets requires_grad to False

=== Layer Update Process ===
If target is already a MissLayer:
1. Calls <code>update_layer</code> on existing layer
2. Adds new adapter to existing MissLayer
3. Preserves other adapters already present

=== Error Handling ===
* Raises ValueError if <code>current_key</code> is None
* Raises ValueError if target is not torch.nn.Linear
* Provides helpful error messages for unsupported layer types

== Advanced Usage ==

=== Combining with Other PEFT Methods ===
<syntaxhighlight lang="python">
# MiSS can be combined with other PEFT methods
from peft import get_peft_model, MissConfig, LoraConfig

# Apply MiSS to some layers
miss_config = MissConfig(
    r=8,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(base_model, miss_config)

# Add LoRA to other layers
lora_config = LoraConfig(
    r=8,
    target_modules=["k_proj", "o_proj"]
)
model.add_adapter("lora_adapter", lora_config)
</syntaxhighlight>

=== Custom Target Module Selection ===
<syntaxhighlight lang="python">
def get_attention_modules(model):
    """Extract all attention module names"""
    attention_modules = []
    for name, module in model.named_modules():
        if "attn" in name.lower() and isinstance(module, torch.nn.Linear):
            attention_modules.append(name)
    return attention_modules

target_modules = get_attention_modules(base_model)
config = MissConfig(r=16, target_modules=target_modules)
model = MissModel(base_model, config, "custom")
</syntaxhighlight>

== References ==
* MiSS Paper: https://huggingface.co/papers/2409.15371
* PEFT Documentation: https://huggingface.co/docs/peft
* Householder Transformation: https://en.wikipedia.org/wiki/Householder_transformation
* Parameter-Efficient Fine-Tuning Survey: https://arxiv.org/abs/2110.04366
* Diffusers Documentation: https://huggingface.co/docs/diffusers
