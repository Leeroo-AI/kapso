= VBLoRAModel =

== Knowledge Sources ==
* [https://github.com/huggingface/peft HuggingFace PEFT Repository]
* [https://huggingface.co/papers/2405.15179 VB-LoRA Paper]
* Source: src/peft/tuners/vblora/model.py

== Domains ==
* [[Natural Language Processing (NLP)]]
* [[Parameter-Efficient Fine-Tuning (PEFT)]]
* [[Transfer Learning]]
* [[Low-Rank Adaptation]]
* [[Model Compression]]

== Overview ==

=== Description ===
VBLoRAModel creates a Vector Bank Low-Rank Adaptation (VB-LoRA) model from a pre-trained transformers model. It is the main model class that coordinates the injection of VB-LoRA adapters into target modules, manages the shared vector bank across all layers, and provides specialized methods for parameter counting that account for the unique storage characteristics of VB-LoRA.

The model extends BaseTuner and handles initialization of the shared vector bank, creation and replacement of target modules with VB-LoRA layers, and provides methods for computing savable parameters based on whether save_only_topk_weights mode is enabled.

=== Usage ===
VBLoRAModel is typically instantiated through get_peft_model() by passing a VBLoRAConfig. It automatically initializes the shared vector bank, identifies target modules, wraps them with VB-LoRA layers, and manages the adaptation process. The model provides special handling for parameter-efficient storage when save_only_topk_weights is enabled.

== Code Reference ==

=== Source Location ===
File: src/peft/tuners/vblora/model.py
Lines: 29-210

=== Class Signature ===
<syntaxhighlight lang="python">
class VBLoRAModel(BaseTuner):
    """
    Creates VB-LoRA model from a pretrained transformers model.

    Args:
        model: The model to be adapted
        config: The VBLoRAConfig configuration
        adapter_name: The name of the adapter (defaults to "default")
        low_cpu_mem_usage: Create empty weights on meta device

    Returns:
        torch.nn.Module: The VB-LoRA model
    """
    prefix: str = "vblora_"
    tuner_layer_cls = VBLoRALayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_VBLORA_TARGET_MODULES_MAPPING
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from peft.tuners.vblora.model import VBLoRAModel
# Or via the main PEFT interface
from peft import get_peft_model, VBLoRAConfig
</syntaxhighlight>

== I/O Contract ==

=== Initialization Parameters ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| model || PreTrainedModel || The base model to be adapted
|-
| config || VBLoRAConfig || Configuration for VB-LoRA adaptation
|-
| adapter_name || str || Name of the adapter (default: "default")
|-
| low_cpu_mem_usage || bool || Create empty adapter weights on meta device (default: False)
|}

=== Class Attributes ===
{| class="wikitable"
! Attribute !! Type !! Description
|-
| prefix || str || Prefix for VB-LoRA parameters ("vblora_")
|-
| tuner_layer_cls || Type || VBLoRALayer class reference
|-
| target_module_mapping || dict || Mapping of architectures to default target modules
|-
| vblora_vector_bank || nn.ParameterDict || Shared vector bank across all layers
|}

=== Key Methods ===
{| class="wikitable"
! Method !! Parameters !! Returns !! Description
|-
| _init_vblora_vector_bank || config, adapter_name || None || Initialize shared vector bank
|-
| _pre_injection_hook || model, config, adapter_name || None || Setup before adapter injection
|-
| _create_and_replace || vblora_config, adapter_name, target, target_name, parent, current_key || None || Create and replace module with VB-LoRA layer
|-
| _create_new_module || vblora_config, vblora_vector_bank, adapter_name, target, kwargs || Linear || Create new VB-LoRA module
|-
| get_nb_savable_parameters || adapter || tuple[int, int] || Get number of savable VB-LoRA and other parameters
|-
| print_savable_parameters || None || None || Print savable parameter statistics
|}

== Usage Examples ==

=== Basic VB-LoRA Model Creation ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import VBLoRAConfig, get_peft_model

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# Create VB-LoRA configuration
config = VBLoRAConfig(
    r=4,
    num_vectors=256,
    vector_length=256,
    topk=2,
    target_modules=["fc1", "fc2", "q_proj", "v_proj"]
)

# Create VB-LoRA model
model = get_peft_model(base_model, config)

# Print trainable and savable parameters
model.print_trainable_parameters()
model.print_savable_parameters()
</syntaxhighlight>

=== Full Training Example ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from peft import VBLoRAConfig, get_peft_model

# Setup
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
config = VBLoRAConfig(
    r=4,
    num_vectors=256,
    vector_length=256,
    topk=2,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    vblora_dropout=0.1
)

model = get_peft_model(base_model, config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./vblora_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=1e-4,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
</syntaxhighlight>

=== Save for Inference Only ===
<syntaxhighlight lang="python">
# Train with full weights
config = VBLoRAConfig(
    r=4,
    num_vectors=256,
    vector_length=256,
    topk=2,
    save_only_topk_weights=True,  # Minimal storage
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(base_model, config)

# Train (save_only_topk_weights applies at save time)
trainer.train()

# Save - only top-k weights are saved
model.save_pretrained("./vblora_inference_only")

# Load for inference
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
model = PeftModel.from_pretrained(base_model, "./vblora_inference_only")
model.eval()
</syntaxhighlight>

=== Multiple Adapters ===
<syntaxhighlight lang="python">
# First adapter
config1 = VBLoRAConfig(
    r=4,
    num_vectors=256,
    vector_length=256,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(base_model, config1, adapter_name="task1")

# Second adapter with different parameters
config2 = VBLoRAConfig(
    r=8,
    num_vectors=512,
    vector_length=128,
    target_modules=["q_proj", "v_proj"]
)
model.add_adapter("task2", config2)

# Switch between adapters
model.set_adapter("task1")
output1 = model(input_ids)

model.set_adapter("task2")
output2 = model(input_ids)
</syntaxhighlight>

=== Checking Parameter Efficiency ===
<syntaxhighlight lang="python">
# Get parameter counts
vblora_params, other_params = model.get_nb_savable_parameters()

print(f"VB-LoRA params (float32-equivalent): {vblora_params:,}")
print(f"Other trainable params: {other_params:,}")
print(f"Total params to-be-saved: {vblora_params + other_params:,}")

# Compare with base model
base_params = sum(p.numel() for p in base_model.parameters())
print(f"Compression ratio: {base_params / (vblora_params + other_params):.2f}x")
</syntaxhighlight>

=== Target Specific Layers ===
<syntaxhighlight lang="python">
# Apply VB-LoRA to specific layers only
config = VBLoRAConfig(
    r=4,
    num_vectors=256,
    vector_length=256,
    target_modules=["q_proj", "v_proj"],
    layers_to_transform=[0, 1, 2, 3],  # First 4 layers
    layers_pattern="layers"
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Merge and Save Full Model ===
<syntaxhighlight lang="python">
# Merge VB-LoRA into base model
model = model.merge_and_unload()

# Save as standard model
model.save_pretrained("./merged_vblora_model")
tokenizer.save_pretrained("./merged_vblora_model")
</syntaxhighlight>

=== Low CPU Memory Usage ===
<syntaxhighlight lang="python">
# Create model with low memory usage
model = get_peft_model(
    base_model,
    config,
    adapter_name="default",
    low_cpu_mem_usage=True  # Create empty weights on meta device
)
</syntaxhighlight>

=== Inference Example ===
<syntaxhighlight lang="python">
import torch
from peft import PeftModel

# Load model
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
model = PeftModel.from_pretrained(base_model, "./vblora_adapter")
model.eval()

# Generate
with torch.no_grad():
    input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids
    outputs = model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=1
    )

print(tokenizer.decode(outputs[0]))
</syntaxhighlight>

=== Custom Target Modules with Regex ===
<syntaxhighlight lang="python">
# Use regex for complex targeting
config = VBLoRAConfig(
    r=4,
    num_vectors=256,
    vector_length=256,
    target_modules=r".*decoder.*(self_attn|encoder_attn).*(q|k|v)_proj$",
    exclude_modules=["lm_head"]
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Comparing Storage Modes ===
<syntaxhighlight lang="python">
# Mode 1: Full weights (can resume training)
config_full = VBLoRAConfig(
    r=4, num_vectors=256, vector_length=256,
    save_only_topk_weights=False
)
model_full = get_peft_model(base_model, config_full)
full_params, _ = model_full.get_nb_savable_parameters()

# Mode 2: Top-k only (inference only)
config_topk = VBLoRAConfig(
    r=4, num_vectors=256, vector_length=256,
    save_only_topk_weights=True
)
model_topk = get_peft_model(base_model, config_topk)
topk_params, _ = model_topk.get_nb_savable_parameters()

print(f"Full mode: {full_params:,} params")
print(f"Top-k mode: {topk_params:,} params")
print(f"Storage reduction: {full_params / topk_params:.2f}x")
</syntaxhighlight>

== Related Pages ==
* [[huggingface_peft_VBLoRAConfig|VBLoRAConfig]] - Configuration class for VB-LoRA
* [[huggingface_peft_VBLoRALayer|VBLoRALayer]] - Layer implementation for VB-LoRA
* [[huggingface_peft_LoraModel|LoraModel]] - Model class for standard LoRA
* [[Low-Rank Adaptation]]
* [[Parameter-Efficient Fine-Tuning (PEFT)]]
* [[Model Compression]]

[[Category:Machine Learning]]
[[Category:PEFT]]
[[Category:Model Adaptation]]
[[Category:HuggingFace]]
