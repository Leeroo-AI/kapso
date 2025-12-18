= ShiraModel =

== Knowledge Sources ==
* [https://github.com/huggingface/peft HuggingFace PEFT Repository]
* Source: src/peft/tuners/shira/model.py

== Domains ==
* [[Natural Language Processing (NLP)]]
* [[Parameter-Efficient Fine-Tuning (PEFT)]]
* [[Transfer Learning]]
* [[Model Adaptation]]
* [[Sparse High Rank Adaptation]]

== Overview ==

=== Description ===
ShiraModel creates a Sparse High Rank Adapter (SHiRA) model from a pre-trained transformers model. It is the main model class that coordinates the injection of SHiRA adapters into target modules of the base model. The class extends BaseTuner and handles the creation, replacement, and management of SHiRA layers throughout the model architecture.

SHiRA is a parameter-efficient fine-tuning method that maintains the same parameter count as LoRA but uses sparse high-rank adaptation instead of low-rank decomposition, potentially providing better adaptation capabilities.

=== Usage ===
ShiraModel is typically instantiated through the get_peft_model() function by passing a ShiraConfig. It automatically identifies and replaces target modules in the base model with SHiRA layers, manages adapter injection, and provides methods for adapter manipulation and merging.

== Code Reference ==

=== Source Location ===
File: src/peft/tuners/shira/model.py
Lines: 29-143

=== Class Signature ===
<syntaxhighlight lang="python">
class ShiraModel(BaseTuner):
    """
    Creates Sparse High Rank Adapter (SHiRA) Model from a pretrained model.

    Args:
        model: The model to be adapted
        config: The ShiraConfig configuration
        adapter_name: The name of the adapter (defaults to "default")

    Returns:
        torch.nn.Module: The SHiRA model
    """
    prefix: str = "shira_"
    tuner_layer_cls = ShiraLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_SHIRA_TARGET_MODULES_MAPPING
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from peft.tuners.shira.model import ShiraModel
# Or via the main PEFT interface
from peft import get_peft_model, ShiraConfig
</syntaxhighlight>

== I/O Contract ==

=== Initialization Parameters ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| model || PreTrainedModel || The base model to be adapted
|-
| config || ShiraConfig || Configuration for SHiRA adaptation
|-
| adapter_name || str || Name of the adapter (default: "default")
|}

=== Class Attributes ===
{| class="wikitable"
! Attribute !! Type !! Description
|-
| prefix || str || Prefix for SHiRA parameters ("shira_")
|-
| tuner_layer_cls || Type || ShiraLayer class reference
|-
| target_module_mapping || dict || Mapping of model architectures to default target modules
|}

=== Key Methods ===
{| class="wikitable"
! Method !! Parameters !! Returns !! Description
|-
| _create_and_replace || shira_config, adapter_name, target, target_name, parent, current_key, optional_kwargs || None || Create and replace target module with SHiRA layer
|-
| _create_new_module || shira_config, adapter_name, target, kwargs || Linear || Create new SHiRA module
|}

== Usage Examples ==

=== Basic SHiRA Model Creation ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import ShiraConfig, get_peft_model

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# Create SHiRA configuration
config = ShiraConfig(r=32)

# Create SHiRA model
model = get_peft_model(base_model, config)

# Print trainable parameters
model.print_trainable_parameters()
</syntaxhighlight>

=== Training with SHiRA ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from peft import ShiraConfig, get_peft_model

# Setup
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
config = ShiraConfig(
    r=32,
    target_modules=["q_proj", "v_proj"],
    init_weights=True
)
model = get_peft_model(base_model, config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./shira_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
</syntaxhighlight>

=== Multiple Adapters ===
<syntaxhighlight lang="python">
# Create model with first adapter
config1 = ShiraConfig(r=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, config1, adapter_name="task1")

# Add second adapter
config2 = ShiraConfig(r=64, target_modules=["q_proj", "v_proj"])
model.add_adapter("task2", config2)

# Switch between adapters
model.set_adapter("task1")
output1 = model(input_ids)

model.set_adapter("task2")
output2 = model(input_ids)
</syntaxhighlight>

=== Save and Load SHiRA Model ===
<syntaxhighlight lang="python">
# Save adapter
model.save_pretrained("./shira_adapter")

# Load adapter
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
model = PeftModel.from_pretrained(base_model, "./shira_adapter")
</syntaxhighlight>

=== Merge and Save Full Model ===
<syntaxhighlight lang="python">
# Merge adapter into base model
model = model.merge_and_unload()

# Save full model
model.save_pretrained("./merged_model")
</syntaxhighlight>

=== Custom Target Modules ===
<syntaxhighlight lang="python">
# Target specific modules with regex
config = ShiraConfig(
    r=32,
    target_modules=r".*decoder.*attn.*(q|v)_proj$",
    modules_to_save=["lm_head"]
)
model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Inference with SHiRA ===
<syntaxhighlight lang="python">
import torch

# Set to evaluation mode
model.eval()

# Generate with SHiRA adapter
with torch.no_grad():
    input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=50)

print(tokenizer.decode(outputs[0]))
</syntaxhighlight>

== Related Pages ==
* [[huggingface_peft_ShiraConfig|ShiraConfig]] - Configuration class for SHiRA
* [[huggingface_peft_ShiraLayer|ShiraLayer]] - Layer implementation for SHiRA
* [[huggingface_peft_LoraModel|LoraModel]] - Model class for LoRA adapters
* [[Parameter-Efficient Fine-Tuning (PEFT)]]
* [[Transfer Learning]]

[[Category:Machine Learning]]
[[Category:PEFT]]
[[Category:Model Adaptation]]
[[Category:HuggingFace]]
