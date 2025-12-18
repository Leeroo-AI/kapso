= RoadModel =

== Knowledge Sources ==
* Source Repository: https://github.com/huggingface/peft
* Paper: https://huggingface.co/papers/2409.00119

== Domains ==
* [[NLP]]
* [[PEFT]] (Parameter-Efficient Fine-Tuning)
* [[Rotation-Based Adaptation]]
* [[Model Adaptation]]
* [[Quantization]]

== Overview ==

=== Description ===
RoadModel is a BaseTuner implementation that applies the RoAd (Rotation Adaptation) parameter-efficient fine-tuning method to neural networks. It manages the creation, replacement, and lifecycle of RoAd layers within a model, enabling efficient adaptation through learned rotations applied to hidden representations.

The model supports mixed adapter batches inference through adapter-specific routing, quantization integration (8-bit, 4-bit), and torchao tensor subclasses. It uses forward pre-hooks to inject adapter names during inference for dynamic adapter selection.

=== Usage ===
RoadModel is typically instantiated through the <code>get_peft_model</code> function with a RoadConfig. It provides context managers for enabling adapter-specific inference and supports multiple adapters on the same base model.

== Code Reference ==

=== Source Location ===
<code>src/peft/tuners/road/model.py</code>

=== Signature ===
<syntaxhighlight lang="python">
class RoadModel(BaseTuner):
    prefix: str = "road_"
    tuner_layer_cls = RoadLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_ROAD_TARGET_MODULES_MAPPING

    def _create_and_replace(self, road_config, adapter_name, target, target_name, parent, current_key)
    @staticmethod
    def _create_new_module(road_config, adapter_name, target, **kwargs)
    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs)
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from peft.tuners.road.model import RoadModel
# Or use through get_peft_model:
from peft import get_peft_model, RoadConfig
</syntaxhighlight>

== I/O Contract ==

=== _create_and_replace Method ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| road_config || RoadConfig || The RoAd configuration
|-
| adapter_name || str || Name of the adapter being added
|-
| target || nn.Module || The target module to replace or update
|-
| target_name || str || Name of the target module
|-
| parent || nn.Module || Parent module containing the target
|-
| current_key || str || Current key path to the module (required, cannot be None)
|}

=== _create_new_module Method ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| road_config || RoadConfig || The RoAd configuration
|-
| adapter_name || str || Name of the adapter
|-
| target || nn.Module || The target module to replace
|-
| **kwargs || dict || Additional arguments including device_map, variant, group_size, init_weights, loaded_in_8bit, loaded_in_4bit, get_apply_tensor_subclass
|}

'''Returns:''' RoadLayer - A new RoAd layer (Linear, 8-bit, 4-bit, or torchao variant)

'''Raises:''' ValueError if target module type is not supported (only <code>torch.nn.Linear</code> is supported)

=== _enable_peft_forward_hooks Method ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| adapter_names || Optional[List[str]] || List of adapter names for mixed batch inference (passed as kwarg)
|-
| *args || tuple || Additional positional arguments
|-
| **kwargs || dict || Additional keyword arguments
|}

'''Yields:''' Context where specified adapters are active for inference

'''Raises:'''
* ValueError if used during training (only inference mode supported)
* ValueError if non-existing adapter names are provided

== Usage Examples ==

=== Basic Model Creation ===
<syntaxhighlight lang="python">
from peft import get_peft_model, RoadConfig
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure RoAd
config = RoadConfig(
    variant="road_2",
    target_modules=["q_proj", "v_proj"],
    group_size=64
)

# Create PEFT model with RoAd
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
</syntaxhighlight>

=== Adding Multiple Adapters ===
<syntaxhighlight lang="python">
from peft import get_peft_model, RoadConfig

# Create model with first adapter
config1 = RoadConfig(
    variant="road_1",
    target_modules=["q_proj", "v_proj"]
)
peft_model = get_peft_model(model, config1, adapter_name="task_a")

# Add second adapter
config2 = RoadConfig(
    variant="road_2",
    target_modules=["q_proj", "v_proj"]
)
peft_model.add_adapter("task_b", config2)

# Switch between adapters
peft_model.set_adapter("task_a")
output_a = peft_model(**inputs)

peft_model.set_adapter("task_b")
output_b = peft_model(**inputs)
</syntaxhighlight>

=== Mixed Adapter Batches Inference ===
<syntaxhighlight lang="python">
from peft import get_peft_model, RoadConfig
import torch

# Setup model with multiple adapters
peft_model = get_peft_model(model, config, adapter_name="adapter1")
peft_model.add_adapter("adapter2", config)
peft_model.add_adapter("adapter3", config)

# Prepare batch with different adapters per sample
inputs = tokenizer(
    ["Sample 1", "Sample 2", "Sample 3"],
    return_tensors="pt",
    padding=True
)

# Must be in eval mode
peft_model.eval()

# Specify which adapter for each sample
adapter_names = ["adapter1", "adapter2", "__base__"]  # "__base__" = no adapter

with torch.no_grad():
    outputs = peft_model(**inputs, adapter_names=adapter_names)

# Each sample uses its specified adapter
</syntaxhighlight>

=== Quantized Model with RoAd ===
<syntaxhighlight lang="python">
from peft import get_peft_model, RoadConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Load model in 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config
)

# Apply RoAd to quantized model
config = RoadConfig(
    variant="road_1",
    target_modules=["q_proj", "v_proj"]
)

peft_model = get_peft_model(model, config)
# RoadModel automatically creates quantized RoAd layers
</syntaxhighlight>

=== Context Manager for Forward Hooks ===
<syntaxhighlight lang="python">
from peft import get_peft_model, RoadConfig

peft_model = get_peft_model(model, config)
peft_model.eval()

# Use context manager for adapter-specific inference
with peft_model._enable_peft_forward_hooks(adapter_names=["task_a"]):
    # All forward passes use "task_a" adapter
    output = peft_model(**inputs)

# Outside context, default adapter is used
output_default = peft_model(**inputs)
</syntaxhighlight>

=== Training and Inference Workflow ===
<syntaxhighlight lang="python">
from peft import get_peft_model, RoadConfig
from transformers import Trainer, TrainingArguments

# Setup
config = RoadConfig(variant="road_2", target_modules=["q_proj", "v_proj"])
peft_model = get_peft_model(model, config)

# Training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

# Save adapter
peft_model.save_pretrained("./road_adapter")

# Inference
peft_model.eval()
with torch.no_grad():
    outputs = peft_model.generate(**inputs, max_length=50)
</syntaxhighlight>

=== Updating Existing RoadLayer ===
<syntaxhighlight lang="python">
# When adding an adapter to a model that already has RoAd layers,
# _create_and_replace updates existing layers instead of replacing

config1 = RoadConfig(variant="road_1", target_modules=["q_proj"])
peft_model = get_peft_model(model, config1, adapter_name="adapter1")

# Adding another adapter updates existing RoadLayers
config2 = RoadConfig(variant="road_2", target_modules=["q_proj"])
peft_model.add_adapter("adapter2", config2)

# The RoadLayer at "q_proj" now has both adapters
# and can switch between them
</syntaxhighlight>

=== Device Map Support ===
<syntaxhighlight lang="python">
from peft import get_peft_model, RoadConfig
from transformers import AutoModelForCausalLM

# Load model with device map for multi-GPU
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    device_map="auto"
)

# RoadModel respects device map
config = RoadConfig(
    variant="road_2",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

peft_model = get_peft_model(model, config)
# RoAd layers are created on appropriate devices
</syntaxhighlight>

== Implementation Details ==

=== Module Dispatching ===
<code>_create_new_module</code> uses a dispatcher pattern to handle different layer types:
# '''BitsAndBytes 8-bit''': <code>dispatch_bnb_8bit</code>
# '''BitsAndBytes 4-bit''': <code>dispatch_bnb_4bit</code>
# '''Default (torch.nn.Linear)''': <code>dispatch_default</code>

First matching dispatcher wins.

=== Adapter Name Injection ===
During mixed adapter inference:
# Pre-hooks are registered on all RoadLayers
# <code>_adapter_names_pre_forward_hook</code> injects <code>adapter_names</code> into kwargs
# RoadLayer uses adapter names to select appropriate parameters
# Hooks are cleaned up after context exit

=== Quantization Integration ===
RoadModel detects quantization through model attributes:
* <code>loaded_in_8bit</code>: Uses 8-bit RoAd layers
* <code>loaded_in_4bit</code>: Uses 4-bit RoAd layers
* Extracts <code>get_apply_tensor_subclass</code> for torchao support

=== Active Adapter Management ===
* New adapters not in <code>active_adapters</code> are created with <code>requires_grad=False</code>
* Call <code>set_adapter()</code> or <code>enable_adapter()</code> to make them trainable

=== Validation ===
The context manager validates:
* Not in training mode (mixed adapters only for inference)
* All specified adapter names exist in at least one layer
* Uses "__base__" for no adapter on specific samples

== Supported Target Modules ==

Currently only <code>torch.nn.Linear</code> modules are supported for RoAd transformation.

Quantized variants supported:
* <code>bnb.nn.Linear8bitLt</code> (if BitsAndBytes available)
* <code>bnb.nn.Linear4bit</code> (if BitsAndBytes 4-bit available)

== Related Pages ==
* [[huggingface_peft_RoadConfig|RoadConfig]] - Configuration for RoadModel
* [[huggingface_peft_RoadLayer|RoadLayer]] - Layer implementation
* [[huggingface_peft_RoadQuantized|RoadQuantized]] - Quantized RoAd layers
* [[huggingface_peft_BaseTuner|BaseTuner]] - Base class for tuners
* [[PEFT]] - Parameter-Efficient Fine-Tuning
* [[Quantization]]

== Categories ==
[[Category:PEFT]]
[[Category:Model]]
[[Category:Rotation-Based Methods]]
[[Category:Quantization]]
[[Category:HuggingFace]]
