= PolyModel =

== Knowledge Sources ==
* Source Repository: https://github.com/huggingface/peft
* Paper: [https://huggingface.co/papers/2202.13914 Polytropon (Poly)]
* Paper: [https://huggingface.co/papers/2211.03831 Multi-Head Routing (MHR)]

== Domains ==
* [[NLP]]
* [[PEFT]] (Parameter-Efficient Fine-Tuning)
* [[Multi-Task Learning]]
* [[LoRA]] (Low-Rank Adaptation)
* [[Model Adaptation]]

== Overview ==

=== Description ===
PolyModel is a BaseTuner implementation that applies the Polytropon (Poly) parameter-efficient fine-tuning method to neural networks. It manages the creation, replacement, and lifecycle of Poly layers within a model, enabling multi-task learning through dynamic routing of multiple LoRA modules.

The model handles task-specific routing by registering forward hooks that inject task IDs into the computation, allowing different combinations of LoRA "skills" to be activated for different tasks.

=== Usage ===
PolyModel is typically instantiated through the <code>get_peft_model</code> function with a PolyConfig. It provides <code>forward</code> and <code>generate</code> methods that accept task IDs to enable task-specific routing during inference.

== Code Reference ==

=== Source Location ===
<code>src/peft/tuners/poly/model.py</code>

=== Signature ===
<syntaxhighlight lang="python">
class PolyModel(BaseTuner):
    prefix: str = "poly_"
    tuner_layer_cls = PolyLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_POLY_TARGET_MODULES_MAPPING

    def forward(self, *args, task_ids=None, **kwargs)
    def generate(self, *args, task_ids=None, **kwargs)
    def _create_and_replace(self, poly_config, adapter_name, target, target_name, parent, **optional_kwargs)
    @staticmethod
    def _create_new_module(poly_config, adapter_name, target, **kwargs)
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from peft.tuners.poly.model import PolyModel
# Or use through get_peft_model:
from peft import get_peft_model, PolyConfig
</syntaxhighlight>

== I/O Contract ==

=== forward Method ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| *args || tuple || - || Positional arguments passed to the base model
|-
| task_ids || Optional[torch.Tensor] || None || Tensor of task IDs for routing
|-
| **kwargs || dict || - || Keyword arguments passed to the base model
|}

'''Returns:''' Model outputs with task-specific Poly routing applied

=== generate Method ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| *args || tuple || - || Positional arguments for generation
|-
| task_ids || Optional[torch.Tensor] || None || Tensor of task IDs for routing
|-
| **kwargs || dict || - || Keyword arguments for generation
|}

'''Returns:''' Generated outputs with task-specific Poly routing

=== _create_new_module Method ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| poly_config || PolyConfig || The Poly configuration
|-
| adapter_name || str || Name of the adapter
|-
| target || nn.Module || The target module to replace
|-
| **kwargs || dict || Optional additional arguments
|}

'''Returns:''' Linear - A new Poly Linear layer

'''Raises:''' ValueError if target module type is not supported (only <code>torch.nn.Linear</code> is supported)

== Usage Examples ==

=== Basic Multi-Task Inference ===
<syntaxhighlight lang="python">
import torch
from peft import get_peft_model, PolyConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("base-model")
tokenizer = AutoTokenizer.from_pretrained("base-model")

# Configure and apply Poly
config = PolyConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    n_tasks=3,
    n_skills=4
)
peft_model = get_peft_model(model, config)

# Prepare inputs
inputs = tokenizer("Example text", return_tensors="pt")

# Forward pass with task ID
task_ids = torch.tensor([0])  # Task 0
outputs = peft_model.forward(**inputs, task_ids=task_ids)
</syntaxhighlight>

=== Text Generation with Task Routing ===
<syntaxhighlight lang="python">
import torch
from peft import get_peft_model, PolyConfig

# Configure Poly for 5 different tasks
config = PolyConfig(
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj"],
    n_tasks=5,
    n_skills=8,
    n_splits=2
)

peft_model = get_peft_model(model, config)
peft_model.eval()

# Generate text for different tasks
for task_id in range(5):
    task_ids = torch.tensor([task_id])

    inputs = tokenizer(f"Task {task_id} prompt", return_tensors="pt")

    outputs = peft_model.generate(
        **inputs,
        task_ids=task_ids,
        max_length=50
    )

    print(f"Task {task_id}: {tokenizer.decode(outputs[0])}")
</syntaxhighlight>

=== Adding Multiple Adapters ===
<syntaxhighlight lang="python">
from peft import get_peft_model, PolyConfig

# Create model with first adapter
config1 = PolyConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    n_tasks=3,
    n_skills=4
)

peft_model = get_peft_model(model, config1, adapter_name="adapter1")

# Add second adapter
config2 = PolyConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    n_tasks=5,
    n_skills=6
)

peft_model.add_adapter("adapter2", config2)

# Switch between adapters
peft_model.set_adapter("adapter1")
outputs1 = peft_model.forward(**inputs, task_ids=torch.tensor([0]))

peft_model.set_adapter("adapter2")
outputs2 = peft_model.forward(**inputs, task_ids=torch.tensor([1]))
</syntaxhighlight>

=== Context Manager for Pre-Hooks ===
<syntaxhighlight lang="python">
# The model internally uses a context manager for pre-hooks
# This is handled automatically in forward/generate methods

# Manual usage (advanced):
with peft_model._manage_pre_hooks(task_ids=torch.tensor([2])):
    # All forward passes in this context will use task_id=2
    output = peft_model.model(**inputs)
</syntaxhighlight>

== Implementation Details ==

=== Module Replacement ===
The <code>_create_and_replace</code> method handles two scenarios:
# '''Updating existing PolyLayer''': If target is already a PolyLayer, updates it with the new adapter
# '''Creating new PolyLayer''': Replaces a standard Linear module with a new Poly Linear layer

=== Pre-Hook Management ===
PolyModel uses forward pre-hooks to inject task IDs:
* Pre-hooks are registered on all Poly Linear modules
* Task IDs are passed through kwargs during forward pass
* Context manager ensures proper cleanup of hooks after execution

=== Supported Target Modules ===
Currently only <code>torch.nn.Linear</code> modules are supported for Poly transformation.

== Related Pages ==
* [[huggingface_peft_PolyConfig|PolyConfig]] - Configuration for PolyModel
* [[huggingface_peft_PolyRouter|PolyRouter]] - Routing mechanism for Poly
* [[huggingface_peft_PolyLayer|PolyLayer]] - Layer implementation
* [[huggingface_peft_BaseTuner|BaseTuner]] - Base class for tuners
* [[PEFT]] - Parameter-Efficient Fine-Tuning
* [[Multi-Task Learning]]

== Categories ==
[[Category:PEFT]]
[[Category:Model]]
[[Category:Multi-Task Learning]]
[[Category:LoRA]]
[[Category:HuggingFace]]
