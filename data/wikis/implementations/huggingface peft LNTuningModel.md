{{Implementation
|domain=NLP,PEFT,Parameter-Efficient Fine-Tuning,Layer Normalization Tuning,Transformers
|link=https://github.com/huggingface/peft
}}

== Overview ==

=== Description ===

The <code>LNTuningModel</code> class creates LayerNorm tuning from a pretrained transformer model. This class is part of the PEFT library and implements the Layer Normalization tuning method described in detail at https://huggingface.co/papers/2312.11420.

LNTuningModel inherits from <code>BaseTuner</code> and provides a parameter-efficient fine-tuning approach by creating trainable copies of LayerNorm layers while keeping the rest of the model frozen. This method significantly reduces the number of trainable parameters while maintaining model performance.

Key features include:
* Automatic target module identification based on model architecture
* Support for multiple adapters
* Adapter creation and replacement mechanism
* Specialized unloading and merging capabilities
* Integration with the broader PEFT ecosystem

=== Usage ===

LNTuningModel is typically instantiated through the <code>get_peft_model</code> function with a LNTuningConfig. Users don't usually instantiate this class directly. The model automatically identifies and wraps appropriate LayerNorm modules in the base model with LNTuningLayer instances.

== Code Reference ==

=== Source Location ===
* '''Repository:''' huggingface/peft
* '''File Path:''' <code>src/peft/tuners/ln_tuning/model.py</code>
* '''Lines:''' 28-133

=== Signature ===
<syntaxhighlight lang="python">
class LNTuningModel(BaseTuner):
    prefix: str = "ln_tuning_"
    tuner_layer_cls = LNTuningLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        peft_config: PeftConfig,
        adapter_name: str,
        target: Module,
        target_name: str,
        parent: Module,
        current_key: str,
    ) -> None

    def _create_new_module(
        self,
        peft_config: PeftConfig,
        target: Module,
        adapter_name: str,
    ) -> Module

    def _unloading_checks(self, adapter_names: Optional[list[str]])

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    )

    def _cast_adapter_dtype(self, adapter_name: str, autocast_adapter_dtype: bool = True) -> None
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.ln_tuning.model import LNTuningModel
# or use via get_peft_model
from peft import get_peft_model, LNTuningConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===

'''Constructor (via get_peft_model):'''
{| class="wikitable"
! Parameter !! Type !! Description
|-
| <code>model</code> || <code>torch.nn.Module</code> || The pretrained model to be adapted with LN tuning
|-
| <code>config</code> || <code>LNTuningConfig</code> || The configuration for LN tuning
|-
| <code>adapter_name</code> || <code>str</code> || The name of the adapter (default: "default")
|-
| <code>low_cpu_mem_usage</code> || <code>bool</code> || No effect on LN tuning; exists for API consistency
|}

'''Method: _create_and_replace'''
{| class="wikitable"
! Parameter !! Type !! Description
|-
| <code>peft_config</code> || <code>PeftConfig</code> || Configuration for the PEFT method
|-
| <code>adapter_name</code> || <code>str</code> || Name of the adapter to create
|-
| <code>target</code> || <code>Module</code> || Target module to replace
|-
| <code>target_name</code> || <code>str</code> || Name of the target module
|-
| <code>parent</code> || <code>Module</code> || Parent module containing the target
|-
| <code>current_key</code> || <code>str</code> || Key path to the current module
|}

'''Method: _unload_and_optionally_merge'''
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| <code>merge</code> || <code>bool</code> || <code>True</code> || Whether to merge adapters before unloading
|-
| <code>progressbar</code> || <code>bool</code> || <code>False</code> || Whether to show a progress bar
|-
| <code>safe_merge</code> || <code>bool</code> || <code>False</code> || Safe merge flag (not used in LN tuning)
|-
| <code>adapter_names</code> || <code>Optional[list[str]]</code> || <code>None</code> || Specific adapters to unload
|}

=== Outputs ===

'''Attributes:'''
{| class="wikitable"
! Attribute !! Type !! Description
|-
| <code>model</code> || <code>PreTrainedModel</code> || The adapted model with LN tuning layers
|-
| <code>peft_config</code> || <code>LNTuningConfig</code> || The configuration used for adaptation
|-
| <code>prefix</code> || <code>str</code> || Prefix for adapter parameters ("ln_tuning_")
|-
| <code>tuner_layer_cls</code> || <code>type</code> || The layer class used (LNTuningLayer)
|}

'''Method Returns:'''
{| class="wikitable"
! Method !! Return Type !! Description
|-
| <code>_create_new_module</code> || <code>Module</code> || New LNTuningLayer wrapping the target
|-
| <code>_unload_and_optionally_merge</code> || <code>torch.nn.Module</code> || The base model with adapters unloaded
|}

== Usage Examples ==

=== Example 1: Basic LN Tuning Model Creation ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import get_peft_model, TaskType, LNTuningConfig

# Create configuration
peft_config = LNTuningConfig(
    task_type=TaskType.CAUSAL_LM,
)

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Apply LN tuning
model = get_peft_model(model, peft_config)

# Check trainable parameters
model.print_trainable_parameters()
# Output: trainable params: X || all params: Y || trainable%: Z
</syntaxhighlight>

=== Example 2: LN Tuning with Custom Target Modules ===
<syntaxhighlight lang="python">
from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LNTuningConfig, TaskType

# Configure with specific targets
config = LNTuningConfig(
    task_type=TaskType.SEQ_CLS,
    target_modules=[".*layer_norm.*"],  # Target all LayerNorm modules
    modules_to_save=["classifier"],  # Save classifier weights
)

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Apply LN tuning
model = get_peft_model(model, config)

# Train the model
# ... training loop ...
</syntaxhighlight>

=== Example 3: Multi-Adapter LN Tuning ===
<syntaxhighlight lang="python">
from peft import get_peft_model, LNTuningConfig, TaskType

# Create base model with first adapter
config1 = LNTuningConfig(
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(base_model, config1, adapter_name="adapter1")

# Add second adapter
config2 = LNTuningConfig(
    task_type=TaskType.CAUSAL_LM,
)
model.add_adapter("adapter2", config2)

# Switch between adapters
model.set_adapter("adapter1")
output1 = model(input_ids)

model.set_adapter("adapter2")
output2 = model(input_ids)
</syntaxhighlight>

=== Example 4: Unloading and Merging Adapters ===
<syntaxhighlight lang="python">
from peft import get_peft_model, LNTuningConfig, TaskType

# Create and train LN tuning model
config = LNTuningConfig(task_type=TaskType.CAUSAL_LM)
model = get_peft_model(base_model, config)

# ... training ...

# Merge and unload adapter
model = model.merge_and_unload()

# Now model is the base model with adapter merged
# Can be saved as a standard model
model.save_pretrained("./merged_model")
</syntaxhighlight>

=== Example 5: LN Tuning for Text Classification ===
<syntaxhighlight lang="python">
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LNTuningConfig, TaskType
import torch

# Load model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Apply LN tuning
config = LNTuningConfig(
    task_type=TaskType.SEQ_CLS,
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)

# Training
model.train()
inputs = tokenizer("This is a test", return_tensors="pt")
outputs = model(**inputs, labels=torch.tensor([1]))
loss = outputs.loss

print(f"Loss: {loss.item()}")
model.print_trainable_parameters()
</syntaxhighlight>

== Implementation Details ==

=== Target Module Mapping ===
LNTuningModel uses <code>TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING</code> to automatically identify appropriate LayerNorm modules for different model architectures. This mapping ensures the correct modules are targeted without manual specification.

=== Adapter Creation Process ===
<syntaxhighlight lang="python">
# Simplified flow:
1. Identify target modules based on config and architecture
2. For each target module:
   - Create LNTuningLayer wrapping the original module
   - Replace original module with LNTuningLayer
   - Set gradient requirements based on adapter state
</syntaxhighlight>

=== Unloading Mechanism ===
The <code>_unload_and_optionally_merge</code> method:
* Checks for <code>modules_to_save</code> conflicts with multiple adapters
* Optionally merges adapters before unloading
* Extracts base layers from LNTuningLayer wrappers
* Returns the original model structure

=== Dtype Handling ===
The <code>_cast_adapter_dtype</code> method is overridden to do nothing, as LN Tuning creates copies of original layers rather than adding new adapter parameters. This prevents unwanted dtype conversions.

== Related Pages ==

* [[huggingface_peft_LNTuningConfig|LNTuningConfig]] - Configuration for LN Tuning
* [[huggingface_peft_LNTuningLayer|LNTuningLayer]] - Layer implementation used by this model
* [[huggingface_peft_BaseTuner|BaseTuner]] - Base class for PEFT tuners
* [[huggingface_peft_get_peft_model|get_peft_model]] - Function to create PEFT models
* [[huggingface_peft_LoraModel|LoraModel]] - Alternative PEFT model using LoRA
* [[huggingface_peft_PeftModel|PeftModel]] - Base PEFT model wrapper

[[Category:PEFT]]
[[Category:Model]]
[[Category:Layer Normalization]]
[[Category:Parameter-Efficient Fine-Tuning]]
[[Category:Transformers]]
