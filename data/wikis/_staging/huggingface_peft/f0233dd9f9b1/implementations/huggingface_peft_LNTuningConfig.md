{{Implementation
|domain=NLP,PEFT,Parameter-Efficient Fine-Tuning,Layer Normalization Tuning
|link=https://github.com/huggingface/peft
}}

== Overview ==

=== Description ===

The <code>LNTuningConfig</code> class is a configuration dataclass for Layer Normalization (LN) Tuning in PEFT. This configuration stores all parameters needed to set up a Layer Normalization tuning model, which is a parameter-efficient fine-tuning method that focuses on tuning only the LayerNorm layers of a pretrained transformer model while keeping other parameters frozen.

LN Tuning is based on the paper detailed at https://huggingface.co/papers/2312.11420. This method provides an efficient way to adapt large language models with minimal trainable parameters by selectively updating LayerNorm layers.

The configuration inherits from <code>PeftConfig</code> and sets the PEFT type to <code>PeftType.LN_TUNING</code> during initialization.

=== Usage ===

LNTuningConfig is used to define which modules in a model should be targeted for Layer Normalization tuning. It supports flexible module targeting through regex patterns and can exclude specific modules if needed. The configuration is passed to the PEFT library when creating an adapted model with LN Tuning enabled.

== Code Reference ==

=== Source Location ===
* '''Repository:''' huggingface/peft
* '''File Path:''' <code>src/peft/tuners/ln_tuning/config.py</code>
* '''Lines:''' 23-71

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class LNTuningConfig(PeftConfig):
    target_modules: Optional[Union[list[str], str]] = None
    exclude_modules: Optional[Union[list[str], str]] = None
    modules_to_save: Optional[Union[list[str], str]] = None
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.ln_tuning.config import LNTuningConfig
# or
from peft import LNTuningConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===

{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| <code>target_modules</code> || <code>Optional[Union[list[str], str]]</code> || <code>None</code> || List of module names or regex expression of the module names to replace with LNTuning. For example, '.*decoder.*' or '.*encoder.*'. If not specified, modules will be chosen according to the model architecture. If the architecture is not known, an error will be raised.
|-
| <code>exclude_modules</code> || <code>Optional[Union[list[str], str]]</code> || <code>None</code> || The names of the modules to not apply the adapter. When passing a string, a regex match will be performed. When passing a list of strings, either an exact match will be performed or it is checked if the name of the module ends with any of the passed strings.
|-
| <code>modules_to_save</code> || <code>Optional[Union[list[str], str]]</code> || <code>None</code> || List of modules to be set as trainable and saved in the final checkpoint. For example, in Sequence Classification or Token Classification tasks, the final layer 'classifier/score' are randomly initialized and need to be trainable and saved.
|}

=== Outputs ===

{| class="wikitable"
! Attribute !! Type !! Description
|-
| <code>peft_type</code> || <code>PeftType</code> || Automatically set to <code>PeftType.LN_TUNING</code> in <code>__post_init__</code> method
|-
| <code>target_modules</code> || <code>Optional[Union[list[str], str]]</code> || Configured target modules for LN tuning
|-
| <code>exclude_modules</code> || <code>Optional[Union[list[str], str]]</code> || Configured modules to exclude from LN tuning
|-
| <code>modules_to_save</code> || <code>Optional[Union[list[str], str]]</code> || Configured modules to save in checkpoint
|}

== Usage Examples ==

=== Example 1: Basic LN Tuning Configuration ===
<syntaxhighlight lang="python">
from peft import LNTuningConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

# Create LN Tuning configuration
config = LNTuningConfig(
    task_type=TaskType.CAUSAL_LM,
)

# Load and adapt model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = get_peft_model(model, config)
model.print_trainable_parameters()
</syntaxhighlight>

=== Example 2: LN Tuning with Specific Target Modules ===
<syntaxhighlight lang="python">
from peft import LNTuningConfig, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification

# Configure with specific target modules
config = LNTuningConfig(
    task_type=TaskType.SEQ_CLS,
    target_modules=[".*decoder.*"],  # Target decoder LayerNorm layers only
    modules_to_save=["classifier"],  # Save classifier layer
)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
model = get_peft_model(model, config)
</syntaxhighlight>

=== Example 3: LN Tuning with Exclusions ===
<syntaxhighlight lang="python">
from peft import LNTuningConfig, get_peft_model, TaskType

# Configure with exclusions
config = LNTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[".*layer_norm.*"],
    exclude_modules=[".*embeddings.*"],  # Exclude embedding LayerNorms
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

== Related Pages ==

* [[huggingface_peft_LNTuningLayer|LNTuningLayer]] - The layer implementation for LN Tuning
* [[huggingface_peft_LNTuningModel|LNTuningModel]] - The model class that applies LN Tuning
* [[huggingface_peft_PeftConfig|PeftConfig]] - Base configuration class
* [[huggingface_peft_LoRAConfig|LoRAConfig]] - Alternative PEFT configuration for LoRA
* [[huggingface_peft_get_peft_model|get_peft_model]] - Function to create PEFT models

[[Category:PEFT]]
[[Category:Configuration]]
[[Category:Layer Normalization]]
[[Category:Parameter-Efficient Fine-Tuning]]
