{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Parameter-Efficient Fine-Tuning]], [[domain::Low-Rank Adaptation]], [[domain::Model Adaptation]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
GraloraModel creates a Vector-based Random Matrix Adaptation (GraLoRA) model from a pretrained transformers model, implementing block-structured low-rank adaptation for improved parameter efficiency and expressivity.

=== Description ===
GraloraModel is a parameter-efficient fine-tuning method that extends LoRA by partitioning low-rank matrices into multiple subblocks. This block structure multiplies the expressivity by the number of subblocks (gralora_k) while maintaining the same parameter count as standard LoRA. The model supports both pure GraLoRA and Hybrid GraLoRA (combining GraLoRA with vanilla LoRA).

The implementation extends BaseTuner and creates GraloraLayer instances to wrap target modules (Linear and Conv1D layers). It uses the same target module mapping as LoRA, making it compatible with architectures that support LoRA fine-tuning.

=== Usage ===
Use GraloraModel when you need:
* More expressive low-rank adaptation than standard LoRA
* Parameter-efficient fine-tuning with block-structured decomposition
* Hybrid approaches combining different adaptation strategies
* Compatible replacement for LoRA with improved performance
* Fine-tuning large language models with limited memory

== Code Reference ==
=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/gralora/model.py src/peft/tuners/gralora/model.py]
* '''Lines:''' 30-143

=== Signature ===
<syntaxhighlight lang="python">
class GraloraModel(BaseTuner):
    def __init__(
        self,
        model: torch.nn.Module,
        config: GraloraConfig,
        adapter_name: str = "default"
    ):
        """
        Args:
            model: The pretrained model to be adapted
            config: The configuration of the GraLoRA model
            adapter_name: The name of the adapter, defaults to "default"
        """

    def _create_and_replace(
        self,
        gralora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        """Create and replace target modules with GraLoRA layers"""

    @staticmethod
    def _create_new_module(gralora_config, adapter_name, target, module_name, **kwargs):
        """Create new GraLoRA module based on target type"""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import GraloraConfig, get_peft_model
# or directly
from peft import GraloraModel
</syntaxhighlight>

== I/O Contract ==
=== Input Parameters ===
{| class="wikitable"
! Parameter !! Type !! Description !! Default
|-
| model || torch.nn.Module || The pretrained transformers model to be adapted || Required
|-
| config || GraloraConfig || Configuration object with GraLoRA parameters || Required
|-
| adapter_name || str || Name identifier for the adapter || "default"
|}

=== Configuration Parameters (via GraloraConfig) ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| r || int || GraLoRA rank (must be divisible by gralora_k)
|-
| hybrid_r || int || Vanilla LoRA rank for hybrid mode
|-
| alpha || int || Scaling factor
|-
| gralora_dropout || float || Dropout probability
|-
| gralora_k || int || Number of subblocks
|-
| target_modules || Union[list[str], str] || Modules to target for adaptation
|-
| fan_in_fan_out || bool || Whether layer stores weights as (fan_in, fan_out)
|-
| init_weights || bool || Whether to initialize weights
|}

=== Output ===
{| class="wikitable"
! Return Type !! Description
|-
| torch.nn.Module || The adapted model with GraLoRA layers injected
|}

=== Attributes ===
{| class="wikitable"
! Attribute !! Type !! Description
|-
| prefix || str || "gralora_" - prefix for GraLoRA parameters
|-
| tuner_layer_cls || type || GraloraLayer class
|-
| target_module_mapping || dict || Mapping of model architectures to default target modules (uses LoRA mapping)
|}

== Usage Examples ==
=== Basic GraLoRA Model ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import GraloraConfig, get_peft_model

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# Configure GraLoRA
config = GraloraConfig(
    r=32,
    gralora_k=2,
    target_modules=["q_proj", "v_proj"],
    alpha=64
)

# Create GraLoRA model
model = get_peft_model(base_model, config)

# Print trainable parameters
model.print_trainable_parameters()
# Output: trainable params: xxx || all params: xxx || trainable%: x.xx%
</syntaxhighlight>

=== Training with GraLoRA ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import GraloraConfig, get_peft_model

# Setup
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

# Apply GraLoRA
config = GraloraConfig(
    r=64,
    gralora_k=4,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    alpha=128,
    gralora_dropout=0.1
)
model = get_peft_model(model, config)

# Training
training_args = TrainingArguments(
    output_dir="./gralora_model",
    per_device_train_batch_size=8,
    learning_rate=3e-4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()
</syntaxhighlight>

=== Hybrid GraLoRA Model ===
<syntaxhighlight lang="python">
# Combine GraLoRA with vanilla LoRA
config = GraloraConfig(
    r=32,  # GraLoRA rank
    hybrid_r=8,  # Vanilla LoRA rank
    gralora_k=2,
    target_modules=["q_proj", "v_proj"],
    alpha=80
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Multiple Adapters ===
<syntaxhighlight lang="python">
# Create model with first adapter
config1 = GraloraConfig(r=32, gralora_k=2, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, config1, adapter_name="task1")

# Add second adapter
config2 = GraloraConfig(r=64, gralora_k=4, target_modules=["q_proj", "v_proj"])
model.add_adapter("task2", config2)

# Switch between adapters
model.set_adapter("task1")
output1 = model(**inputs)

model.set_adapter("task2")
output2 = model(**inputs)
</syntaxhighlight>

=== Full Example with Inference ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import GraloraConfig, get_peft_model, PeftModel

# Training phase
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
config = GraloraConfig(
    r=32,
    gralora_k=2,
    target_modules=["q_proj", "v_proj"],
    alpha=64
)
model = get_peft_model(base_model, config)

# ... training code ...
model.save_pretrained("./gralora_model")

# Inference phase
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
model = PeftModel.from_pretrained(base_model, "./gralora_model")

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
</syntaxhighlight>

=== Target All Linear Layers ===
<syntaxhighlight lang="python">
# Apply GraLoRA to all linear layers
config = GraloraConfig(
    r=32,
    gralora_k=2,
    target_modules="all-linear",
    alpha=64
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Conv1D Layer Support (GPT-2) ===
<syntaxhighlight lang="python">
# For models using Conv1D like GPT-2
base_model = AutoModelForCausalLM.from_pretrained("gpt2")

config = GraloraConfig(
    r=32,
    gralora_k=2,
    target_modules=["c_attn", "c_proj"],
    fan_in_fan_out=True,  # Required for Conv1D
    alpha=64
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

== Technical Details ==
=== Supported Layer Types ===
* torch.nn.Linear
* transformers.pytorch_utils.Conv1D

=== Block Structure ===
GraLoRA partitions the rank r into gralora_k subblocks:
* Subblock rank: r / gralora_k
* Total expressivity: multiplied by gralora_k
* Parameter count: same as LoRA with rank r

=== Hybrid Mode ===
When hybrid_r > 0:
* GraLoRA parameters: r
* Standard LoRA parameters: hybrid_r
* Total trainable rank: r + hybrid_r

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
