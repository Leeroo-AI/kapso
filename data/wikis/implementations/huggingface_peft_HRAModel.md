{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|Householder Reflection Adaptation|https://huggingface.co/papers/2405.17484]]
|-
! Domains
| [[domain::NLP]], [[domain::Computer Vision]], [[domain::PEFT]], [[domain::Parameter-Efficient Fine-Tuning]], [[domain::Orthogonal Transformation]], [[domain::Model Adaptation]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
HRAModel creates a Householder Reflection Adaptation (HRA) model from a pretrained model, using orthogonal transformations based on Householder reflections for parameter-efficient fine-tuning.

=== Description ===
HRAModel implements Householder Reflection Adaptation, a PEFT method that uses Householder reflections to construct orthogonal transformations for model adaptation. Unlike methods based on low-rank decomposition (like LoRA), HRA maintains orthogonality through reflection-based transformations, which can provide different inductive biases and potentially better performance for certain tasks.

The implementation extends BaseTuner and supports both Linear and Conv2d layers, making it applicable to both language models and vision models (e.g., Stable Diffusion). The model supports optional Gram-Schmidt orthogonalization for improved numerical stability.

=== Usage ===
Use HRAModel when you need:
* Orthogonal transformation-based model adaptation
* Parameter-efficient fine-tuning with different inductive biases than LoRA
* Fine-tuning of both NLP and vision models (Linear and Conv2d support)
* Better preservation of model properties through orthogonal constraints
* Stable adaptation through optional Gram-Schmidt orthogonalization

== Code Reference ==
=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/hra/model.py src/peft/tuners/hra/model.py]
* '''Lines:''' 24-132

=== Signature ===
<syntaxhighlight lang="python">
class HRAModel(BaseTuner):
    def __init__(
        self,
        model: torch.nn.Module,
        config: HRAConfig,
        adapter_name: str = "default",
        low_cpu_mem_usage: bool = False
    ):
        """
        Args:
            model: The model to which adapter tuner layers will be attached
            config: The configuration of the HRA model
            adapter_name: The name of the adapter, defaults to "default"
            low_cpu_mem_usage: Create empty adapter weights on meta device
        """

    def _create_and_replace(
        self,
        hra_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        """Create and replace target modules with HRA layers"""

    @staticmethod
    def _create_new_module(hra_config, adapter_name, target, **kwargs):
        """Create new HRA module based on target type"""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import HRAModel, HRAConfig
from peft import get_peft_model
</syntaxhighlight>

== I/O Contract ==
=== Input Parameters ===
{| class="wikitable"
! Parameter !! Type !! Description !! Default
|-
| model || torch.nn.Module || The model to be adapted (language or vision) || Required
|-
| config || HRAConfig || Configuration object with HRA parameters || Required
|-
| adapter_name || str || Name identifier for the adapter || "default"
|-
| low_cpu_mem_usage || bool || Whether to create empty weights on meta device || False
|}

=== Configuration Parameters (via HRAConfig) ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| r || int || Rank of HRA (number of Householder reflections)
|-
| apply_GS || bool || Whether to apply Gram-Schmidt orthogonalization
|-
| target_modules || Union[list[str], str] || Modules to target for adaptation
|-
| exclude_modules || Union[list[str], str] || Modules to exclude from adaptation
|-
| init_weights || bool || Whether to initialize HRA weights
|-
| layers_to_transform || Union[list[int], int] || Specific layer indices to transform
|-
| layers_pattern || Union[list[str], str] || Layer pattern name for targeting
|}

=== Output ===
{| class="wikitable"
! Return Type !! Description
|-
| torch.nn.Module || The adapted model with HRA layers injected
|}

=== Attributes ===
{| class="wikitable"
! Attribute !! Type !! Description
|-
| prefix || str || "hra_" - prefix for HRA parameters
|-
| tuner_layer_cls || type || HRALayer class
|-
| target_module_mapping || dict || Mapping of model architectures to default target modules
|}

== Usage Examples ==
=== Basic Language Model Fine-tuning ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import HRAConfig, get_peft_model

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# Configure HRA
config = HRAConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    apply_GS=True
)

# Create HRA model
model = get_peft_model(base_model, config)

# Print trainable parameters
model.print_trainable_parameters()
</syntaxhighlight>

=== Stable Diffusion Fine-tuning ===
<syntaxhighlight lang="python">
from diffusers import StableDiffusionPipeline
from peft import HRAModel, HRAConfig

# Configuration for text encoder
config_te = HRAConfig(
    r=8,
    target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
    init_weights=True,
    apply_GS=True
)

# Configuration for UNet
config_unet = HRAConfig(
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
    apply_GS=True
)

# Load model
model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# Apply HRA
model.text_encoder = HRAModel(model.text_encoder, config_te, "default")
model.unet = HRAModel(model.unet, config_unet, "default")
</syntaxhighlight>

=== Training with HRA ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import HRAConfig, get_peft_model

# Setup model
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# Apply HRA with Gram-Schmidt
config = HRAConfig(
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    apply_GS=True,
    init_weights=True
)
model = get_peft_model(model, config)

# Training
training_args = TrainingArguments(
    output_dir="./hra_model",
    per_device_train_batch_size=8,
    learning_rate=3e-4,
    num_train_epochs=3
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

# Save
model.save_pretrained("./hra_model")
</syntaxhighlight>

=== Multiple Adapters ===
<syntaxhighlight lang="python">
# Create model with first adapter
config1 = HRAConfig(r=8, target_modules=["q_proj", "v_proj"], apply_GS=True)
model = get_peft_model(base_model, config1, adapter_name="task1")

# Add second adapter
config2 = HRAConfig(r=16, target_modules=["q_proj", "v_proj"], apply_GS=True)
model.add_adapter("task2", config2)

# Switch between adapters
model.set_adapter("task1")
output1 = model(**inputs)

model.set_adapter("task2")
output2 = model(**inputs)
</syntaxhighlight>

=== Excluding Specific Modules ===
<syntaxhighlight lang="python">
# Target all linear layers except specific ones
config = HRAConfig(
    r=8,
    target_modules="all-linear",
    exclude_modules=["lm_head"],
    apply_GS=True
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Layer-Specific Fine-tuning ===
<syntaxhighlight lang="python">
# Only transform first 4 layers
config = HRAConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    layers_to_transform=[0, 1, 2, 3],
    layers_pattern="layers",
    apply_GS=True
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Loading Pretrained HRA Model ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# Load HRA adapter
model = PeftModel.from_pretrained(base_model, "./hra_model")

# Inference
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
</syntaxhighlight>

=== Conv2d Support for Vision Models ===
<syntaxhighlight lang="python">
from torchvision.models import resnet50
from peft import HRAConfig, get_peft_model

# Load vision model
model = resnet50(pretrained=True)

# Configure HRA for Conv2d layers
config = HRAConfig(
    r=8,
    target_modules=["conv1", "conv2"],  # Conv2d layers
    apply_GS=True,
    init_weights=True
)

model = get_peft_model(model, config)
</syntaxhighlight>

=== Low Memory Loading ===
<syntaxhighlight lang="python">
# Create model with low CPU memory usage
config = HRAConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    apply_GS=True
)

model = HRAModel(
    base_model,
    config,
    adapter_name="default",
    low_cpu_mem_usage=True  # Speeds up loading
)
</syntaxhighlight>

== Technical Details ==
=== Householder Reflections ===
HRA constructs orthogonal transformations using Householder reflections:
* Each reflection is defined by a vector v of dimension d
* Reflection formula: H = I - 2(vv^T)/(v^Tv)
* Multiple reflections are composed: H_1 * H_2 * ... * H_r
* Maintains orthogonality by construction

=== Gram-Schmidt Orthogonalization ===
When apply_GS=True:
* Applies Gram-Schmidt process to reflection vectors
* Improves numerical stability
* Ensures strict orthogonality during training

=== Supported Layer Types ===
* '''torch.nn.Linear''' - For language models and transformers
* '''torch.nn.Conv2d''' - For vision models and CNNs

=== Parameter Count ===
For a layer with dimension d and rank r:
* Number of parameters: r * d
* Comparable to LoRA but with orthogonal constraints

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
