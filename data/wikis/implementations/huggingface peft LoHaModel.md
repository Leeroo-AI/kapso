{{Implementation
|domain=NLP,Computer Vision,PEFT,Parameter-Efficient Fine-Tuning,LoHa,Low-Rank Adaptation,Diffusion Models
|link=https://github.com/huggingface/peft
}}

== Overview ==

=== Description ===

The <code>LoHaModel</code> class creates Low-Rank Hadamard Product (LoHa) models from pretrained models. This class implements a parameter-efficient fine-tuning method that uses Hadamard products combined with low-rank matrix decomposition to adapt neural networks with minimal trainable parameters.

LoHaModel inherits from <code>LycorisTuner</code> and supports various layer types including Linear, Conv1d, and Conv2d. The method is partially described in https://huggingface.co/papers/2108.06098, and the current implementation heavily borrows from the LyCORIS repository (https://github.com/KohakuBlueleaf/LyCORIS).

Key features include:
* Support for Linear, Conv1d, and Conv2d layers
* Automatic target module identification based on model architecture
* Layer-specific rank and alpha configuration through patterns
* Efficient decomposition for convolutional layers
* Compatible with diffusion models, vision transformers, and language models

=== Usage ===

LoHaModel is typically instantiated through the PEFT library's standard interface or directly for more control. It's particularly popular for fine-tuning Stable Diffusion models and other large vision or language models where parameter efficiency is crucial.

== Code Reference ==

=== Source Location ===
* '''Repository:''' huggingface/peft
* '''File Path:''' <code>src/peft/tuners/loha/model.py</code>
* '''Lines:''' 27-117

=== Signature ===
<syntaxhighlight lang="python">
class LoHaModel(LycorisTuner):
    prefix: str = "hada_"
    tuner_layer_cls = LoHaLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LOHA_TARGET_MODULES_MAPPING
    layers_mapping: dict[type[torch.nn.Module], type[LoHaLayer]] = {
        torch.nn.Conv2d: Conv2d,
        torch.nn.Conv1d: Conv1d,
        torch.nn.Linear: Linear,
    }

    def _create_and_replace(
        self,
        config: LycorisConfig,
        adapter_name: str,
        target: Union[LoHaLayer, nn.Module],
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.loha.model import LoHaModel
# or
from peft import LoHaModel
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===

'''Constructor:'''
{| class="wikitable"
! Parameter !! Type !! Description
|-
| <code>model</code> || <code>torch.nn.Module</code> || The model to which the adapter tuner layers will be attached
|-
| <code>config</code> || <code>LoHaConfig</code> || The configuration of the LoHa model
|-
| <code>adapter_name</code> || <code>str</code> || The name of the adapter (default: "default")
|-
| <code>low_cpu_mem_usage</code> || <code>bool</code> || Create empty adapter weights on meta device for faster loading (default: False)
|}

'''Method: _create_and_replace'''
{| class="wikitable"
! Parameter !! Type !! Description
|-
| <code>config</code> || <code>LycorisConfig</code> || Configuration for the adapter
|-
| <code>adapter_name</code> || <code>str</code> || Name of the adapter to create
|-
| <code>target</code> || <code>Union[LoHaLayer, nn.Module]</code> || Target module to replace or update
|-
| <code>target_name</code> || <code>str</code> || Name of the target module
|-
| <code>parent</code> || <code>nn.Module</code> || Parent module containing the target
|-
| <code>current_key</code> || <code>str</code> || Key path to the current module
|}

=== Outputs ===

'''Attributes:'''
{| class="wikitable"
! Attribute !! Type !! Description
|-
| <code>model</code> || <code>torch.nn.Module</code> || The adapted model with LoHa layers
|-
| <code>peft_config</code> || <code>LoHaConfig</code> || The configuration used for adaptation
|-
| <code>prefix</code> || <code>str</code> || Parameter prefix for LoHa adapters ("hada_")
|-
| <code>tuner_layer_cls</code> || <code>type</code> || The base layer class (LoHaLayer)
|-
| <code>layers_mapping</code> || <code>dict</code> || Mapping from PyTorch layer types to LoHa layer types
|}

'''Returns:'''
{| class="wikitable"
! Type !! Description
|-
| <code>torch.nn.Module</code> || The LoHa-adapted model ready for training or inference
|}

== Usage Examples ==

=== Example 1: Basic LoHa Model for Language Models ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import LoHaModel, LoHaConfig

# Configure LoHa
config = LoHaConfig(
    r=8,
    alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    rank_dropout=0.0,
    module_dropout=0.0,
    init_weights=True,
)

# Load and adapt model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = LoHaModel(model, config, "default")

# Check trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable: {trainable_params} / {total_params} ({100*trainable_params/total_params:.2f}%)")
</syntaxhighlight>

=== Example 2: LoHa for Stable Diffusion Text Encoder and UNet ===
<syntaxhighlight lang="python">
from diffusers import StableDiffusionPipeline
from peft import LoHaModel, LoHaConfig

# Configuration for text encoder
config_te = LoHaConfig(
    r=8,
    alpha=32,
    target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
    rank_dropout=0.0,
    module_dropout=0.0,
    init_weights=True,
)

# Configuration for UNet with effective conv2d
config_unet = LoHaConfig(
    r=8,
    alpha=32,
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
    rank_dropout=0.0,
    module_dropout=0.0,
    init_weights=True,
    use_effective_conv2d=True,  # Important for conv layers
)

# Load pipeline and adapt
model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
model.text_encoder = LoHaModel(model.text_encoder, config_te, "default")
model.unet = LoHaModel(model.unet, config_unet, "default")

# Now ready for fine-tuning
# ... training loop ...
</syntaxhighlight>

=== Example 3: LoHa with Layer-Specific Ranks ===
<syntaxhighlight lang="python">
from transformers import AutoModel
from peft import LoHaModel, LoHaConfig

# Configure with rank patterns
config = LoHaConfig(
    r=8,
    alpha=8,
    target_modules=["query", "key", "value"],
    rank_pattern={
        # Lower layers get higher rank
        "^model.encoder.layer.[0-5].*": 16,
        # Middle layers get default rank (8)
        # Upper layers get lower rank
        "^model.encoder.layer.(1[0-9]|2[0-9]|3[0-9]).*": 4,
    },
    alpha_pattern={
        "^model.encoder.layer.[0-5].*": 32,
    },
    init_weights=True,
)

model = AutoModel.from_pretrained("bert-base-uncased")
model = LoHaModel(model, config, "default")
</syntaxhighlight>

=== Example 4: Multi-Adapter LoHa Model ===
<syntaxhighlight lang="python">
from peft import LoHaModel, LoHaConfig

# Create base LoHa model
config1 = LoHaConfig(
    r=8,
    alpha=16,
    target_modules=["q_proj", "v_proj"],
)
model = LoHaModel(base_model, config1, adapter_name="task1")

# Add second adapter for different task
config2 = LoHaConfig(
    r=16,
    alpha=32,
    target_modules=["q_proj", "v_proj", "o_proj"],
)
# Note: Adding adapters typically done through PeftModel interface
# This example shows the conceptual structure

# Switch between adapters during inference
# model.set_adapter("task1")
# output1 = model(input_ids)

# model.set_adapter("task2")
# output2 = model(input_ids)
</syntaxhighlight>

=== Example 5: LoHa with Dropout for Regularization ===
<syntaxhighlight lang="python">
from transformers import AutoModelForSeq2SeqLM
from peft import LoHaModel, LoHaConfig

config = LoHaConfig(
    r=8,
    alpha=16,
    target_modules=["q", "k", "v", "o"],
    rank_dropout=0.1,  # 10% rank dropout during training
    module_dropout=0.05,  # 5% chance to skip entire module
    init_weights=True,
    use_effective_conv2d=False,
)

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
model = LoHaModel(model, config, "default")

# Train with dropout enabled
model.train()
# ... training loop ...

# Inference without dropout
model.eval()
# ... generate outputs ...
</syntaxhighlight>

== Implementation Details ==

=== Layer Mapping System ===
LoHaModel uses a <code>layers_mapping</code> dictionary to map PyTorch layer types to corresponding LoHa implementations:
<syntaxhighlight lang="python">
layers_mapping = {
    torch.nn.Conv2d: Conv2d,  # LoHa Conv2d layer
    torch.nn.Conv1d: Conv1d,  # LoHa Conv1d layer
    torch.nn.Linear: Linear,  # LoHa Linear layer
}
</syntaxhighlight>

This allows LoHa to automatically adapt different layer types with appropriate implementations.

=== Adapter Creation Process ===
The <code>_create_and_replace</code> method:
1. Extracts rank and alpha from patterns or uses defaults
2. Checks if target is already a LoHaLayer
   - If yes: updates the layer with new adapter
   - If no: creates new LoHaLayer wrapping the target
3. Replaces the original module in the parent

<syntaxhighlight lang="python">
# Simplified flow:
r_key = get_pattern_key(config.rank_pattern.keys(), current_key)
alpha_key = get_pattern_key(config.alpha_pattern.keys(), current_key)
kwargs["r"] = config.rank_pattern.get(r_key, config.r)
kwargs["alpha"] = config.alpha_pattern.get(alpha_key, config.alpha)

if isinstance(target, LoHaLayer):
    target.update_layer(adapter_name, **kwargs)
else:
    new_module = self._create_new_module(config, adapter_name, target, **kwargs)
    self._replace_module(parent, target_name, new_module, target)
</syntaxhighlight>

=== Target Module Resolution ===
LoHaModel uses <code>TRANSFORMERS_MODELS_TO_LOHA_TARGET_MODULES_MAPPING</code> to automatically identify appropriate modules for different model architectures, eliminating the need for manual specification in many cases.

=== Parameter Prefix ===
All LoHa parameters use the prefix "hada_" (Hadamard), making them easily identifiable in the model's state dict.

== Related Pages ==

* [[huggingface_peft_LoHaConfig|LoHaConfig]] - Configuration for LoHa models
* [[huggingface_peft_LoHaLayer|LoHaLayer]] - Base layer class for LoHa adapters
* [[huggingface_peft_LycorisTuner|LycorisTuner]] - Base class for Lycoris-based tuners
* [[huggingface_peft_LoKrModel|LoKrModel]] - Related model using Kronecker products
* [[huggingface_peft_LoraModel|LoraModel]] - Alternative low-rank adaptation method
* [[huggingface_peft_get_peft_model|get_peft_model]] - Standard function to create PEFT models

[[Category:PEFT]]
[[Category:Model]]
[[Category:LoHa]]
[[Category:Low-Rank Adaptation]]
[[Category:Parameter-Efficient Fine-Tuning]]
[[Category:Diffusion Models]]
