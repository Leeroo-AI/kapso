= OFTModel =

== Knowledge Sources ==

* [https://github.com/huggingface/peft HuggingFace PEFT Repository]
* [https://arxiv.org/abs/2306.07280 OFT: Controlling Text-to-Image Diffusion by Orthogonal Finetuning]

== Domains ==

[[Category:NLP]]
[[Category:PEFT]]
[[Category:Orthogonal_Fine_Tuning]]
[[Category:Model_Architecture]]

== Overview ==

=== Description ===

The <code>OFTModel</code> class is the main implementation of Orthogonal Fine-Tuning (OFT) in the PEFT library. It extends <code>BaseTuner</code> to create and manage OFT adapter layers attached to a pre-trained model. OFT is a parameter-efficient fine-tuning method that applies orthogonal transformations to model weights, preserving their norm while enabling task-specific adaptation.

The class handles the creation, replacement, and management of OFT layers across different module types and quantization backends. It provides a unified interface for applying OFT to various model architectures and supports multiple quantization methods including GPTQ, AQLM, AWQ, EETQ, HQQ, and bits-and-bytes (BNB).

Key features:
* Automatic detection and replacement of target modules with OFT layers
* Support for multiple quantization backends
* Dispatcher system for selecting appropriate OFT implementation per layer type
* Configuration-based module targeting (target_modules, exclude_modules)
* Layer-specific transformation via layers_to_transform
* Merge capability checking for different backends
* Integration with HuggingFace Transformers models

=== Usage ===

The <code>OFTModel</code> is typically instantiated through PEFT's <code>get_peft_model</code> function, which automatically creates and attaches OFT adapters to a pre-trained model based on the provided <code>OFTConfig</code>.

== Code Reference ==

=== Source Location ===

File: <code>src/peft/tuners/oft/model.py</code>

Repository: HuggingFace PEFT (Parameter-Efficient Fine-Tuning)

=== Class: OFTModel ===

==== Signature ====

<syntaxhighlight lang="python">
class OFTModel(BaseTuner):
    prefix: str = "oft_"
    tuner_layer_cls = OFTLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_OFT_TARGET_MODULES_MAPPING
</syntaxhighlight>

==== Import ====

<syntaxhighlight lang="python">
from peft import OFTModel, OFTConfig
# or
from peft.tuners.oft.model import OFTModel
</syntaxhighlight>

== I/O Contract ==

=== Constructor (Implicit via get_peft_model) ===

{| class="wikitable"
! Parameter !! Type !! Description
|-
| model || torch.nn.Module || The pretrained model to which adapter layers will be attached
|-
| config || OFTConfig || Configuration object for OFT
|-
| adapter_name || str || Name of the adapter (default: "default")
|-
| low_cpu_mem_usage || bool || Create empty adapter weights on meta device (default: False)
|}

=== Key Methods ===

==== _create_and_replace ====

Internal method to create and replace target modules with OFT layers.

{| class="wikitable"
! Parameter !! Type !! Description
|-
| oft_config || OFTConfig || OFT configuration
|-
| adapter_name || str || Name for the adapter
|-
| target || torch.nn.Module || The module to replace
|-
| target_name || str || Name of the target module
|-
| parent || torch.nn.Module || Parent module containing the target
|-
| current_key || str || Full key path to the module
|}

==== _create_new_module ====

Static method to create a new OFT module using the dispatcher system.

{| class="wikitable"
! Parameter !! Type !! Description
|-
| oft_config || OFTConfig || OFT configuration
|-
| adapter_name || str || Name for the adapter
|-
| target || torch.nn.Module || The module to wrap
|-
| **kwargs || Any || Additional keyword arguments
|}

{| class="wikitable"
! Return !! Type !! Description
|-
| new_module || torch.nn.Module || The newly created OFT-wrapped module
|}

==== _check_merge_allowed ====

Verifies that the configuration supports merging.

Raises <code>ValueError</code> if:
* Model is GPTQ quantized
* Base model layers are replicated

== Usage Examples ==

=== Basic Usage ===

<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import get_peft_model, OFTConfig

# Load a pre-trained model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Configure OFT
oft_config = OFTConfig(
    r=8,
    target_modules=["c_attn"],  # Target attention projection
    module_dropout=0.1,
)

# Create OFT model - automatically wraps with OFTModel
peft_model = get_peft_model(model, oft_config)

# Model now has OFT adapters attached
print(peft_model)
</syntaxhighlight>

=== Diffusion Model Example ===

<syntaxhighlight lang="python">
from diffusers import StableDiffusionPipeline
from peft import OFTModel, OFTConfig

# Configure OFT for text encoder
config_te = OFTConfig(
    r=8,
    target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
    module_dropout=0.0,
    init_weights=True,
)

# Configure OFT for UNet
config_unet = OFTConfig(
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
    module_dropout=0.0,
    init_weights=True,
)

# Load model
model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# Apply OFT to both text encoder and UNet
model.text_encoder = OFTModel(model.text_encoder, config_te, "default")
model.unet = OFTModel(model.unet, config_unet, "default")

# Fine-tune the model
# ... training code ...
</syntaxhighlight>

=== With Quantized Models ===

<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, OFTConfig

# Load model with 8-bit quantization
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto",
)

# Configure OFT
oft_config = OFTConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    module_dropout=0.1,
)

# Apply OFT - automatically uses BNB 8-bit OFT layers
peft_model = get_peft_model(model, oft_config)
</syntaxhighlight>

=== Layer-Specific Transformation ===

<syntaxhighlight lang="python">
from peft import OFTConfig, get_peft_model

# Only transform first 4 transformer layers
oft_config = OFTConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    layers_to_transform=[0, 1, 2, 3],  # First 4 layers only
    layers_pattern="layers",  # Pattern for nn.ModuleList
)

peft_model = get_peft_model(model, oft_config)
</syntaxhighlight>

=== Multiple Adapters ===

<syntaxhighlight lang="python">
from peft import get_peft_model, OFTConfig

# Add first adapter
oft_config1 = OFTConfig(r=8, target_modules=["q_proj", "v_proj"])
peft_model = get_peft_model(model, oft_config1, adapter_name="task1")

# Add second adapter
oft_config2 = OFTConfig(r=16, target_modules=["q_proj", "v_proj"])
peft_model.add_adapter("task2", oft_config2)

# Switch between adapters
peft_model.set_adapter("task1")
output1 = peft_model(input_ids)

peft_model.set_adapter("task2")
output2 = peft_model(input_ids)
</syntaxhighlight>

=== Checking Merge Capability ===

<syntaxhighlight lang="python">
from peft import get_peft_model, OFTConfig

# GPTQ model - merging not allowed
gptq_model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    quantization_config={"method": "gptq"}
)
peft_model = get_peft_model(gptq_model, oft_config)

# Try to merge
try:
    peft_model.merge_adapter()
except ValueError as e:
    print(e)  # "Cannot merge OFT layers when the model is gptq quantized"
</syntaxhighlight>

== Implementation Details ==

=== Dispatcher System ===

The <code>_create_new_module</code> method uses a dispatcher system to select the appropriate OFT implementation:

1. '''BNB 8-bit''' (if available): <code>dispatch_bnb_8bit</code>
2. '''BNB 4-bit''' (if available): <code>dispatch_bnb_4bit</code>
3. '''EETQ''': <code>dispatch_eetq</code>
4. '''AQLM''': <code>dispatch_aqlm</code>
5. '''AWQ''': <code>dispatch_awq</code>
6. '''GPTQ''': <code>dispatch_gptq</code>
7. '''HQQ''': <code>dispatch_hqq</code>
8. '''INC''': <code>dispatch_inc</code>
9. '''Default''' (Linear, Conv2d): <code>dispatch_default</code>

The first dispatcher that returns a non-None module is used.

=== Quantization Configuration Handling ===

The class automatically detects and passes quantization configuration to dispatchers:

<syntaxhighlight lang="python">
quant_methods = ["gptq", "aqlm", "awq"]
for quant_method in quant_methods:
    quantization_config = get_quantization_config(self.model, method=quant_method)
    if quantization_config is not None:
        kwargs[f"{quant_method}_quantization_config"] = quantization_config
</syntaxhighlight>

=== Module Replacement Logic ===

When creating/replacing modules:

1. Check if target is already an OFTLayer
2. If not, create new module via dispatcher system
3. If newly added adapter (not in active_adapters), set requires_grad=False
4. Replace module in parent
5. If already OFTLayer, update existing layer with new adapter

=== Merge Restrictions ===

The <code>_check_merge_allowed</code> method enforces:

1. Call parent class check (general BaseTuner restrictions)
2. No GPTQ quantization:
<syntaxhighlight lang="python">
if getattr(self.model, "quantization_method", None) == "gptq":
    raise ValueError("Cannot merge OFT layers when the model is gptq quantized")
</syntaxhighlight>

3. No layer replication:
<syntaxhighlight lang="python">
if self.peft_config.get("layer_replication"):
    raise ValueError("Cannot merge OFT layers when base model layers are replicated")
</syntaxhighlight>

=== Target Module Mapping ===

The class uses <code>TRANSFORMERS_MODELS_TO_OFT_TARGET_MODULES_MAPPING</code> to automatically determine target modules for known architectures when <code>target_modules</code> is not explicitly specified.

=== Low CPU Memory Mode ===

When <code>low_cpu_mem_usage=True</code>, adapter weights are created on meta device to speed up loading, particularly useful for large models.

== Related Pages ==

* [[huggingface_peft_OFTConfig|OFTConfig]] - Configuration class for OFT
* [[huggingface_peft_OFT_AQLM|OFT AQLM Integration]] - AQLM quantized OFT layers
* [[huggingface_peft_OFT_AWQ|OFT AWQ Integration]] - AWQ quantized OFT layers
* [[huggingface_peft_OFT_GPTQ|OFT GPTQ Integration]] - GPTQ quantized OFT layers
* [[huggingface_peft_OFT_EETQ|OFT EETQ Integration]] - EETQ quantized OFT layers
* [[huggingface_peft_OFT_HQQ|OFT HQQ Integration]] - HQQ quantized OFT layers
* [[huggingface_peft_OFT_IntelFP8|OFT Intel FP8 Integration]] - Intel Neural Compressor integration

== See Also ==

* OFT Paper: [https://arxiv.org/abs/2306.07280 Controlling Text-to-Image Diffusion by Orthogonal Finetuning]
* PEFT Documentation: [https://huggingface.co/docs/peft HuggingFace PEFT Docs]
* BaseTuner: Base class for all PEFT tuner implementations
* OFTLayer: Base layer class for OFT implementations
