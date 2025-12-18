{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Quantization Documentation|https://huggingface.co/docs/transformers/main_classes/quantization]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Concrete method for converting model modules to quantization-specific implementations provided by HuggingFace Transformers.

=== Description ===
HfQuantizer._convert_model_for_quantization implements the Linear_Layer_Replacement principle by traversing the model graph and replacing standard modules with quantization-specific equivalents. It uses a registry (MODULES_TO_PATCH_FOR_QUANTIZATION) that maps module class names to their quantized replacements, checking that the current quantization method supports each module type. The conversion happens on the meta device using accelerate's init_empty_weights context, allowing structural changes without memory allocation.

=== Usage ===
This method is called automatically during preprocess_model when pre_quantized=True. It's typically invoked for quantization backends that require different module types before loading (GPTQ, AWQ). For on-the-fly quantization (bitsandbytes), module conversion happens during postprocess_model instead. Extend MODULES_TO_PATCH_FOR_QUANTIZATION to support new model architectures with special quantization requirements.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/quantizers/base.py
* '''Lines:''' 299-313

=== Signature ===
<syntaxhighlight lang="python">
class HfQuantizer(ABC):
    """
    Abstract class of the HuggingFace quantizer.
    """

    def _convert_model_for_quantization(self, model):
        """
        Convert model modules to quantization-specific implementations.
        Used for architectures with special modules that need replacement before loading.

        Args:
            model (~transformers.PreTrainedModel): The model to convert. Should be on meta device.

        Returns:
            None: Model is modified in-place.

        Note:
            This method uses MODULES_TO_PATCH_FOR_QUANTIZATION registry to determine
            which modules need conversion. Each entry maps:
            - module_class_name (str): The original module's class name
            - module_name (class): The replacement module class
            - quantization_methods (list[str]): Supported quantization backends
        """
        from accelerate import init_empty_weights

        for name, module in model.named_modules():
            module_class_name = module.__class__.__name__
            if module_class_name in MODULES_TO_PATCH_FOR_QUANTIZATION and (
                self.quantization_config.quant_method
                in MODULES_TO_PATCH_FOR_QUANTIZATION[module_class_name]["quantization_methods"]
            ):
                with init_empty_weights():
                    parent_module, name = get_module_from_name(model, name)
                    parent_module._modules[name] = MODULES_TO_PATCH_FOR_QUANTIZATION[module_class_name]["module_name"](
                        model.config.get_text_config()
                    )


# Module patching registry
MODULES_TO_PATCH_FOR_QUANTIZATION = {
    "Llama4TextExperts": {
        "module_name": SequentialLlama4TextExperts,
        "quantization_methods": ["bitsandbytes", "gptq", "awq"],
    },
    # Additional module types can be registered here
}
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers.quantizers.base import HfQuantizer
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel || Yes || Model to convert, should be on meta device
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| None || NoneType || Model is modified in-place with modules replaced
|}

== Usage Examples ==

=== Basic Conversion Flow ===
<syntaxhighlight lang="python">
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.quantizers.auto import AutoHfQuantizer
from transformers import GPTQConfig
from accelerate import init_empty_weights

# Step 1: Initialize model on meta device
with init_empty_weights():
    config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = AutoModelForCausalLM.from_config(config)

# Step 2: Create quantizer
gptq_config = GPTQConfig(bits=4, group_size=128)
quantizer = AutoHfQuantizer.from_config(gptq_config)

# Step 3: Convert modules
print("Before conversion:")
for name, module in model.named_modules():
    if "experts" in name:
        print(f"  {name}: {type(module).__name__}")

quantizer._convert_model_for_quantization(model)

print("\nAfter conversion:")
for name, module in model.named_modules():
    if "experts" in name:
        print(f"  {name}: {type(module).__name__}")

# Output:
# Before: Llama4TextExperts
# After: SequentialLlama4TextExperts
</syntaxhighlight>

=== MoE Expert Conversion Example ===
<syntaxhighlight lang="python">
from transformers.quantizers.base import SequentialLlama4TextExperts
from torch.nn import ModuleList

# Before conversion:
# model.layers[0].block_sparse_moe.experts is Llama4TextExperts
# (custom ModuleList with parallel forward)

original_experts = model.layers[0].block_sparse_moe.experts
print(f"Original type: {type(original_experts).__name__}")
print(f"Original experts: {len(original_experts)}")

# After _convert_model_for_quantization:
converted_experts = model.layers[0].block_sparse_moe.experts
print(f"Converted type: {type(converted_experts).__name__}")
print(f"Converted experts: {len(converted_experts)}")

# SequentialLlama4TextExperts maintains same interface
# but processes experts sequentially for quantization compatibility
</syntaxhighlight>

=== Extending Registry for Custom Modules ===
<syntaxhighlight lang="python">
from transformers.quantizers.base import MODULES_TO_PATCH_FOR_QUANTIZATION, HfQuantizer
from torch.nn import Module

# Custom expert module that needs special handling
class CustomExpertModule(Module):
    """Custom MoE expert implementation"""
    def __init__(self, config):
        super().__init__()
        self.experts = ModuleList([...])

    def forward(self, x):
        # Parallel expert processing
        return torch.stack([expert(x) for expert in self.experts])


# Custom sequential version for quantization
class SequentialCustomExperts(Module):
    """Sequential version for quantization compatibility"""
    def __init__(self, config):
        super().__init__()
        self.experts = ModuleList([...])

    def forward(self, x):
        # Sequential expert processing (quantization-friendly)
        outputs = []
        for expert in self.experts:
            outputs.append(expert(x))
        return torch.stack(outputs)


# Register custom module for patching
MODULES_TO_PATCH_FOR_QUANTIZATION["CustomExpertModule"] = {
    "module_name": SequentialCustomExperts,
    "quantization_methods": ["bitsandbytes", "gptq"],
}

# Now CustomExpertModule will be converted automatically
quantizer = AutoHfQuantizer.from_config(quantization_config)
quantizer._convert_model_for_quantization(model)
# CustomExpertModule → SequentialCustomExperts
</syntaxhighlight>

=== Conditional Conversion by Quantization Method ===
<syntaxhighlight lang="python">
from transformers.quantizers.base import MODULES_TO_PATCH_FOR_QUANTIZATION

# Register module with method restrictions
MODULES_TO_PATCH_FOR_QUANTIZATION["SpecialLinear"] = {
    "module_name": QuantizedSpecialLinear,
    "quantization_methods": ["gptq", "awq"],  # Only for these methods
}

# With GPTQ: conversion happens
gptq_quantizer = AutoHfQuantizer.from_config(GPTQConfig(bits=4))
gptq_quantizer._convert_model_for_quantization(model)
# SpecialLinear → QuantizedSpecialLinear

# With BitsAndBytes: no conversion (not in supported methods)
bnb_quantizer = AutoHfQuantizer.from_config(BitsAndBytesConfig(load_in_4bit=True))
bnb_quantizer._convert_model_for_quantization(model)
# SpecialLinear remains unchanged
</syntaxhighlight>

=== Meta Device Conversion ===
<syntaxhighlight lang="python">
from accelerate import init_empty_weights
from transformers.quantizers.base import get_module_from_name

def custom_convert_model(model, quantizer_config):
    """Custom conversion logic showing meta device handling"""
    from accelerate import init_empty_weights

    for name, module in model.named_modules():
        if should_convert(module):
            # Get parent reference
            parent_module, child_name = get_module_from_name(model, name)

            # Create replacement on meta device (no memory)
            with init_empty_weights():
                new_module = QuantizedModule(
                    in_features=module.in_features,
                    out_features=module.out_features,
                )

            # Replace in parent
            parent_module._modules[child_name] = new_module

            print(f"Converted {name}: {type(module).__name__} → {type(new_module).__name__}")
            print(f"  New module device: {new_module.weight.device}")  # meta

# All conversions happen without memory allocation
# Weights will be loaded later directly into converted modules
</syntaxhighlight>

=== Pre-Quantized vs On-the-Fly Conversion ===
<syntaxhighlight lang="python">
from transformers import BitsAndBytesConfig, GPTQConfig
from transformers.quantizers.auto import AutoHfQuantizer

# GPTQ (pre-quantized): Convert before loading
gptq_config = GPTQConfig(bits=4, group_size=128)
gptq_quantizer = AutoHfQuantizer.from_config(gptq_config)
print(f"GPTQ pre_quantized: {gptq_quantizer.pre_quantized}")  # True

# During preprocess_model:
# if self.pre_quantized:
#     self._convert_model_for_quantization(model)  # Called here
# Linear → QuantLinear happens before weight loading

# BitsAndBytes (on-the-fly): Convert after loading
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
bnb_quantizer = AutoHfQuantizer.from_config(bnb_config)
print(f"BnB pre_quantized: {bnb_quantizer.pre_quantized}")  # False

# During preprocess_model:
# if self.pre_quantized:  # False, so skip
#     self._convert_model_for_quantization(model)  # Not called

# Conversion happens later in postprocess_model instead
# Loads FP16 weights first, then converts Linear → Linear4bit
</syntaxhighlight>

=== Complete Conversion Pipeline ===
<syntaxhighlight lang="python">
from transformers import AutoConfig, AutoModelForCausalLM, GPTQConfig
from transformers.quantizers.auto import AutoHfQuantizer
from accelerate import init_empty_weights
import torch

def load_pre_quantized_model(model_name, quantization_config):
    """Complete flow showing module conversion"""

    # Step 1: Initialize on meta device
    with init_empty_weights():
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config)

    print("Model initialized on meta device")

    # Step 2: Create quantizer
    quantizer = AutoHfQuantizer.from_config(quantization_config)

    # Step 3: Validate environment
    quantizer.validate_environment(device_map="auto")
    print("Environment validated")

    # Step 4: Preprocess (includes conversion for pre-quantized)
    print(f"\nBefore conversion:")
    count_linear = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    print(f"  torch.nn.Linear modules: {count_linear}")

    quantizer.preprocess_model(model)
    # This calls _convert_model_for_quantization if pre_quantized=True

    print(f"\nAfter conversion:")
    count_linear = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    print(f"  torch.nn.Linear modules: {count_linear}")
    # Should be 0 or fewer (converted to QuantLinear)

    # Step 5: Load pre-quantized weights
    # (weights load directly into QuantLinear modules)

    return model

# Usage
gptq_config = GPTQConfig(bits=4, group_size=128, desc_act=False)
model = load_pre_quantized_model("TheBloke/Llama-2-7B-GPTQ", gptq_config)

# All Linear layers are now QuantLinear
# Ready to load INT4 pre-quantized weights
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Linear_Layer_Replacement]]

=== Called By ===
* [[called_by::Implementation:huggingface_transformers_Quantizer_preprocess]]

=== Related ===
* [[related::Implementation:huggingface_transformers_Skip_modules_handling]]
