= LoRA TorchAO Implementation =

== Knowledge Sources ==
* '''Repository:''' [https://github.com/huggingface/peft HuggingFace PEFT]
* '''Source File:''' src/peft/tuners/lora/torchao.py

== Domains ==
* [[Natural Language Processing]]
* [[Parameter-Efficient Fine-Tuning]]
* [[Quantization]]
* [[Low-Rank Adaptation]]
* [[PyTorch]]

== Overview ==

=== Description ===
The TorchAO LoRA implementation provides support for applying Low-Rank Adaptation (LoRA) to models quantized with PyTorch's torchao quantization library. TorchAO provides efficient quantization methods for PyTorch models, and this module enables parameter-efficient fine-tuning while maintaining the benefits of quantization.

The implementation consists of two main components:
* '''TorchaoLoraLinear''': A specialized LoRA layer for torchao quantized linear layers
* '''dispatch_torchao''': A dispatcher function that creates TorchAO LoRA layers when appropriate

Key features:
* Compatible with torchao AffineQuantizedTensor and LinearActivationQuantizedTensor
* Support for merge and unmerge operations with automatic dequantization/requantization
* Safe merge option with NaN detection
* Currently supports int8 weights (int4 support planned)
* Automatic dtype checking and validation
* LoRA bias is not yet supported

The implementation handles the complexity of working with quantized weights by:
1. Dequantizing weights when merging/unmerging
2. Applying LoRA delta weights
3. Requantizing back to the original quantization format

=== Usage ===
This module is automatically used by PEFT when applying LoRA to models that have been quantized with torchao. The dispatcher function detects torchao quantized tensors and applies the appropriate LoRA wrapper.

== Code Reference ==

=== Source Location ===
<code>src/peft/tuners/lora/torchao.py</code>

=== Class Signature ===
<syntaxhighlight lang="python">
class TorchaoLoraLinear(Linear):
    def __init__(
        self,
        *args,
        get_apply_tensor_subclass,
        **kwargs
    )
</syntaxhighlight>

=== Dispatcher Function ===
<syntaxhighlight lang="python">
def dispatch_torchao(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs: Any,
) -> Optional[torch.nn.Module]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.lora.torchao import TorchaoLoraLinear, dispatch_torchao
</syntaxhighlight>

== I/O Contract ==

=== TorchaoLoraLinear Constructor Parameters ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| *args || Any || - || Positional arguments passed to parent Linear class
|-
| get_apply_tensor_subclass || callable || Required || Function to get the quantization subclass for requantization
|-
| **kwargs || Any || - || Keyword arguments passed to parent Linear class (r, lora_alpha, lora_dropout, etc.)
|}

'''Note:''' Raises ValueError if lora_bias is set to True.

=== merge Method ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| safe_merge || bool || False || If True, checks for NaNs before merging
|-
| adapter_names || Optional[list[str]] || None || List of adapter names to merge; None merges all active adapters
|}

'''Behavior:'''
* Dequantizes base weights
* Adds LoRA delta weights
* Checks for NaNs if safe_merge=True
* Requantizes weights using torchao
* Raises NotImplementedError if dequantization is not supported

=== unmerge Method ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| (none) || - || - || No parameters
|}

'''Behavior:'''
* Dequantizes base weights
* Subtracts LoRA delta weights
* Requantizes weights using torchao
* Raises NotImplementedError if dequantization is not supported

=== dispatch_torchao Parameters ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| target || torch.nn.Module || The module to potentially wrap
|-
| adapter_name || str || Name of the adapter
|-
| lora_config || LoraConfig || LoRA configuration object
|-
| **kwargs || Any || Additional arguments passed to TorchaoLoraLinear
|}

{| class="wikitable"
! Return !! Type !! Description
|-
| new_module || Optional[torch.nn.Module] || TorchaoLoraLinear if target uses torchao quantization, None otherwise
|}

== Usage Examples ==

=== Using with PEFT and TorchAO Quantization ===
<syntaxhighlight lang="python">
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM
from torchao.quantization import quantize_, int8_weight_only

# Load model
model = AutoModelForCausalLM.from_pretrained("model_name")

# Apply int8 quantization using torchao
quantize_(model, int8_weight_only())

# Configure LoRA - PEFT will automatically use TorchAO dispatcher
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",  # lora_bias not supported yet
    task_type="CAUSAL_LM"
)

# PEFT automatically applies TorchaoLoraLinear to quantized layers
peft_model = get_peft_model(model, lora_config)

# Train with LoRA adapters
trainer.train(peft_model)
</syntaxhighlight>

=== Merging and Unmerging Adapters ===
<syntaxhighlight lang="python">
from peft import PeftModel

# Load model with LoRA adapters
peft_model = PeftModel.from_pretrained(base_model, "path/to/adapters")

# Merge adapters into base weights (with safety check)
peft_model.merge_adapter(safe_merge=True)

# Model now runs without adapter overhead
output = peft_model(input_ids)

# Unmerge to restore original weights + adapters
peft_model.unmerge_adapter()
</syntaxhighlight>

=== Safe Merge with NaN Detection ===
<syntaxhighlight lang="python">
import torch
from peft import get_peft_model, LoraConfig

# Setup model with TorchAO quantization and LoRA
# ... (quantization and PEFT setup)

# Attempt safe merge - will check for NaNs
try:
    peft_model.merge_adapter(safe_merge=True, adapter_names=["default"])
    print("Merge successful!")
except ValueError as e:
    print(f"Merge failed due to NaNs: {e}")
    # The adapter may be broken, investigate training process
</syntaxhighlight>

=== Internal Dispatcher Usage ===
<syntaxhighlight lang="python">
from peft.tuners.lora.torchao import dispatch_torchao
from peft.tuners.lora.config import LoraConfig
from torchao.dtypes import AffineQuantizedTensor

# Check if layer uses torchao quantization
target_layer = model.some_layer

# Create LoRA config
lora_config = LoraConfig(r=8, lora_alpha=16)

# Dispatcher automatically wraps if appropriate
new_layer = dispatch_torchao(
    target=target_layer,
    adapter_name="default",
    lora_config=lora_config,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)

if new_layer is not None:
    # Successfully created TorchAO LoRA layer
    model.some_layer = new_layer
    print("Applied TorchAO LoRA wrapper")
</syntaxhighlight>

=== Handling Unsupported Quantization Types ===
<syntaxhighlight lang="python">
from torchao.quantization import int4_weight_only

# Currently only int8 is fully supported
try:
    quantize_(model, int4_weight_only())
    peft_model = get_peft_model(model, lora_config)
except ValueError as e:
    print(f"Error: {e}")
    # Output: "TorchaoLoraLinear only supports int8 weights for now."

# Use int8 instead
quantize_(model, int8_weight_only())
peft_model = get_peft_model(model, lora_config)  # Works!
</syntaxhighlight>

=== Checking Quantization Status ===
<syntaxhighlight lang="python">
# The implementation checks dtype automatically
from torchao.dtypes import AffineQuantizedTensor

# Check if a layer is quantized with torchao
layer = model.layer

if isinstance(layer.weight, AffineQuantizedTensor):
    print("Layer is quantized with torchao")

    # Check dtype support (int8 vs int4)
    if hasattr(layer.weight, 'tensor_impl'):
        # torchao 0.7.0+
        dtype = layer.weight.tensor_impl.data.dtype
    else:
        # torchao < 0.7.0
        dtype = layer.weight.layout_tensor.data.dtype

    print(f"Quantization dtype: {dtype}")
</syntaxhighlight>

== Related Pages ==
* [[LoRA Layer]]
* [[LoRA Linear]]
* [[PEFT Configuration]]
* [[TorchAO Quantization]]
* [[Parameter-Efficient Fine-Tuning]]
* [[Model Quantization]]
* [[Low-Rank Adaptation]]
* [[Int8 Quantization]]
* [[Weight Quantization]]

== Notes ==
* '''Supported Quantization Types:''' Currently only int8 weights are fully supported; int4 support is planned
* '''LoRA Bias:''' Not yet supported - will raise ValueError if enabled
* '''Merge/Unmerge:''' Fully implemented with automatic dequantization/requantization
* '''Safe Merge:''' Optional NaN detection to catch broken adapters
* '''TorchAO Versions:''' Code handles both torchao 0.7.0+ and earlier versions
* '''Memory Efficiency:''' Merge operation requires temporary dequantization, using additional memory
* '''Quantization Preservation:''' After merge/unmerge, weights remain in quantized format

== Implementation Details ==

=== Dtype Checking ===
The implementation checks for int8 dtype support in the constructor:
* For torchao >= 0.7.0: Checks <code>weight.tensor_impl.data.dtype</code>
* For torchao < 0.7.0: Checks <code>weight.layout_tensor.data.dtype</code>

=== Merge/Unmerge Process ===
1. Dequantize base weights using <code>weight.dequantize()</code>
2. Apply delta weights (add for merge, subtract for unmerge)
3. Delete old quantized weight
4. Assign new float weight
5. Requantize using <code>quantize_(base_layer, get_apply_tensor_subclass())</code>

=== Error Handling ===
* Raises <code>NotImplementedError</code> if dequantization is not supported
* Raises <code>ValueError</code> if safe_merge detects NaNs
* Raises <code>ValueError</code> if lora_bias is enabled
* Raises <code>ValueError</code> if dtype is not int8

== References ==
* TorchAO Repository: https://github.com/pytorch/ao
* LoRA Paper: https://arxiv.org/abs/2106.09685
* PEFT Documentation: https://huggingface.co/docs/peft
* PyTorch Quantization: https://pytorch.org/docs/stable/quantization.html
