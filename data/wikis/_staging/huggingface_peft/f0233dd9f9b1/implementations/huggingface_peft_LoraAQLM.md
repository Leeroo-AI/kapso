= LoRA AQLM Implementation =

== Knowledge Sources ==
* '''Repository:''' [https://github.com/huggingface/peft HuggingFace PEFT]
* '''Source File:''' src/peft/tuners/lora/aqlm.py

== Domains ==
* [[Natural Language Processing]]
* [[Parameter-Efficient Fine-Tuning]]
* [[Quantization]]
* [[Low-Rank Adaptation]]

== Overview ==

=== Description ===
The AQLM LoRA implementation provides support for applying Low-Rank Adaptation (LoRA) to AQLM (Additive Quantization of Language Models) quantized linear layers. AQLM is a quantization method that compresses model weights while maintaining performance. This module enables fine-tuning of AQLM-quantized models using LoRA adapters without requiring full weight updates.

The implementation consists of two main components:
* '''AqlmLoraLinear''': A specialized LoRA layer for AQLM quantized linear layers
* '''dispatch_aqlm''': A dispatcher function that creates AQLM LoRA layers when appropriate

Key features:
* Compatible with AQLM quantized models
* Prevents merging of adapters (not supported for quantized layers)
* Automatic dtype casting for compatibility with quantized weights
* Support for multiple active adapters
* DoRA (Weight-Decomposed Low-Rank Adaptation) is not yet supported

=== Usage ===
This module is typically used internally by PEFT when applying LoRA to models that have been quantized with AQLM. The dispatcher function automatically detects AQLM quantized layers and applies the appropriate LoRA wrapper.

== Code Reference ==

=== Source Location ===
<code>src/peft/tuners/lora/aqlm.py</code>

=== Class Signature ===
<syntaxhighlight lang="python">
class AqlmLoraLinear(torch.nn.Module, LoraLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    )
</syntaxhighlight>

=== Dispatcher Function ===
<syntaxhighlight lang="python">
def dispatch_aqlm(
    target: torch.nn.Module,
    adapter_name: str,
    **kwargs: Any,
) -> Optional[torch.nn.Module]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.lora.aqlm import AqlmLoraLinear, dispatch_aqlm
</syntaxhighlight>

== I/O Contract ==

=== AqlmLoraLinear Constructor Parameters ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| base_layer || torch.nn.Module || Required || The AQLM quantized base layer to wrap
|-
| adapter_name || str || Required || Name of the adapter to create
|-
| r || int || 0 || Rank of the LoRA decomposition
|-
| lora_alpha || int || 1 || Scaling factor for LoRA updates
|-
| lora_dropout || float || 0.0 || Dropout probability for LoRA layers
|-
| init_lora_weights || bool || True || Whether to initialize LoRA weights
|-
| use_rslora || bool || False || Whether to use rank-stabilized LoRA
|-
| use_dora || bool || False || Whether to use DoRA (raises error if True)
|-
| lora_bias || bool || False || Whether to include bias in LoRA layers
|}

=== Forward Method ===
{| class="wikitable"
! Input !! Type !! Description
|-
| x || torch.Tensor || Input tensor to the layer
|}

{| class="wikitable"
! Output !! Type !! Description
|-
| result || torch.Tensor || Output tensor with LoRA adaptations applied
|}

=== dispatch_aqlm Parameters ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| target || torch.nn.Module || The module to potentially wrap
|-
| adapter_name || str || Name of the adapter
|-
| **kwargs || Any || Additional arguments passed to AqlmLoraLinear
|}

{| class="wikitable"
! Return !! Type !! Description
|-
| new_module || Optional[torch.nn.Module] || AqlmLoraLinear if target is AQLM quantized, None otherwise
|}

== Usage Examples ==

=== Internal Dispatcher Usage ===
The dispatcher is typically called internally by PEFT's layer injection mechanism:

<syntaxhighlight lang="python">
from peft.tuners.lora.aqlm import dispatch_aqlm
import torch.nn as nn

# Assuming model has AQLM quantized layers
target_layer = model.layer  # An AQLM QuantizedLinear layer

# Dispatcher automatically wraps if appropriate
new_layer = dispatch_aqlm(
    target=target_layer,
    adapter_name="default",
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)

if new_layer is not None:
    # Successfully created AQLM LoRA layer
    model.layer = new_layer
</syntaxhighlight>

=== Using with PEFT LoraConfig ===
<syntaxhighlight lang="python">
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM

# Load AQLM quantized model
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    quantization_config={"method": "aqlm"}
)

# Configure LoRA - PEFT will automatically use AQLM dispatcher
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    use_dora=False  # DoRA not supported for AQLM
)

# PEFT automatically applies AqlmLoraLinear to AQLM layers
model = get_peft_model(model, lora_config)
</syntaxhighlight>

=== Forward Pass Behavior ===
<syntaxhighlight lang="python">
import torch

# Assuming aqlm_lora_layer is an instance of AqlmLoraLinear
input_tensor = torch.randn(batch_size, seq_len, hidden_size)

# Forward pass applies base layer + LoRA adaptations
output = aqlm_lora_layer(input_tensor)

# The forward method:
# 1. Passes input through AQLM base layer
# 2. For each active adapter:
#    - Applies dropout
#    - Passes through lora_A
#    - Passes through lora_B
#    - Scales by scaling factor
#    - Adds to base output
</syntaxhighlight>

== Related Pages ==
* [[LoRA Layer]]
* [[PEFT Configuration]]
* [[Quantization Methods]]
* [[AQLM Quantization]]
* [[Parameter-Efficient Fine-Tuning]]
* [[Model Compression]]
* [[Low-Rank Adaptation]]
* [[Rank-Stabilized LoRA]]

== Notes ==
* DoRA (Weight-Decomposed Low-Rank Adaptation) is not currently supported for AQLM layers
* Merging of adapters is not supported for AQLM quantized layers
* The implementation handles dtype conversion automatically to ensure compatibility with quantized weights
* Multiple adapters can be active simultaneously
* The forward pass does not support merged adapters due to quantization constraints

== References ==
* AQLM Library: https://github.com/Vahe1994/AQLM
* LoRA Paper: https://arxiv.org/abs/2106.09685
* PEFT Documentation: https://huggingface.co/docs/peft
