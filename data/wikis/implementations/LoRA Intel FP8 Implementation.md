= LoRA Intel FP8 Implementation =

== Knowledge Sources ==
* '''Repository:''' [https://github.com/huggingface/peft HuggingFace PEFT]
* '''Source File:''' src/peft/tuners/lora/inc.py
* '''Tests:''' [https://github.com/huggingface/optimum-habana Optimum-Habana Repository]

== Domains ==
* [[Natural Language Processing]]
* [[Parameter-Efficient Fine-Tuning]]
* [[Quantization]]
* [[Low-Rank Adaptation]]
* [[Intel Neural Compressor]]

== Overview ==

=== Description ===
The Intel FP8 LoRA implementation provides support for applying Low-Rank Adaptation (LoRA) to models quantized with Intel Neural Compressor (INC) FP8 quantization. This module enables parameter-efficient fine-tuning of FP8-quantized models while maintaining the memory and computational benefits of quantization.

The implementation consists of two main components:
* '''IncLoraLinear''': A specialized LoRA layer that extends the standard Linear LoRA layer for INC FP8 quantized layers
* '''dispatch_inc''': A dispatcher function that creates INC LoRA layers when appropriate

Key features:
* Compatible with Intel Neural Compressor FP8 quantization
* Inherits standard LoRA Linear functionality
* Explicitly disables merge/unmerge operations (not yet implemented for INC layers)
* Optimized for Intel Habana hardware
* Tested through the Optimum-Habana test suite

Limitations:
* Merging adapters into base weights is not yet implemented
* Unmerging adapters from base weights is not yet implemented

=== Usage ===
This module is automatically used by PEFT when applying LoRA to models that have been quantized with Intel Neural Compressor's FP8 quantization method. The dispatcher function detects INC PatchedLinear layers and applies the appropriate LoRA wrapper.

== Code Reference ==

=== Source Location ===
<code>src/peft/tuners/lora/inc.py</code>

=== Class Signature ===
<syntaxhighlight lang="python">
class IncLoraLinear(Linear):
    def __init__(
        self,
        base_layer: torch.nn.Module,
        adapter_name: str,
        **kwargs,
    )
</syntaxhighlight>

=== Dispatcher Function ===
<syntaxhighlight lang="python">
def dispatch_inc(
    target: torch.nn.Module,
    adapter_name: str,
    **kwargs
) -> Optional[torch.nn.Module]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.lora.inc import IncLoraLinear, dispatch_inc
</syntaxhighlight>

== I/O Contract ==

=== IncLoraLinear Constructor Parameters ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| base_layer || torch.nn.Module || Required || The INC FP8 quantized base layer to wrap
|-
| adapter_name || str || Required || Name of the adapter to create
|-
| **kwargs || Any || - || Additional arguments passed to parent Linear class (r, lora_alpha, lora_dropout, etc.)
|}

=== merge Method ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| safe_merge || bool || False || If True, checks for NaNs before merging
|-
| adapter_names || Optional[list[str]] || None || List of adapter names to merge
|}

'''Behavior:''' Raises NotImplementedError - merging not yet supported for INC layers.

=== unmerge Method ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| (none) || - || - || No parameters
|}

'''Behavior:''' Raises NotImplementedError - unmerging not yet supported for INC layers.

=== dispatch_inc Parameters ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| target || torch.nn.Module || The module to potentially wrap
|-
| adapter_name || str || Name of the adapter
|-
| **kwargs || Any || Additional arguments passed to IncLoraLinear
|}

{| class="wikitable"
! Return !! Type !! Description
|-
| new_module || Optional[torch.nn.Module] || IncLoraLinear if target is INC PatchedLinear, None otherwise
|}

== Usage Examples ==

=== Using with PEFT and INC Quantization ===
<syntaxhighlight lang="python">
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM
from neural_compressor.torch import quantize

# Load and quantize model with Intel Neural Compressor
model = AutoModelForCausalLM.from_pretrained("model_name")

# Apply FP8 quantization using INC
# (Example - actual quantization config may vary)
quantized_model = quantize(model, quant_config=fp8_config)

# Configure LoRA - PEFT will automatically use INC dispatcher
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# PEFT automatically applies IncLoraLinear to INC layers
peft_model = get_peft_model(quantized_model, lora_config)

# Train the model with LoRA adapters
# Only LoRA parameters will be updated
</syntaxhighlight>

=== Internal Dispatcher Usage ===
<syntaxhighlight lang="python">
from peft.tuners.lora.inc import dispatch_inc
from neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules import PatchedLinear

# Assuming target_layer is an INC PatchedLinear layer
target_layer = model.some_layer  # PatchedLinear instance

# Dispatcher automatically wraps if appropriate
new_layer = dispatch_inc(
    target=target_layer,
    adapter_name="default",
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)

if new_layer is not None:
    # Successfully created INC LoRA layer
    model.some_layer = new_layer
</syntaxhighlight>

=== Attempting Merge (Will Fail) ===
<syntaxhighlight lang="python">
# Note: This will raise NotImplementedError

# Assuming inc_lora_layer is an instance of IncLoraLinear
try:
    inc_lora_layer.merge(safe_merge=False, adapter_names=["default"])
except NotImplementedError as e:
    print(f"Merge not supported: {e}")
    # Output: "Merging LoRA with INC layers is not yet implemented"

try:
    inc_lora_layer.unmerge()
except NotImplementedError as e:
    print(f"Unmerge not supported: {e}")
    # Output: "Unmerging LoRA from INC layers is not yet implemented"
</syntaxhighlight>

=== Training with INC LoRA ===
<syntaxhighlight lang="python">
from transformers import TrainingArguments, Trainer

# Model with INC LoRA applied
peft_model = get_peft_model(quantized_model, lora_config)

# Standard training setup
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train only the LoRA parameters
trainer.train()

# Save only the LoRA adapters (not the full model)
peft_model.save_pretrained("./lora_adapters")
</syntaxhighlight>

== Related Pages ==
* [[LoRA Layer]]
* [[LoRA Linear]]
* [[PEFT Configuration]]
* [[Intel Neural Compressor]]
* [[FP8 Quantization]]
* [[Parameter-Efficient Fine-Tuning]]
* [[Model Quantization]]
* [[Low-Rank Adaptation]]
* [[Optimum Habana]]

== Notes ==
* PEFT tests for INC are maintained in the Optimum-Habana repository
* LLM tests: https://github.com/huggingface/optimum-habana/blob/main/tests/test_peft_inference.py
* Diffusers tests: https://github.com/huggingface/optimum-habana/blob/main/tests/test_diffusers.py
* The implementation inherits all standard LoRA Linear functionality except merge/unmerge
* Merge and unmerge operations will be implemented in future versions
* This implementation is optimized for Intel Habana accelerators
* The IncLoraLinear class extends the standard Linear LoRA layer with INC-specific handling

== References ==
* Intel Neural Compressor: https://github.com/intel/neural-compressor
* Optimum Habana: https://github.com/huggingface/optimum-habana
* LoRA Paper: https://arxiv.org/abs/2106.09685
* PEFT Documentation: https://huggingface.co/docs/peft
* FP8 Quantization Overview: https://arxiv.org/abs/2209.05433
