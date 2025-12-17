{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Method that transforms model architecture to support quantized weights by replacing standard modules with quantization-aware implementations.

=== Description ===

The `preprocess_model()` method is called by HfQuantizer to prepare the model skeleton for quantized weight loading. It:

* Identifies modules to convert based on type and exclusion lists
* Replaces standard layers (e.g., nn.Linear) with quantized equivalents
* Configures quantization parameters on the new layers
* Sets model-level quantization attributes
* Works on meta device for efficiency

Each quantizer subclass implements `_process_model_before_weight_loading()` with method-specific logic.

=== Usage ===

Called automatically during model loading when a quantization_config is present. Happens after model skeleton initialization but before checkpoint weights are loaded. Users don't call this directly.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/quantizers/base.py

=== Signature ===
<syntaxhighlight lang="python">
class HfQuantizer:
    def preprocess_model(
        self,
        model: PreTrainedModel,
        dtype=None,
        **kwargs
    ) -> None:
        """Prepare model architecture for quantized weights."""
        pass

    def _process_model_before_weight_loading(
        self,
        model: PreTrainedModel,
        **kwargs
    ) -> PreTrainedModel:
        """Subclass-specific preprocessing (abstract)."""
        pass
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
| model || PreTrainedModel || Yes || Model to prepare (on meta device)
|-
| dtype || torch.dtype || No || Target dtype for model
|-
| device_map || dict || No || Device placement mapping
|-
| kwargs || dict || No || Additional method-specific parameters
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| None || None || Model is modified in-place
|}

== Usage Examples ==

=== BitsAndBytes 4-bit Preprocessing ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Configuration triggers preprocessing
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_skip_modules=["lm_head"],
)

# During from_pretrained:
# 1. Model skeleton created on meta device
# 2. Quantizer initialized
# 3. quantizer.preprocess_model() called
#    - Replaces nn.Linear with bnb.nn.Linear4bit
#    - Skips lm_head as configured
# 4. Weights loaded into prepared structure

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=config,
    device_map="auto",
)

# Inspect replaced modules
for name, module in model.named_modules():
    if "layers.0.self_attn.q_proj" in name:
        print(type(module))
        # <class 'bitsandbytes.nn.modules.Linear4bit'>
    if "lm_head" in name:
        print(type(module))
        # <class 'torch.nn.modules.linear.Linear'> (not quantized)
</syntaxhighlight>

=== GPTQ Preprocessing ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, GPTQConfig

# GPTQ preprocessing
config = GPTQConfig(
    bits=4,
    group_size=128,
    backend="marlin",
)

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-GPTQ",
    quantization_config=config,
    device_map="auto",
)

# During preprocessing:
# - Replaces Linear with GPTQLinear layers
# - Configures group_size, bits parameters
# - Sets up backend-specific structures
</syntaxhighlight>

=== Module Exclusion Pattern ===
<syntaxhighlight lang="python">
from transformers.quantizers.base import HfQuantizer

class CustomQuantizer(HfQuantizer):
    def _process_model_before_weight_loading(self, model, **kwargs):
        from transformers.integrations import replace_with_custom_linear

        # Get modules to skip
        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model,
            skip_modules=self.quantization_config.modules_to_not_convert,
            keep_in_fp32_modules=model._keep_in_fp32_modules,
        )

        # Replace appropriate modules
        model = replace_with_custom_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
        )

        return model
</syntaxhighlight>

=== Handling MoE Models ===
<syntaxhighlight lang="python">
# For models with expert layers (e.g., Mixtral)
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_skip_modules=["gate"],  # Don't quantize routing gates
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1",
    quantization_config=config,
    device_map="auto",
)

# Preprocessing handles expert modules specially:
# - Each expert's Linear layers quantized individually
# - Gate layers kept in full precision
# - Preserves MoE routing logic
</syntaxhighlight>

== Implementation Details ==

=== BnB Preprocessing Steps ===

For BitsAndBytes quantizers:

1. '''Identify conversion candidates:'''
   * Find all nn.Linear modules
   * Check against skip_modules list
   * Identify tied weights and output embeddings

2. '''Replace modules:'''
   * Create bnb.nn.Linear4bit or Linear8bit
   * Copy configuration (in_features, out_features, bias)
   * Set quantization parameters

3. '''Configure quantization:'''
   * Set quant_type (nf4, fp4, int8)
   * Set compute_dtype
   * Enable double_quant if configured

4. '''Mark model state:'''
   * Set model.is_quantized = True
   * Set model.quantization_method

=== Memory Efficiency ===

Preprocessing on meta device means:
* No actual memory allocation during replacement
* Fast module swapping
* Only allocate when weights are loaded

Memory footprint during preprocessing: ~50 MB regardless of model size

=== Module Type Mapping ===

{| class="wikitable"
|-
! Standard Module !! Quantized Equivalent !! Method
|-
| nn.Linear || bnb.nn.Linear4bit || BitsAndBytes 4-bit
|-
| nn.Linear || bnb.nn.Linear8bitLt || BitsAndBytes 8-bit
|-
| nn.Linear || GPTQLinear || GPTQ
|-
| nn.Linear || WQLinear || AWQ
|-
| nn.Linear || QuantLinear || AQLM
|-
| nn.Linear || QLinear || Quanto
|-
| nn.Linear || EetqLinear || EETQ
|}

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Quantized_Model_Preparation]]

=== Requires ===
* [[requires::Implementation:huggingface_transformers_get_hf_quantizer_init]]

=== Enables ===
* [[enables::Implementation:huggingface_transformers_quantizer_postprocess_model]]

=== Requires Environment ===
