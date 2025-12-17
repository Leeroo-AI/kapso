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

Finalizes the quantized model after weight loading by configuring runtime behavior and validating quantization state.

=== Description ===

The `postprocess_model()` method is called after quantized weights have been loaded into the model. It:

* Attaches quantization config to model for serialization
* Sets model-level attributes (is_loaded_in_4bit, is_4bit_serializable)
* Performs any method-specific post-loading setup
* Optionally dequantizes if configured
* Validates the final quantized state

Each quantizer implements `_process_model_after_weight_loading()` with method-specific finalization logic.

=== Usage ===

Called automatically at the end of model loading after all weights are in place. Users don't call this directly. The method ensures the model is ready for inference or fine-tuning.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/quantizers/base.py

=== Signature ===
<syntaxhighlight lang="python">
class HfQuantizer:
    def postprocess_model(
        self,
        model: PreTrainedModel,
        **kwargs
    ) -> PreTrainedModel:
        """Finalize model after weight loading."""
        pass

    def _process_model_after_weight_loading(
        self,
        model: PreTrainedModel,
        **kwargs
    ) -> PreTrainedModel:
        """Subclass-specific postprocessing (abstract)."""
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
| model || PreTrainedModel || Yes || Model with loaded quantized weights
|-
| kwargs || dict || No || Additional method-specific parameters
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PreTrainedModel || Finalized quantized model ready for use
|}

== Usage Examples ==

=== BitsAndBytes 4-bit Postprocessing ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)

# During from_pretrained:
# 1. Model skeleton created and preprocessed
# 2. Weights loaded into Linear4bit modules
# 3. quantizer.postprocess_model() called:
#    - Sets model.is_loaded_in_4bit = True
#    - Sets model.is_4bit_serializable = True
#    - Attaches config to model.config.quantization_config

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=config,
    device_map="auto",
)

# Verify postprocessing
print(model.is_loaded_in_4bit)        # True
print(model.is_4bit_serializable)     # True
print(model.config.quantization_config)
# BitsAndBytesConfig {...}

# Model is ready for inference
output = model.generate(**inputs, max_new_tokens=50)
</syntaxhighlight>

=== Dequantization During Postprocessing ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, Mxfp4Config

# Request dequantization after loading
config = Mxfp4Config(dequantize=True)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct-mxfp4",
    quantization_config=config,
    device_map="auto",
)

# During postprocessing:
# 1. Quantized weights loaded
# 2. Dequantization performed
# 3. Quantized layers replaced with standard layers
# 4. Quantization config removed

print(hasattr(model, "quantization_method"))  # False
print(model.is_quantized)                     # False

# Model is now in BF16, no longer quantized
for name, module in model.named_modules():
    if "linear" in name.lower():
        print(type(module))  # torch.nn.Linear
</syntaxhighlight>

=== GPTQ Postprocessing with Backend Configuration ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, GPTQConfig

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

# Postprocessing for GPTQ:
# 1. Configures backend-specific kernels
# 2. Validates weight shapes match config
# 3. Sets up fast inference paths
# 4. Prepares autograd hooks if trainable

# Backend-specific optimizations applied
print(model.config.quantization_config.backend)  # "marlin"
</syntaxhighlight>

=== Custom Postprocessing Implementation ===
<syntaxhighlight lang="python">
from transformers.quantizers.base import HfQuantizer

class CustomQuantizer(HfQuantizer):
    def _process_model_after_weight_loading(self, model, **kwargs):
        # Custom finalization logic

        # 1. Validate quantization
        self._validate_quantized_state(model)

        # 2. Configure runtime optimizations
        if self.quantization_config.use_fast_kernels:
            self._enable_fast_kernels(model)

        # 3. Set model attributes
        model.is_custom_quantized = True
        model.custom_quant_version = "1.0"

        # 4. Prepare for inference
        model.eval()

        return model

    def _validate_quantized_state(self, model):
        """Check that all expected layers are quantized."""
        for name, module in model.named_modules():
            if isinstance(module, CustomQuantizedLinear):
                assert hasattr(module, "scales"), f"Missing scales in {name}"
                assert module.weight.dtype == torch.int8, f"Wrong dtype in {name}"
</syntaxhighlight>

=== Serialization Preparation ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=config,
    device_map="auto",
)

# After postprocessing, model can be saved
if model.is_4bit_serializable:
    model.save_pretrained("./llama2-7b-4bit")

    # Reload later
    model_reloaded = AutoModelForCausalLM.from_pretrained(
        "./llama2-7b-4bit",
        device_map="auto",
    )
    # Automatically detects and loads as quantized
</syntaxhighlight>

== Postprocessing Actions by Method ==

{| class="wikitable"
|-
! Method !! Postprocessing Actions
|-
| BitsAndBytes 4-bit ||
* Set is_loaded_in_4bit = True
* Set is_4bit_serializable = True
* Attach quantization config
|-
| BitsAndBytes 8-bit ||
* Set is_loaded_in_8bit = True
* Set is_8bit_serializable = True
* Configure outlier handling
|-
| GPTQ ||
* Configure backend kernels (marlin, exllama)
* Validate weight shapes
* Set up fast matmul paths
|-
| AWQ ||
* Configure kernel backend
* Set up fused operations
* Validate format compatibility
|-
| Quanto ||
* Freeze quantized layers
* Set up activation quantization (if enabled)
|-
| EETQ ||
* Configure INT8 kernels
* Set inference mode
|-
| HQQ ||
* Configure backend (TorchAO, BitBlas)
* Set up fast dequantization
|}

== Model Attributes Set ==

After postprocessing, models have:

<syntaxhighlight lang="python">
# Generic attributes
model.is_quantized = True
model.quantization_method = QuantizationMethod.BITS_AND_BYTES
model.hf_quantizer = Bnb4BitHfQuantizer(...)
model.config.quantization_config = BitsAndBytesConfig(...)

# Method-specific attributes
model.is_loaded_in_4bit = True  # BnB 4-bit
model.is_loaded_in_8bit = True  # BnB 8-bit
model.is_4bit_serializable = True
model.is_8bit_serializable = True
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Quantized_Weight_Loading]]

=== Requires ===
* [[requires::Implementation:huggingface_transformers_quantizer_preprocess_model]]

=== Enables ===
* [[enables::Principle:huggingface_transformers_Quantized_Runtime_Optimization]]
