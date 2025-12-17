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

Factory function that instantiates the appropriate HfQuantizer subclass based on quantization configuration and method.

=== Description ===

The `get_hf_quantizer()` function and `AutoHfQuantizer.from_config()` are the entry points for quantizer initialization. They:

* Parse quantization configuration from model or arguments
* Merge configurations when both are present
* Dispatch to the correct quantizer implementation
* Validate environment and update device map
* Return the initialized quantizer instance

This function handles the complexity of supporting multiple quantization methods while providing a unified interface.

=== Usage ===

Called internally by `from_pretrained()` when loading models with quantization. Can also be used directly for custom loading workflows or testing quantizer behavior.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/quantizers/auto.py

=== Signature ===
<syntaxhighlight lang="python">
def get_hf_quantizer(
    config,
    quantization_config,
    device_map,
    weights_only,
    user_agent
) -> tuple[HfQuantizer | None, PretrainedConfig, dict]:
    """Get quantizer instance and update config/device_map."""
    pass

class AutoHfQuantizer:
    @classmethod
    def from_config(
        cls,
        quantization_config: QuantizationConfigMixin | dict,
        **kwargs
    ) -> HfQuantizer:
        """Create quantizer from configuration."""
        pass

    @classmethod
    def merge_quantization_configs(
        cls,
        quantization_config: dict | QuantizationConfigMixin,
        quantization_config_from_args: QuantizationConfigMixin | None,
    ) -> QuantizationConfigMixin:
        """Merge config from model and arguments."""
        pass
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers.quantizers import AutoHfQuantizer, get_hf_quantizer
from transformers.quantizers.auto import AutoHfQuantizer
</syntaxhighlight>

== I/O Contract ==

=== Inputs (get_hf_quantizer) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| config || PretrainedConfig || Yes || Model configuration
|-
| quantization_config || QuantizationConfigMixin || No || Quantization config from arguments
|-
| device_map || dict[str, Any] || No || Device placement mapping
|-
| weights_only || bool || Yes || Whether loading weights only
|-
| user_agent || dict || Yes || User agent for telemetry
|}

=== Inputs (AutoHfQuantizer.from_config) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| quantization_config || QuantizationConfigMixin or dict || Yes || Quantization configuration
|-
| pre_quantized || bool || No || Whether model is pre-quantized (default True)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| hf_quantizer || HfQuantizer or None || Initialized quantizer instance
|-
| config || PretrainedConfig || Updated model configuration
|-
| device_map || dict || Updated device mapping
|}

== Usage Examples ==

=== Automatic Quantizer Selection ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.quantizers import get_hf_quantizer

# Configuration-based dispatch
bnb_config = BitsAndBytesConfig(load_in_4bit=True)

# This happens internally in from_pretrained
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,  # Triggers quantizer initialization
    device_map="auto",
)

# The quantizer is automatically selected based on quant_method
print(type(model.hf_quantizer))
# <class 'transformers.quantizers.quantizer_bnb_4bit.Bnb4BitHfQuantizer'>
</syntaxhighlight>

=== Direct Quantizer Creation ===
<syntaxhighlight lang="python">
from transformers.quantizers import AutoHfQuantizer
from transformers import BitsAndBytesConfig, AutoConfig

# Create configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)

# Direct instantiation
quantizer = AutoHfQuantizer.from_config(
    quantization_config=bnb_config,
    pre_quantized=False,
)

print(quantizer.quantization_config.quant_method)
# QuantizationMethod.BITS_AND_BYTES

# Validate environment
quantizer.validate_environment(device_map="auto", weights_only=False)
</syntaxhighlight>

=== Configuration Merging ===
<syntaxhighlight lang="python">
from transformers import AutoConfig, GPTQConfig
from transformers.quantizers import AutoHfQuantizer

# Model already has quantization config
model_config = AutoConfig.from_pretrained("TheBloke/Llama-2-7B-GPTQ")
print(model_config.quantization_config)
# {'quant_method': 'gptq', 'bits': 4, 'group_size': 128, ...}

# User wants to override backend
user_config = GPTQConfig(backend="marlin")

# Merge configurations
merged_config = AutoHfQuantizer.merge_quantization_configs(
    quantization_config=model_config.quantization_config,
    quantization_config_from_args=user_config,
)

print(merged_config.backend)  # 'marlin'
print(merged_config.bits)     # 4 (from model)
</syntaxhighlight>

=== Method Support Checking ===
<syntaxhighlight lang="python">
from transformers.quantizers import AutoHfQuantizer

# Check if a method is supported
config_dict = {
    "quant_method": "gptq",
    "bits": 4,
    "group_size": 128,
}

is_supported = AutoHfQuantizer.supports_quant_method(config_dict)
print(is_supported)  # True

# Unknown method
unknown_config = {"quant_method": "unknown_method"}
is_supported = AutoHfQuantizer.supports_quant_method(unknown_config)
print(is_supported)  # False (with warning)
</syntaxhighlight>

=== Custom Quantizer Registration ===
<syntaxhighlight lang="python">
from transformers.quantizers import register_quantizer, HfQuantizer
from transformers.utils.quantization_config import QuantizationMethod

# Define custom quantizer
class MyCustomQuantizer(HfQuantizer):
    requires_calibration = False

    def is_serializable(self):
        return True

    @property
    def is_trainable(self):
        return False

    # Implement required methods...

# Register it
@register_quantizer("my_custom_method")
class MyCustomQuantizer(HfQuantizer):
    pass

# Now it's available for auto-dispatch
# AutoHfQuantizer.from_config() will use it when
# quant_method == "my_custom_method"
</syntaxhighlight>

== Quantizer Dispatch Mapping ==

{| class="wikitable"
|-
! quant_method !! Quantizer Class !! Config Class
|-
| bitsandbytes || Bnb4BitHfQuantizer / Bnb8BitHfQuantizer || BitsAndBytesConfig
|-
| gptq || GptqHfQuantizer || GPTQConfig
|-
| awq || AwqQuantizer || AwqConfig
|-
| aqlm || AqlmHfQuantizer || AqlmConfig
|-
| quanto || QuantoHfQuantizer || QuantoConfig
|-
| eetq || EetqHfQuantizer || EetqConfig
|-
| hqq || HqqHfQuantizer || HqqConfig
|-
| compressed-tensors || CompressedTensorsHfQuantizer || CompressedTensorsConfig
|-
| fp8 || FineGrainedFP8HfQuantizer || FineGrainedFP8Config
|-
| torchao || TorchAoHfQuantizer || TorchAoConfig
|-
| bitnet || BitNetHfQuantizer || BitNetQuantConfig
|}

== Environment Validation ==

Each quantizer validates:

'''BitsAndBytes:'''
* `accelerate` >= minimum version
* `bitsandbytes` installed
* CUDA/ROCm/NPU backend available
* Device map compatibility

'''GPTQ:'''
* `gptqmodel` or `optimum` available
* Supported backend (marlin, exllama, etc.)
* CUDA for GPU backends

'''AWQ:'''
* `autoawq` installed
* CUDA available
* Compatible PyTorch version

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Quantizer_Initialization]]

=== Uses ===
* [[uses::Implementation:huggingface_transformers_BitsAndBytesConfig]]

=== Enables ===
* [[enables::Implementation:huggingface_transformers_quantizer_preprocess_model]]
