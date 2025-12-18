{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Quantization Documentation|https://huggingface.co/docs/transformers/main_classes/quantization]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Software_Engineering]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Concrete factory class for dispatching to the correct HfQuantizer implementation based on configuration.

=== Description ===
AutoHfQuantizer implements the Quantizer_Selection principle by providing a factory that maps quantization method identifiers to concrete quantizer classes. It handles both dictionary-based and object-based configuration inputs, applies backend-specific transformations (e.g., splitting "bitsandbytes" into 4-bit vs 8-bit variants), and validates that the requested method is supported. The class maintains an AUTO_QUANTIZER_MAPPING registry that new quantization backends can extend.

=== Usage ===
Use AutoHfQuantizer.from_config when you need to:
* Instantiate the correct quantizer from a QuantizationConfig object
* Load models with serialized quantization configs from disk
* Support multiple quantization backends through a single interface
* Convert dictionary-based configs to typed quantizer objects
* Validate quantization method availability before model loading

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/quantizers/auto.py
* '''Lines:''' 155-185

=== Signature ===
<syntaxhighlight lang="python">
class AutoHfQuantizer:
    """
    The Auto-HF quantizer class that takes care of automatically instantiating to the correct
    `HfQuantizer` given the `QuantizationConfig`.
    """

    @classmethod
    def from_config(cls, quantization_config: QuantizationConfigMixin | dict, **kwargs):
        """
        Instantiate the correct HfQuantizer based on the quantization config.

        Args:
            quantization_config (QuantizationConfigMixin or dict): The quantization configuration.
                Can be a config object or dictionary.
            **kwargs: Additional arguments passed to the quantizer constructor.

        Returns:
            HfQuantizer: The instantiated quantizer for the specified method.

        Raises:
            ValueError: If the quantization method is not supported.
        """
        # Convert it to a QuantizationConfig if the q_config is a dict
        if isinstance(quantization_config, dict):
            quantization_config = AutoQuantizationConfig.from_dict(quantization_config)

        quant_method = quantization_config.quant_method

        # Special handling for bnb as we have a single quantization config
        # class for both 4-bit and 8-bit quantization
        if quant_method == QuantizationMethod.BITS_AND_BYTES:
            if quantization_config.load_in_8bit:
                quant_method += "_8bit"
            else:
                quant_method += "_4bit"

        if quant_method not in AUTO_QUANTIZER_MAPPING:
            raise ValueError(
                f"Unknown quantization type, got {quant_method} - supported types are:"
                f" {list(AUTO_QUANTIZER_MAPPING.keys())}"
            )

        target_cls = AUTO_QUANTIZER_MAPPING[quant_method]
        return target_cls(quantization_config, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load quantization config from a pretrained model and instantiate quantizer.

        Args:
            pretrained_model_name_or_path (str): Model identifier or path.
            **kwargs: Additional arguments.

        Returns:
            HfQuantizer: The instantiated quantizer.
        """
        quantization_config = AutoQuantizationConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls.from_config(quantization_config)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers.quantizers.auto import AutoHfQuantizer
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| quantization_config || QuantizationConfigMixin or dict || Yes || Configuration specifying quantization method and parameters
|-
| **kwargs || dict || No || Additional arguments passed to quantizer constructor
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| quantizer || HfQuantizer || Concrete quantizer instance (e.g., Bnb4BitQuantizer, GptqQuantizer)
|}

== Usage Examples ==

=== Dispatch from BitsAndBytesConfig ===
<syntaxhighlight lang="python">
from transformers import BitsAndBytesConfig
from transformers.quantizers.auto import AutoHfQuantizer

# Create 4-bit quantization config
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)

# Dispatch to Bnb4BitQuantizer
quantizer = AutoHfQuantizer.from_config(config)
print(type(quantizer))  # <class 'transformers.quantizers.quantizer_bnb_4bit.Bnb4BitQuantizer'>

# Use quantizer in model loading
# (This is typically done internally by from_pretrained)
</syntaxhighlight>

=== Dispatch from Dictionary Config ===
<syntaxhighlight lang="python">
from transformers.quantizers.auto import AutoHfQuantizer

# Config loaded from JSON or model checkpoint
config_dict = {
    "quant_method": "gptq",
    "bits": 4,
    "group_size": 128,
    "desc_act": False,
}

# AutoHfQuantizer handles conversion and dispatch
quantizer = AutoHfQuantizer.from_config(config_dict)
print(type(quantizer))  # <class 'transformers.quantizers.quantizer_gptq.GptqQuantizer'>
</syntaxhighlight>

=== Load from Pretrained Model ===
<syntaxhighlight lang="python">
from transformers.quantizers.auto import AutoHfQuantizer

# Load quantizer based on model's saved config
quantizer = AutoHfQuantizer.from_pretrained("TheBloke/Llama-2-7B-GPTQ")

# Quantizer is automatically configured for GPTQ
print(quantizer.quantization_config.quant_method)  # "gptq"
</syntaxhighlight>

=== Handle 8-bit vs 4-bit Dispatch ===
<syntaxhighlight lang="python">
from transformers import BitsAndBytesConfig
from transformers.quantizers.auto import AutoHfQuantizer

# 8-bit config
config_8bit = BitsAndBytesConfig(load_in_8bit=True)
quantizer_8bit = AutoHfQuantizer.from_config(config_8bit)
# Dispatches to Bnb8BitQuantizer

# 4-bit config
config_4bit = BitsAndBytesConfig(load_in_4bit=True)
quantizer_4bit = AutoHfQuantizer.from_config(config_4bit)
# Dispatches to Bnb4BitQuantizer

# Both use same config class, dispatcher handles differentiation
</syntaxhighlight>

=== Error Handling for Unsupported Methods ===
<syntaxhighlight lang="python">
from transformers.quantizers.auto import AutoHfQuantizer

# Invalid quantization method
config_dict = {
    "quant_method": "fake_quantization",
    "bits": 4,
}

try:
    quantizer = AutoHfQuantizer.from_config(config_dict)
except ValueError as e:
    print(e)
    # "Unknown quantization type, got fake_quantization - supported types are: [...]"
</syntaxhighlight>

=== Integration with Model Loading ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.quantizers.auto import AutoHfQuantizer

# This is what happens internally in from_pretrained
config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")

# Step 1: Dispatch to quantizer
quantizer = AutoHfQuantizer.from_config(config)

# Step 2: Quantizer validates environment
quantizer.validate_environment(device_map="auto")

# Step 3: Quantizer preprocesses model skeleton
# Step 4: Weights are loaded and quantized
# Step 5: Quantizer postprocesses model

# Users just call:
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=config,
    device_map="auto",
)
# Dispatch happens automatically
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Quantizer_Selection]]

=== Requires ===
* [[requires::Implementation:huggingface_transformers_BitsAndBytesConfig_setup]]

=== Used By ===
* [[used_by::Implementation:huggingface_transformers_Quantizer_validate_environment]]
* [[used_by::Implementation:huggingface_transformers_Quantizer_preprocess]]
