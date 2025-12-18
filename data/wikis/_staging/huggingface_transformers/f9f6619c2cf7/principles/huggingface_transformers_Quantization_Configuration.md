{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Model Loading|https://huggingface.co/docs/transformers/main_classes/model]]
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Model_Management]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Quantization Configuration is the process of setting up model compression parameters and instantiating appropriate quantization handlers based on the desired precision reduction strategy.

=== Description ===
Large language models often require quantization to reduce memory footprint and improve inference speed. Quantization Configuration handles the setup of quantization schemes such as 4-bit/8-bit quantization (bitsandbytes), GPTQ, AWQ, and GGUF formats. This principle involves parsing quantization parameters from model configurations, validating hardware compatibility, selecting appropriate quantization backends, and instantiating quantizer objects that will handle the actual weight conversion during model loading.

The principle abstracts the differences between various quantization methods by providing a unified configuration interface. It ensures that quantization parameters are validated early, before expensive model loading operations, and that the appropriate low-level quantization libraries are available and properly configured.

=== Usage ===
Apply Quantization Configuration when loading models that are already quantized or when you want to quantize a model during the loading process. This is essential for deploying large models on consumer hardware with limited memory, or when optimizing inference latency for production deployments.

== Theoretical Basis ==

Quantization configuration follows a factory pattern for instantiating the correct quantizer:

1. '''Configuration Parsing''': Extract quantization parameters from config dict
   * quant_method: Type of quantization (bitsandbytes, gptq, awq, etc.)
   * bits: Target bit width (4, 8, etc.)
   * Additional method-specific parameters
2. '''Method Validation''': Ensure quantization method is supported
3. '''Hardware Compatibility''': Check if hardware supports the quantization method
   * CUDA availability for GPU quantization
   * CPU fallback considerations
4. '''Quantizer Selection''': Map configuration to appropriate quantizer class
5. '''Quantizer Instantiation''': Create quantizer object with validated config

'''Decision Tree''':
```
function configure_quantization(quantization_config):
    if quantization_config is None:
        return None

    # Normalize to config object if dict
    if is_dict(quantization_config):
        quantization_config = parse_quantization_config(quantization_config)

    method = quantization_config.quant_method

    # Special handling for bitsandbytes (has 4-bit and 8-bit variants)
    if method == "bitsandbytes":
        if quantization_config.load_in_8bit:
            method = "bitsandbytes_8bit"
        else:
            method = "bitsandbytes_4bit"

    # Validate method is supported
    if method not in SUPPORTED_QUANTIZERS:
        throw ValueError("Unsupported quantization method: " + method)

    # Get quantizer class for this method
    quantizer_class = QUANTIZER_MAPPING[method]

    # Instantiate and return
    return quantizer_class(quantization_config)
```

'''Quantization Methods''':
* '''bitsandbytes (4-bit/8-bit)''': Reduces weights to 4 or 8 bits using custom CUDA kernels
* '''GPTQ''': Group-wise quantization with learned rounding
* '''AWQ''': Activation-aware weight quantization
* '''GGUF''': CPU-optimized quantized format
* '''AQLM''': Additive quantization of language models

The key properties are:
* '''Method Specificity''': Each quantization method requires different configuration parameters
* '''Early Validation''': Configuration errors should be caught before model loading
* '''Hardware Awareness''': Quantizer must be compatible with available hardware

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Quantizer_setup]]
