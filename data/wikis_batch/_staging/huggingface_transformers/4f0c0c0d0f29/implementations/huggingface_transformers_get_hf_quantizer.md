{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Model_Loading]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Concrete tool for instantiating the appropriate quantizer object based on model quantization configuration provided by the HuggingFace Transformers library.

=== Description ===

`get_hf_quantizer()` is a factory function that analyzes a model's configuration and quantization settings to instantiate the correct HfQuantizer subclass. It handles the complexity of supporting multiple quantization methods (GPTQ, AWQ, bitsandbytes, GGUF, etc.) through a unified interface. The function merges configuration-based and argument-based quantization settings, validates the environment supports the requested quantization method, and updates device mappings and tensor parallel plans accordingly. This is a critical step in the model loading pipeline that determines how weights will be loaded and stored in memory.

=== Usage ===

Use this when you need to:
* Load quantized models with automatic quantizer detection
* Implement custom model loading pipelines that support quantization
* Validate quantization compatibility before loading large models
* Merge user-provided quantization settings with checkpoint defaults
* Update device placement based on quantization requirements

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/quantizers/auto.py (lines 305-337)

=== Signature ===
<syntaxhighlight lang="python">
def get_hf_quantizer(
    config,
    quantization_config,
    device_map,
    weights_only,
    user_agent
):
    """
    Get the appropriate HfQuantizer instance for a model based on its configuration.

    Args:
        config (PretrainedConfig): Model configuration object that may contain
            a quantization_config attribute from the checkpoint
        quantization_config (QuantizationConfigMixin | None): User-provided
            quantization configuration that overrides checkpoint settings
        device_map (dict | str | None): Device placement mapping for model layers
        weights_only (bool): Whether to load weights only without initializing buffers
        user_agent (dict): User agent dictionary for telemetry purposes

    Returns:
        HfQuantizer | None: Instantiated quantizer object for the detected quantization
            method, or None if no quantization is configured. Also updates config,
            device_map, and user_agent as side effects.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers.quantizers.auto import get_hf_quantizer
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| config || PretrainedConfig || Yes || Model configuration object, potentially containing quantization_config attribute from checkpoint
|-
| quantization_config || QuantizationConfigMixin | None || Yes || User-provided quantization configuration to override or merge with checkpoint config
|-
| device_map || dict | str | None || Yes || Device placement mapping for model layers (e.g., "auto", "cuda:0", or explicit layer mapping)
|-
| weights_only || bool || Yes || Whether to load only weights without initializing buffers
|-
| user_agent || dict || Yes || User agent dictionary for telemetry tracking
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| hf_quantizer || HfQuantizer | None || Instantiated quantizer object (e.g., Bnb4BitHfQuantizer, GptqHfQuantizer) or None if no quantization configured
|}

'''Side Effects:'''
* '''config.quantization_config''' is updated with merged quantization settings
* '''device_map''' is modified according to quantizer requirements
* '''config''' tensor parallel and expert parallel plans are updated by quantizer
* '''user_agent''' dictionary is updated with quantization method for telemetry

== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
from transformers import AutoConfig
from transformers.quantizers.auto import get_hf_quantizer
from transformers.utils.quantization_config import BitsAndBytesConfig

# Load a pre-quantized model configuration
config = AutoConfig.from_pretrained("TheBloke/Llama-2-7B-GPTQ")
user_agent = {}

# Get quantizer for pre-quantized checkpoint
hf_quantizer = get_hf_quantizer(
    config=config,
    quantization_config=None,
    device_map="auto",
    weights_only=False,
    user_agent=user_agent
)

print(f"Quantizer type: {type(hf_quantizer).__name__}")
print(f"Quantization method: {user_agent.get('quant')}")

# Load model with custom quantization config
config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4"
)

hf_quantizer = get_hf_quantizer(
    config=config,
    quantization_config=bnb_config,
    device_map="auto",
    weights_only=False,
    user_agent={}
)

# Check if quantization is supported
if hf_quantizer is None:
    print("No quantization configured")
else:
    print(f"Using {hf_quantizer.quantization_config.quant_method} quantization")

# Merge user config with checkpoint config (e.g., override loading attributes)
config = AutoConfig.from_pretrained("TheBloke/Llama-2-7B-GPTQ")
from transformers.utils.quantization_config import GPTQConfig

# Override specific loading parameters
user_gptq_config = GPTQConfig(
    bits=4,
    disable_exllama=True  # Override checkpoint setting
)

hf_quantizer = get_hf_quantizer(
    config=config,
    quantization_config=user_gptq_config,
    device_map={"": 0},  # Load everything on GPU 0
    weights_only=False,
    user_agent={}
)

# Device map may be updated by quantizer
print(f"Updated device map: {hf_quantizer.update_device_map({'': 0})}")

# Example with no quantization
config = AutoConfig.from_pretrained("google-bert/bert-base-uncased")
hf_quantizer = get_hf_quantizer(
    config=config,
    quantization_config=None,
    device_map=None,
    weights_only=False,
    user_agent={}
)
assert hf_quantizer is None, "BERT not quantized by default"
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Quantization_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_PyTorch]]
