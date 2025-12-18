{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Model Loading Documentation|https://huggingface.co/docs/transformers/main_classes/model]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Model_Management]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Concrete tool for instantiating appropriate quantization handlers from configuration objects provided by HuggingFace Transformers.

=== Description ===
AutoHfQuantizer.from_config is the factory method that maps quantization configurations to their corresponding quantizer implementations. It handles the complexity of supporting multiple quantization methods (bitsandbytes, GPTQ, AWQ, GGUF, AQLM) through a unified interface. The method performs intelligent dispatching based on the quantization method specified in the configuration, with special handling for bitsandbytes which has separate 4-bit and 8-bit variants. It validates that the requested quantization method is supported and returns a properly configured quantizer object that will be used during model weight loading.

This implementation is crucial for the plugin-style architecture that allows the Transformers library to support new quantization methods without modifying core loading logic.

=== Usage ===
AutoHfQuantizer.from_config is called internally during the model loading process when a quantization configuration is detected in the model config. Advanced users might call it directly when implementing custom model loading pipelines or when pre-validating quantization configurations before committing to a full model load.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/quantizers/auto.py
* '''Lines:''' 161-184

=== Signature ===
<syntaxhighlight lang="python">
@classmethod
def from_config(
    cls,
    quantization_config: QuantizationConfigMixin | dict,
    **kwargs
) -> HfQuantizer
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers.quantizers import AutoHfQuantizer
from transformers import BitsAndBytesConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| quantization_config || QuantizationConfigMixin or dict || Yes || Quantization configuration object or dictionary specifying quantization method and parameters
|-
| **kwargs || dict || No || Additional arguments to pass to the quantizer constructor
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| quantizer || HfQuantizer || Instantiated quantizer object (subclass varies by quantization method)
|}

== Usage Examples ==

=== Loading 4-bit Quantized Model ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Define 4-bit quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# AutoHfQuantizer.from_config is called internally
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)

print(f"Model memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
</syntaxhighlight>

=== Loading 8-bit Quantized Model ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 8-bit quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)
</syntaxhighlight>

=== Loading GPTQ Quantized Model ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, GPTQConfig

# GPTQ quantization config
quantization_config = GPTQConfig(
    bits=4,
    group_size=128,
    desc_act=False
)

# Load pre-quantized GPTQ model
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-GPTQ",
    quantization_config=quantization_config,
    device_map="auto"
)
</syntaxhighlight>

=== Loading AWQ Quantized Model ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, AwqConfig

# AWQ quantization config
quantization_config = AwqConfig(
    bits=4,
    group_size=128,
    version="gemm"
)

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-AWQ",
    quantization_config=quantization_config,
    device_map="auto"
)
</syntaxhighlight>

=== Direct Quantizer Usage (Advanced) ===
<syntaxhighlight lang="python">
from transformers.quantizers import AutoHfQuantizer
from transformers import BitsAndBytesConfig

# Create quantization config
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16"
)

# Manually instantiate quantizer (normally done internally)
quantizer = AutoHfQuantizer.from_config(config)

print(f"Quantizer type: {type(quantizer)}")
print(f"Supports {quantizer.quant_method} quantization")

# Check if quantizer is available (has required dependencies)
if quantizer.is_available():
    print("Quantizer is ready to use")
else:
    print("Missing dependencies for this quantization method")
</syntaxhighlight>

=== Comparing Memory Footprints ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_name = "meta-llama/Llama-2-7b-hf"

# Load in different quantization modes
configs = {
    "4-bit": BitsAndBytesConfig(load_in_4bit=True),
    "8-bit": BitsAndBytesConfig(load_in_8bit=True),
}

for mode, config in configs.items():
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=config,
        device_map="auto"
    )
    memory_gb = model.get_memory_footprint() / 1e9
    print(f"{mode} quantization: {memory_gb:.2f} GB")
    del model
    torch.cuda.empty_cache()
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Quantization_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Loading_Environment]]
