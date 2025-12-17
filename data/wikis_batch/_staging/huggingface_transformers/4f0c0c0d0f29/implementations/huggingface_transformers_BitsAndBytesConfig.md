{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
* [[source::Paper|QLoRA|https://arxiv.org/abs/2305.14314]]
|-
! Domains
| [[domain::NLP]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Configuration class for BitsAndBytes quantization parameters supporting 8-bit (LLM.int8) and 4-bit (FP4/NF4) precision.

=== Description ===

BitsAndBytesConfig encapsulates all parameters needed for quantization using the bitsandbytes library. It supports both 8-bit quantization with outlier handling (LLM.int8) and 4-bit quantization with FP4/NF4 data types. The configuration controls precision, compute dtype, double quantization, and module exclusions.

=== Usage ===

Pass this configuration to `from_pretrained()` to load a model in quantized format or to quantize a model during loading. Use 4-bit for maximum memory savings or 8-bit for better accuracy. The configuration is saved with quantized models for consistency during reload.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/utils/quantization_config.py

=== Signature ===
<syntaxhighlight lang="python">
class BitsAndBytesConfig(QuantizationConfigMixin):
    def __init__(
        self,
        load_in_8bit=False,
        load_in_4bit=False,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=False,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=None,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_storage=None,
        **kwargs,
    )
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import BitsAndBytesConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| load_in_8bit || bool || No || Enable 8-bit quantization (LLM.int8)
|-
| load_in_4bit || bool || No || Enable 4-bit quantization (FP4/NF4)
|-
| llm_int8_threshold || float || No || Outlier threshold for 8-bit (default 6.0)
|-
| llm_int8_skip_modules || list[str] || No || Modules to keep in full precision
|-
| llm_int8_enable_fp32_cpu_offload || bool || No || Allow CPU offloading in FP32
|-
| llm_int8_has_fp16_weight || bool || No || Keep FP16 weights for fine-tuning
|-
| bnb_4bit_compute_dtype || torch.dtype || No || Compute dtype for 4-bit (default float32)
|-
| bnb_4bit_quant_type || str || No || Quantization type: "fp4" or "nf4" (default "fp4")
|-
| bnb_4bit_use_double_quant || bool || No || Enable nested quantization (default False)
|-
| bnb_4bit_quant_storage || torch.dtype || No || Storage type for packed params (default uint8)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| config || BitsAndBytesConfig || Validated configuration object
|}

== Usage Examples ==

=== 4-bit NF4 Quantization (QLoRA) ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# QLoRA configuration - optimal for fine-tuning
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

print(f"Model memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
# ~3.5 GB vs ~14 GB for FP16
</syntaxhighlight>

=== 8-bit LLM.int8() Quantization ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 8-bit configuration with outlier handling
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=["lm_head"],
)

model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-7b1",
    quantization_config=bnb_config,
    device_map="auto",
)
# ~7 GB vs ~14 GB for FP16, better accuracy than 4-bit
</syntaxhighlight>

=== Mixed Precision Configuration ===
<syntaxhighlight lang="python">
# 4-bit storage, BF16 compute for best speed/accuracy trade-off
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["lm_head", "embed_tokens"],
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
</syntaxhighlight>

=== CPU Offloading ===
<syntaxhighlight lang="python">
# Allow some layers on CPU in FP32
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

device_map = {
    "transformer.h.0": 0,  # GPU
    "transformer.h.1": 0,
    # ...
    "transformer.h.30": "cpu",  # CPU
    "transformer.h.31": "cpu",
}

model = AutoModelForCausalLM.from_pretrained(
    "gpt-j-6b",
    quantization_config=bnb_config,
    device_map=device_map,
)
</syntaxhighlight>

== Configuration Details ==

=== Quantization Types ===

'''FP4 (4-bit Floating Point):'''
* 16 evenly distributed floating-point values
* Good for general weights distribution

'''NF4 (4-bit NormalFloat):'''
* Information-theoretically optimal for N(0,1)
* Recommended for most language models
* Used in QLoRA paper

=== Double Quantization ===

When `bnb_4bit_use_double_quant=True`:
* Quantizes the quantization constants (scales)
* Saves ~0.4 bits per parameter
* ~3% additional memory savings
* Minimal accuracy impact

=== Compute Dtype Selection ===

{| class="wikitable"
|-
! Dtype !! Memory !! Speed !! Accuracy !! Use Case
|-
| float32 || High || Slow || Best || Maximum accuracy needed
|-
| float16 || Medium || Fast || Good || Most GPUs, good default
|-
| bfloat16 || Medium || Fast || Good || Ampere+ GPUs, training
|}

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Quantization_Config_Setup]]

=== Used By ===
* [[used_by::Implementation:huggingface_transformers_get_hf_quantizer_init]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_CUDA]]
* [[requires_env::Environment:huggingface_transformers_BitsAndBytes]]
