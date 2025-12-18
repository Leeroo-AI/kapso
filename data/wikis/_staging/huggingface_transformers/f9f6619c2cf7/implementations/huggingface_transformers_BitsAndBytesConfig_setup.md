{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Quantization Documentation|https://huggingface.co/docs/transformers/main_classes/quantization]]
* [[source::Repo|bitsandbytes|https://github.com/TimDettmers/bitsandbytes]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Concrete configuration object for bitsandbytes quantization provided by HuggingFace Transformers.

=== Description ===
BitsAndBytesConfig implements the Quantization_Config principle by providing a typed, validated configuration object for the bitsandbytes quantization library. It supports LLM.int8(), FP4, and NF4 quantization schemes with comprehensive control over computation precision, storage format, outlier handling, and module-level exceptions. The class performs mutual exclusivity validation (cannot load in both 4-bit and 8-bit simultaneously) and type coercion for torch.dtype parameters.

=== Usage ===
Import and instantiate BitsAndBytesConfig when you need to:
* Configure 4-bit or 8-bit quantization for model loading via `from_pretrained`
* Specify NF4 vs FP4 quantization types for 4-bit models
* Enable double quantization to further reduce memory
* Set compute dtype (e.g., bfloat16) separate from storage dtype
* Define modules to skip during quantization (e.g., lm_head)
* Configure INT8 outlier detection thresholds

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/utils/quantization_config.py
* '''Lines:''' 387-530

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
    ):
        """
        Configuration class for bitsandbytes quantization.

        Args:
            load_in_8bit (bool, optional): Enable 8-bit quantization with LLM.int8(). Defaults to False.
            load_in_4bit (bool, optional): Enable 4-bit quantization with FP4/NF4. Defaults to False.
            llm_int8_threshold (float, optional): Outlier threshold for LLM.int8(). Defaults to 6.0.
            llm_int8_skip_modules (list[str], optional): Modules to keep in original dtype.
            llm_int8_enable_fp32_cpu_offload (bool, optional): Enable CPU offload for int8. Defaults to False.
            llm_int8_has_fp16_weight (bool, optional): Keep weights in FP16 for fine-tuning. Defaults to False.
            bnb_4bit_compute_dtype (torch.dtype or str, optional): Compute dtype for 4-bit. Defaults to torch.float32.
            bnb_4bit_quant_type (str, optional): Quantization type: "fp4" or "nf4". Defaults to "fp4".
            bnb_4bit_use_double_quant (bool, optional): Enable nested quantization. Defaults to False.
            bnb_4bit_quant_storage (torch.dtype or str, optional): Storage dtype for 4-bit params. Defaults to torch.uint8.
        """
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
| load_in_8bit || bool || No || Enable 8-bit quantization (default: False)
|-
| load_in_4bit || bool || No || Enable 4-bit quantization (default: False)
|-
| llm_int8_threshold || float || No || Outlier detection threshold (default: 6.0)
|-
| llm_int8_skip_modules || list[str] || No || Module names to skip quantization
|-
| llm_int8_enable_fp32_cpu_offload || bool || No || Enable CPU offload for mixed precision (default: False)
|-
| llm_int8_has_fp16_weight || bool || No || Keep FP16 weights for training (default: False)
|-
| bnb_4bit_compute_dtype || torch.dtype or str || No || Computation dtype (default: torch.float32)
|-
| bnb_4bit_quant_type || str || No || "fp4" or "nf4" (default: "fp4")
|-
| bnb_4bit_use_double_quant || bool || No || Enable double quantization (default: False)
|-
| bnb_4bit_quant_storage || torch.dtype or str || No || Storage dtype (default: torch.uint8)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| config || BitsAndBytesConfig || Validated quantization configuration object
|}

== Usage Examples ==

=== 4-bit NF4 Quantization with BFloat16 Compute ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Configure 4-bit quantization with NF4 and bfloat16 compute
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto",
)

# Model is now loaded in 4-bit NF4 format
# Memory usage: ~3.5GB instead of ~14GB for FP32
</syntaxhighlight>

=== 8-bit Quantization with Module Skipping ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Configure 8-bit quantization, keeping lm_head in full precision
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=["lm_head"],
)

model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-3b",
    quantization_config=quantization_config,
    device_map="auto",
)

# Model uses INT8 quantization except for lm_head
# Memory usage: ~3GB instead of ~12GB for FP32
</syntaxhighlight>

=== Custom Threshold for Unstable Models ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Lower threshold for fine-tuned or smaller models
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=4.0,  # More aggressive outlier detection
    llm_int8_has_fp16_weight=True,  # Keep FP16 for potential fine-tuning
)

model = AutoModelForCausalLM.from_pretrained(
    "my-finetuned-model",
    quantization_config=quantization_config,
    device_map="auto",
)
</syntaxhighlight>

=== Maximum Memory Efficiency with Double Quantization ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Maximum compression: 4-bit NF4 + double quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # Quantize the quantization constants
    bnb_4bit_quant_storage=torch.uint8,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    quantization_config=quantization_config,
    device_map="auto",
)

# 13B model fits in ~6.5GB instead of ~52GB
# Enables running on consumer GPUs
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Quantization_Config]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Quantization_Environment]]

=== Used By ===
* [[used_by::Implementation:huggingface_transformers_AutoHfQuantizer_dispatch]]
