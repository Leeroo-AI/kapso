{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|QLoRA|https://arxiv.org/abs/2305.14314]]
* [[source::Doc|Transformers Quantization|https://huggingface.co/docs/transformers/quantization]]
|-
! Domains
| [[domain::Quantization]], [[domain::Memory_Efficiency]], [[domain::Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Concrete tool for configuring 4-bit quantization using bitsandbytes for memory-efficient QLoRA training.

=== Description ===

`BitsAndBytesConfig` configures the quantization parameters for loading models in 4-bit precision. This is the foundation of QLoRA, enabling training of large models on consumer GPUs. The NF4 (Normal Float 4-bit) quantization type provides optimal quantization for normally distributed weights.

=== Usage ===

Use this when setting up QLoRA training to reduce VRAM requirements by ~4x. Pass this config to `AutoModelForCausalLM.from_pretrained()`. Use `bnb_4bit_compute_dtype=torch.bfloat16` for faster computation on Ampere+ GPUs, or `torch.float16` for older GPUs.

== Code Reference ==

=== Source Location ===
* '''Library:''' [https://github.com/huggingface/transformers transformers]
* '''Class:''' `transformers.BitsAndBytesConfig`

=== Signature ===
<syntaxhighlight lang="python">
class BitsAndBytesConfig:
    def __init__(
        self,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        llm_int8_threshold: float = 6.0,
        llm_int8_skip_modules: Optional[List[str]] = None,
        llm_int8_enable_fp32_cpu_offload: bool = False,
        llm_int8_has_fp16_weight: bool = False,
        bnb_4bit_compute_dtype: Optional[torch.dtype] = None,
        bnb_4bit_quant_type: str = "fp4",
        bnb_4bit_use_double_quant: bool = False,
        bnb_4bit_quant_storage: Optional[torch.dtype] = None,
    ):
        """
        Configure quantization for model loading.

        Args:
            load_in_4bit: Enable 4-bit quantization
            bnb_4bit_compute_dtype: Compute dtype (torch.float16 or torch.bfloat16)
            bnb_4bit_quant_type: Quantization type ("nf4" recommended, "fp4" alternative)
            bnb_4bit_use_double_quant: Nested quantization for extra memory savings
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import BitsAndBytesConfig
import torch
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| load_in_4bit || bool || Yes || Enable 4-bit quantization. Set to True for QLoRA
|-
| bnb_4bit_compute_dtype || torch.dtype || No || Compute precision. bfloat16 for Ampere+, float16 for older
|-
| bnb_4bit_quant_type || str || No || "nf4" (recommended) or "fp4". NF4 is optimized for normal distributions
|-
| bnb_4bit_use_double_quant || bool || No || Nested quantization. Saves ~0.4 bits/param extra. Default: False
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| config || BitsAndBytesConfig || Configuration object for model loading
|}

== Usage Examples ==

=== Standard QLoRA Config ===
<syntaxhighlight lang="python">
from transformers import BitsAndBytesConfig
import torch

# Standard QLoRA configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 for Ampere GPUs
    bnb_4bit_quant_type="nf4",              # NF4 quantization
    bnb_4bit_use_double_quant=False,
)
</syntaxhighlight>

=== Maximum Memory Savings ===
<syntaxhighlight lang="python">
from transformers import BitsAndBytesConfig
import torch

# Maximum memory efficiency with double quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,  # Extra ~0.4 bits/param savings
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_peft_Quantization_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_peft_Quantization_Environment]]
