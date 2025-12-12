{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|bitsandbytes GitHub|https://github.com/TimDettmers/bitsandbytes]]
* [[source::Paper|QLoRA Paper|https://arxiv.org/abs/2305.14314]]
* [[source::Doc|HuggingFace Quantization|https://huggingface.co/docs/transformers/quantization]]
|-
! Domains
| [[domain::Quantization]], [[domain::Memory_Optimization]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Concrete tool for 4-bit model quantization using bitsandbytes NF4 format, enabling large model training on consumer GPUs.

=== Description ===
The bitsandbytes library provides 4-bit quantization that reduces model memory by ~75% with minimal quality loss. The NF4 (NormalFloat4) format is specifically designed for normally-distributed neural network weights. Unsloth integrates seamlessly with bitsandbytes, automatically applying 4-bit quantization when `load_in_4bit=True`.

=== Usage ===
This implementation is used automatically when loading models with Unsloth's `load_in_4bit=True` parameter. Understanding it helps diagnose issues and optimize memory further. Essential for running 7B+ models on 16GB or less VRAM.

== Code Signature ==
<syntaxhighlight lang="python">
# Via Unsloth (recommended)
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "...",
    load_in_4bit = True,  # Uses bitsandbytes NF4
)

# Direct bitsandbytes configuration
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",         # or "fp4"
    bnb_4bit_compute_dtype: torch.dtype = torch.float16,
    bnb_4bit_use_double_quant: bool = True,   # Nested quantization
)
</syntaxhighlight>

== I/O Contract ==
* **Consumes:**
    * Model weights in FP16/BF16/FP32 format
    * Configuration for quantization type and compute dtype
* **Produces:**
    * Quantized model with ~75% memory reduction
    * Preserved forward/backward computation capability

== Memory Comparison ==
{| class="wikitable"
! Model Size !! FP16 VRAM !! 4-bit VRAM !! Reduction
|-
|| 7B || ~14GB || ~4GB || 71%
|-
|| 13B || ~26GB || ~7GB || 73%
|-
|| 70B || ~140GB || ~35GB || 75%
|}

== Example Usage ==
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Simple approach - Unsloth handles everything
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",  # Pre-quantized
    max_seq_length = 2048,
    load_in_4bit = True,
)

# Or quantize on-the-fly
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Meta-Llama-3-8B",   # Full precision source
    max_seq_length = 2048,
    load_in_4bit = True,   # Quantize during loading
)
</syntaxhighlight>

== Related Pages ==
=== Context & Requirements ===
* [[requires_env::Environment:Unsloth_CUDA_Environment]]

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:AdamW_8bit_Optimizer_Usage]]

