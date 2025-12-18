# Heuristic: huggingface_transformers_Quantization_Selection

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Quantization Guide|https://huggingface.co/docs/transformers/quantization]]
|-
! Domains
| [[domain::Quantization]], [[domain::Inference]], [[domain::Memory]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
Choose 4-bit NF4 with double quantization for maximum memory savings (~4x), or 8-bit for better quality with 2x savings.

=== Description ===
Quantization reduces model memory footprint by using lower-precision representations. The choice between 4-bit and 8-bit depends on your quality vs. memory trade-off. NF4 (Normal Float 4-bit) with double quantization provides the best memory efficiency while maintaining acceptable quality for inference and fine-tuning.

=== Usage ===
Use 4-bit quantization when VRAM is severely limited (< 16GB for 7B models) or for QLoRA fine-tuning. Use 8-bit when you need better quality or for production inference where accuracy matters more than memory.

== The Insight (Rule of Thumb) ==

* **4-bit NF4:** Best for QLoRA fine-tuning, uses ~4GB for 7B model
* **4-bit FP4:** Slightly faster inference than NF4, marginally lower quality
* **8-bit INT8:** Best quality among quantized options, ~8GB for 7B model
* **Double Quantization:** Adds ~0.4 bits overhead but saves additional 10-15% memory

== Reasoning ==

Memory requirements per billion parameters:
- fp32: ~4GB per 1B params
- fp16/bf16: ~2GB per 1B params
- int8: ~1GB per 1B params
- int4: ~0.5GB per 1B params

Quality retention (typical):
- 8-bit: ~99% of fp16 quality
- 4-bit NF4: ~95-98% of fp16 quality
- 4-bit FP4: ~93-96% of fp16 quality

== Code Evidence ==

From `quantizers/auto.py:L65-86`:

<syntaxhighlight lang="python">
AUTO_QUANTIZER_MAPPING = {
    "bitsandbytes_4bit": Bnb4BitHfQuantizer,
    "bitsandbytes_8bit": Bnb8BitHfQuantizer,
    "gptq": GptqHfQuantizer,
    "awq": AwqQuantizer,
    "hqq": HqqHfQuantizer,
    "torchao": TorchAoHfQuantizer,
    # ... more methods
}
</syntaxhighlight>

BitsAndBytesConfig options from `quantization_config.py`:

<syntaxhighlight lang="python">
BitsAndBytesConfig(
    load_in_4bit: bool = False,              # Enable 4-bit
    load_in_8bit: bool = False,              # Enable 8-bit
    bnb_4bit_quant_type: str = "fp4",        # "fp4" or "nf4"
    bnb_4bit_compute_dtype: torch.dtype,     # Compute dtype (bf16 recommended)
    bnb_4bit_use_double_quant: bool = False, # Nested quantization
)
</syntaxhighlight>

== Example Usage ==

<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Maximum memory efficiency (QLoRA-style)
config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4 for best quality
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,      # Extra memory savings
)

# Better quality, still memory efficient
config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,              # Outlier threshold
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=config_4bit,
    device_map="auto",
)
</syntaxhighlight>

== Decision Matrix ==

{| class="wikitable"
|-
! Scenario !! Recommendation !! Memory (7B) !! Quality
|-
| QLoRA fine-tuning || 4-bit NF4 + double quant || ~4GB || Good
|-
| Inference, limited VRAM || 4-bit NF4 || ~4.5GB || Good
|-
| Production inference || 8-bit INT8 || ~8GB || Excellent
|-
| Best quality || No quantization (fp16) || ~14GB || Full
|-
| Research/experimentation || GPTQ/AWQ (pre-quantized) || ~4-5GB || Very Good
|}

== Important Notes ==

* **Keep lm_head in fp16:** The output head (`llm_int8_skip_modules=["lm_head"]`) should stay in higher precision for numerical stability
* **Compute dtype matters:** Use `bnb_4bit_compute_dtype=torch.bfloat16` on Ampere+ GPUs
* **Training support:** Only 8-bit and 4-bit with PEFT/LoRA support training; GPTQ/AWQ are inference-only

== Related Pages ==
* [[uses_heuristic::Implementation:huggingface_transformers_BitsAndBytesConfig_setup]]
* [[uses_heuristic::Implementation:huggingface_transformers_AutoHfQuantizer_dispatch]]
* [[uses_heuristic::Workflow:huggingface_transformers_Model_Quantization]]
