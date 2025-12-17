# File: `src/transformers/quantizers/quantizer_bnb_8bit.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 172 |
| Classes | `Bnb8BitHfQuantizer` |
| Imports | base, quantizers_utils, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements 8-bit quantization using the BitsAndBytes library, enabling INT8 quantization for memory-efficient model loading and inference. This quantizer supports both pre-quantized models and on-the-fly quantization during loading.

**Mechanism:** The `Bnb8BitHfQuantizer` sets `requires_calibration = False`, allowing dynamic quantization. Validates bitsandbytes library and backend availability, enforcing GPU device placement unless `llm_int8_enable_fp32_cpu_offload=True`. During preprocessing, `replace_with_bnb_linear()` converts linear layers to `Linear8bitLt` modules. Automatically sets device_map to current GPU if not specified. Adjusts target_dtype to int8 and max_memory to 90%. Supports full training, dequantization via `dequantize_and_replace()`, and weight conversion through `Bnb8bitDeserialize` for pre-quantized models.

**Significance:** BitsAndBytes is one of the most widely-used quantization libraries in the LLM ecosystem, pioneered by Tim Dettmers. The 8-bit quantization provides a good balance between memory savings (~50% reduction) and accuracy preservation, making it popular for fine-tuning large models. The CPU offload capability enables handling models larger than GPU memory.
