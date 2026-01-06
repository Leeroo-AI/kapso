# File: `src/transformers/quantizers/quantizer_bnb_8bit.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 172 |
| Classes | `Bnb8BitHfQuantizer` |
| Imports | base, quantizers_utils, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements 8-bit quantizer using the bitsandbytes library, enabling INT8 quantization with minimal accuracy loss.

**Mechanism:** The `Bnb8BitHfQuantizer` class extends `HfQuantizer` with `requires_calibration=False`, allowing both pre-quantized and on-the-fly quantization. `validate_environment` checks for accelerate and bitsandbytes, plus backend availability (CUDA/XPU/NPU/HPU). It validates device_map to prevent CPU/disk offloading unless `llm_int8_enable_fp32_cpu_offload=True`. `update_device_map` auto-selects device if none specified. `adjust_target_dtype` returns torch.int8 for memory planning. In preprocessing, `replace_with_bnb_linear` converts layers to `Linear8bitLt` modules. Supports dequantization via `_dequantize`. Provides weight conversion logic for loading pre-quantized weights using `Bnb8bitDeserialize`.

**Significance:** One of the most popular and accessible quantization methods, providing good quality-performance tradeoff at 8-bit precision. Fully trainable, making it ideal for fine-tuning large models in resource-constrained environments. Core component of the quantization toolkit widely used in the community.
