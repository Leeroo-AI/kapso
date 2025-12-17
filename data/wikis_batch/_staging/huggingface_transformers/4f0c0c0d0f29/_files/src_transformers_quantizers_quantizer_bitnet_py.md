# File: `src/transformers/quantizers/quantizer_bitnet.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 109 |
| Classes | `BitNetHfQuantizer` |
| Imports | base, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements 1.58-bit quantization for the BitNet method (https://huggingface.co/papers/2402.17764), an extreme quantization approach that represents weights using only three values {-1, 0, 1}. This quantizer converts linear layers to BitLinear layers during model loading.

**Mechanism:** The `BitNetHfQuantizer` requires calibration and GPU for efficient inference. During preprocessing, `replace_with_bitnet_linear()` converts linear layers to BitLinear implementations, respecting modules_to_not_convert. Validates that device_map doesn't include CPU/disk for multi-device scenarios (warns for single-device CPU loads). Adjusts target_dtype to int8 and reduces max_memory to 90% for quantization overhead. Training and QAT (quantization-aware training) support is conditional on using "autobitlinear" class with "online" quantization mode.

**Significance:** Represents cutting-edge extreme quantization research, enabling models to run with ~1.58 bits per weight versus typical 16-bit or 4-bit methods. This dramatic compression enables deployment on very resource-constrained devices, though with inference speed considerations on CPU due to unpacking overhead.
