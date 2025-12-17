# File: `src/transformers/quantizers/quantizer_eetq.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 108 |
| Classes | `EetqHfQuantizer` |
| Imports | base, quantizers_utils, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements 8-bit quantization using EETQ (Efficient Embedding Table Quantization) method, which provides fast INT8 inference on CUDA GPUs. This quantizer supports both pre-quantized models and dynamic quantization.

**Mechanism:** The `EetqHfQuantizer` sets `requires_calibration = False` allowing on-the-fly quantization. Requires CUDA GPU and validates kernels library availability. During preprocessing, `replace_with_eetq_linear()` swaps linear layers with `EetqLinear` modules. Validates device_map doesn't include CPU/disk for multi-device scenarios. Suggests float16 dtype for optimal efficiency. The `param_needs_quantization()` method checks if parameters belong to EetqLinear modules and aren't biases. Fully trainable and serializable, with `EetqQuantize` operations provided for quantization.

**Significance:** EETQ offers efficient 8-bit quantization optimized for inference speed on CUDA devices, competing with BitsAndBytes. The focus on CUDA-only deployment reflects its optimization for production inference scenarios. Training support makes it viable for both fine-tuning and deployment workflows.
