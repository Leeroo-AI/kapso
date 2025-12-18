# File: `src/transformers/quantizers/quantizer_eetq.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 108 |
| Classes | `EetqHfQuantizer` |
| Imports | base, quantizers_utils, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements 8-bit quantization using the EETQ (Efficient Efficient Transformer Quantization) method for GPU-accelerated model compression.

**Mechanism:** Extends HfQuantizer to replace standard Linear layers with EetqLinear modules before weight loading. Validates GPU availability and proper device mapping (CPU/disk not supported). Determines which parameters need quantization by checking if they belong to EetqLinear modules. Provides quantize operations through EetqQuantize and supports both serialization and training.

**Significance:** Specialized quantizer for EETQ 8-bit quantization, requiring CUDA GPU and the kernels package. Designed for memory-efficient inference with training support, making it suitable for fine-tuning quantized models.
