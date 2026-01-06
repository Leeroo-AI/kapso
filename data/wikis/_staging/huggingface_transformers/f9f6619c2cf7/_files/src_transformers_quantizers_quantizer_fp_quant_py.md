# File: `src/transformers/quantizers/quantizer_fp_quant.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 150 |
| Classes | `FPQuantHfQuantizer` |
| Imports | base, quantizers_utils, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements FP-Quant (floating-point quantization) method supporting both real quantization on Blackwell GPUs via qutlass kernels and pseudo-quantization for emulation.

**Mechanism:** Extends HfQuantizer with dual-mode support: real quantization (requires Blackwell GPU + qutlass) or pseudo-quantization (Triton-based emulation without speedup). Replaces Linear layers with FPQuantLinear modules. Enforces bfloat16 dtype and validates device requirements. Provides weight conversion handling for both qweight (real) and dqweight (pseudo) formats. Supports training only when store_master_weights=True, maintaining full-precision copies alongside quantized weights.

**Significance:** Cutting-edge quantizer for latest GPU hardware (Blackwell) while offering pseudo-quantization fallback for testing/validation. QAT-trainable and serializable, making it suitable for research and fine-tuning workflows. Unique in requiring specific GPU generation for optimal performance.
