# File: `src/transformers/quantizers/quantizer_quanto.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 119 |
| Classes | `QuantoHfQuantizer` |
| Imports | base, quantizers_utils, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements integration with optimum-quanto library, providing flexible quantization supporting int8, fp8, int4, and int2 weight quantization without activation quantization.

**Mechanism:** Extends HfQuantizer requiring optimum-quanto and accelerate libraries. Replaces Linear layers with quanto's QModuleMixin layers via replace_with_quanto_layers. Implements custom memory and dtype adjustments: reduces max_memory by 10% for safety and maps quantization bits to appropriate dtypes (including CustomDtype for FP8/INT4/INT2). Determines quantization needs by checking if weights belong to QModuleMixin and aren't frozen. Explicitly disallows activation quantization in Transformers (directing users to quanto library directly for that).

**Significance:** User-friendly quantizer providing simple API for multiple quantization formats through quanto backend. Trainable but not serializable (requires re-quantization). Notable for supporting very low bitwidths (int2) and being explicit about scope limitations (no activation quantization in Transformers), encouraging advanced users to use quanto directly.
