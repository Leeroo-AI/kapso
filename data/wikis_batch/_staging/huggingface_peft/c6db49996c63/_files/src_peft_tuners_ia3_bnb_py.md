# File: `src/peft/tuners/ia3/bnb.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 129 |
| Classes | `Linear8bitLt`, `Linear4bit` |
| Imports | layer, peft, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** IA3 layer implementations for bitsandbytes quantized models (8-bit and 4-bit)

**Mechanism:** Linear8bitLt and Linear4bit wrap quantized base layers with ia3_l scaling vectors. For feedforward: result = base(x * ia3_scaling), for attention: result = base(x) * ia3_scaling. Applies dtype conversions for quantized weights, clones result for 4-bit due to backprop view manipulation issues

**Significance:** Extends IA3 to quantized models - enables extremely memory-efficient fine-tuning by combining learned rescaling vectors (IA3) with 8-bit/4-bit quantization, minimizing both parameter count and memory footprint
