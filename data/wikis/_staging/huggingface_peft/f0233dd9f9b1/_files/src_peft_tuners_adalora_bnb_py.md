# File: `src/peft/tuners/adalora/bnb.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 143 |
| Classes | `SVDLinear8bitLt`, `SVDLinear4bit` |
| Imports | layer, peft, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** BitsAndBytes quantized AdaLora layers

**Mechanism:** SVDLinear8bitLt/SVDLinear4bit extend AdaLoraLayer for 8-bit/4-bit quantized base layers. Handle dtype conversion for matmul operations (float32 for computations, cast to expected_dtype for output). SVDLinear4bit uses defensive clone for backprop compatibility.

**Significance:** Enables AdaLora on memory-efficient quantized models. Critical for combining parameter-efficient tuning with quantization for reduced memory footprint.
