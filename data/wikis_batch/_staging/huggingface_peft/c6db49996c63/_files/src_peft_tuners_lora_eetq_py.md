# File: `src/peft/tuners/lora/eetq.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 118 |
| Classes | `EetqLoraLinear` |
| Functions | `dispatch_eetq` |
| Imports | peft, torch, typing |

## Understanding

**Status:** ✅ Documented

**Purpose:** LoRA adapter for EETQ (Efficient Embedding and Transformer Quantization) quantized layers

**Mechanism:** EetqLoraLinear wraps EETQ's quantized linear layers, handling the specialized quantization format that focuses on embedding and attention layer efficiency. Implements custom merge/unmerge logic that dequantizes, applies LoRA deltas, and requantizes using EETQ's quantization scheme.

**Significance:** Enables fine-tuning of models using EETQ quantization, which optimizes specifically for transformer architectures. Provides another quantization backend option for memory-efficient LoRA training, particularly useful when EETQ's optimization profile matches the target hardware.
