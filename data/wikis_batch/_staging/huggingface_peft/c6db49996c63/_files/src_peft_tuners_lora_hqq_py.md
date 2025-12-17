# File: `src/peft/tuners/lora/hqq.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 251 |
| Classes | `HqqLoraLinear` |
| Functions | `dispatch_hqq` |
| Imports | __future__, copy, layer, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Documented

**Purpose:** LoRA adapter for HQQ (Half-Quadratic Quantization) quantized models

**Mechanism:** HqqLoraLinear wraps HQQ's quantized layers that use iterative half-quadratic optimization for quantization. Implements merge/unmerge with HQQ's specialized dequantization paths and handles HQQ's unique quantization metadata. Provides extensive warnings about merge compatibility and dtype requirements.

**Significance:** Enables fine-tuning with HQQ quantization, which achieves strong accuracy-compression tradeoffs through optimization-based quantization. Particularly useful for scenarios where HQQ's quality advantages over simpler quantization methods justify its computational overhead, providing another backend for memory-efficient adaptation.
