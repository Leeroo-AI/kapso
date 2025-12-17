# File: `src/peft/tuners/road/bnb.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 407 |
| Classes | `Linear8bitLt`, `Linear4bit` |
| Functions | `dispatch_bnb_8bit`, `dispatch_bnb_4bit` |
| Imports | __future__, bitsandbytes, config, layer, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Documented

**Purpose:** Provides RoAd implementations for bitsandbytes quantized layers (8-bit and 4-bit), enabling RoAd with quantized base models.

**Mechanism:** Both classes extend RoadLayer and implement merge/unmerge by dequantizing weights, computing rotation matrix R from theta/alpha parameters, applying R @ W (and R @ bias if present), and re-quantizing. Forward pass applies _apply_road to activations without dequantization for efficiency. Merge warns about potential rounding errors from quantization. Includes dispatch_bnb_8bit and dispatch_bnb_4bit factory functions for creating appropriate instances.

**Significance:** Enables using RoAd with quantized models for memory-efficient fine-tuning. The rotation-based adaptation works naturally with quantized weights since rotations are applied to activations during inference. Critical for running large models on limited hardware while maintaining RoAd's benefits.
