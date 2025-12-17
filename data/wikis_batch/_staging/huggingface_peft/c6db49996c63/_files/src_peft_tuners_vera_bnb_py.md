# File: `src/peft/tuners/vera/bnb.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 411 |
| Classes | `Linear8bitLt`, `Linear4bit` |
| Imports | __future__, bitsandbytes, layer, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements VeRA adapter layers for quantized models (8-bit and 4-bit) using bitsandbytes.

**Mechanism:** Provides Linear8bitLt and Linear4bit classes that extend VeraLayer to support quantized base weights. Implements merge/unmerge operations by dequantizing weights, applying VeRA delta (lambda_b * vera_B @ lambda_d * vera_A), and re-quantizing. Forward pass applies adaptation without dequantization via sequential linear operations.

**Significance:** Enables VeRA to work with memory-efficient quantized models, maintaining parameter efficiency while reducing memory footprint. Critical for deploying VeRA on resource-constrained hardware with large language models.
