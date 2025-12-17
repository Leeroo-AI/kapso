# File: `src/peft/tuners/randlora/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 350 |
| Classes | `UniqueBaseGrad`, `RandLoraLayer`, `Linear` |
| Imports | _buffer_dict, peft, torch, transformers, typing, warnings |

## Understanding

**Status:** ✅ Documented

**Purpose:** Implements RandLoRA adapter layers for Linear operations using shared random bases (randlora_A, randlora_B) with per-layer trainable diagonal scaling (randlora_lambda, randlora_gamma).

**Mechanism:** Stores references to shared randlora_A/B BufferDicts (not trainable) and owns trainable randlora_lambda (r × num_bases) and randlora_gamma (num_bases × min_dim) parameters. The custom UniqueBaseGrad autograd function efficiently computes gradients for the scaled bases: output = lambda[:,:,None] * A * gamma[None,:]. During forward, slices appropriate submatrices from shared A/B, applies scaling, and computes linear(linear(x, B), A). Supports merge/unmerge with delta_weight computation.

**Significance:** Core RandLoRA implementation enabling extreme parameter efficiency by sharing random projections across all layers. Only lambda and gamma are trained per layer, dramatically reducing parameters compared to LoRA while maintaining competitive performance. The custom autograd function ensures memory-efficient training.
