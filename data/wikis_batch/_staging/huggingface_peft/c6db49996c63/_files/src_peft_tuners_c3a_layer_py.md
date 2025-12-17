# File: `src/peft/tuners/c3a/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 202 |
| Classes | `C3ALayer`, `C3ALinear` |
| Imports | __future__, math, peft, torch, typing, utils, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements C3A adapter layers that apply block circulant convolution to linear layers for parameter-efficient fine-tuning.

**Mechanism:** C3ALayer maintains c3a_kernel parameters (shape: out_features//block_size × in_features//block_size × block_size). C3ALinear's forward pass uses BlockCircularConvolution (FFT-based) for efficient computation. get_delta_weight reconstructs full circulant matrix using get_circulant_fast. Supports merge/unmerge operations for adapter weight integration.

**Significance:** Core implementation of C3A's block circulant matrix approach. Uses FFT operations in fp32 for numerical stability (fp16/bf16 have limited FFT support). Achieves parameter efficiency by storing only block diagonal circulant representations instead of full weight matrices.
