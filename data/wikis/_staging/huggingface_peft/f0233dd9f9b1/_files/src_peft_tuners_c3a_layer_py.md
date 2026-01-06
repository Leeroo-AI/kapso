# File: `src/peft/tuners/c3a/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 202 |
| Classes | `C3ALayer`, `C3ALinear` |
| Imports | __future__, math, peft, torch, typing, utils, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** C3A layer implementations

**Mechanism:** C3ALayer stores c3a_kernel (out_features//block_size x in_features//block_size x block_size in fp32 for FFT). get_delta_weight() converts kernel to circulant matrix via get_circulant_fast(). C3ALinear forward applies BlockCircularConvolution.apply() with kernel. Supports xavier_uniform, kaiming_uniform, gaussian initializations.

**Significance:** Core layer implementing circular convolution adapter. FFT-based BlockCircularConvolution enables efficient block circulant operations. fp32 requirement for FFT compatibility. Essential for C3A parameter-efficient adaptation.
