# File: `src/peft/tuners/boft/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 1011 |
| Classes | `FastBlockDiag`, `MultiplicativeDropoutLayer`, `BOFTLayer`, `Linear`, `Conv2d` |
| Functions | `patch_environment`, `get_fbd_cuda` |
| Imports | __future__, contextlib, math, os, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** BOFT layer implementations

**Mechanism:** BOFTLayer uses boft_R (skew-symmetric params), boft_s (scaling), boft_P (permutation matrices). cayley_batch() converts skew-symmetric to orthogonal matrices via Cayley parametrization. FastBlockDiag (custom CUDA autograd) accelerates block diagonal operations. MultiplicativeDropoutLayer randomly sets blocks to identity. Linear/Conv2d implement forward with butterfly OFT matrix multiplication.

**Significance:** Core BOFT implementation. Cayley parametrization ensures orthogonality. CUDA extension (get_fbd_cuda) dramatically accelerates butterfly factorization. Critical for efficient orthogonal fine-tuning on Linear and Conv2d layers.
