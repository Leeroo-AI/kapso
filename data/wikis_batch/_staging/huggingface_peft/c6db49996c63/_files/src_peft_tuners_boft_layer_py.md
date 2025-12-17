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

**Purpose:** Implementation of BOFT layer adaptations for Linear and Conv2d layers

**Mechanism:** Uses Cayley parametrization on skew-symmetric matrices to generate orthogonal transformations, applies butterfly permutations and block diagonal operations (with optional CUDA acceleration), combines with scaling factors; supports multiplicative dropout

**Significance:** Core implementation of BOFT's butterfly-factorized orthogonal transformations that enable parameter-efficient finetuning while maintaining orthogonality constraints
