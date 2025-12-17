# File: `src/peft/tuners/hra/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 461 |
| Classes | `HRALayer`, `HRALinear`, `HRAConv2d` |
| Imports | math, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements HRA adapter layers using Householder reflections for orthogonal weight transformations.

**Mechanism:** HRALayer stores hra_u parameters (in_features × r vectors). get_delta_weight computes orthogonal matrix via: (1) Gram-Schmidt mode: normalizes vectors, builds reflection matrix at once; (2) Iterative mode: applies sequential reflections H_i = I - 2*u_i*u_i^T. HRALinear/HRAConv2d forward passes multiply base weights by composed orthogonal transformation. Symmetric initialization uses repeated half-rank patterns. Supports reverse transformations for unmerge.

**Significance:** Core implementation of Householder reflection-based adaptation. Each reflection is a rank-1 update producing an orthogonal matrix. Composing r reflections creates flexible orthogonal transformations while maintaining parameter efficiency (only r*in_features parameters). Works with both Linear and Conv2d layers.
