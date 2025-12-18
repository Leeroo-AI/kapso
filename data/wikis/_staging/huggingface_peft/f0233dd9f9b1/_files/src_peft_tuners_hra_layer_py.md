# File: `src/peft/tuners/hra/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 461 |
| Classes | `HRALayer`, `HRALinear`, `HRAConv2d` |
| Imports | math, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements HRA layer classes that apply sequences of Householder reflections to create orthogonal weight transformations for parameter-efficient fine-tuning.

**Mechanism:** HRALayer is the base class storing hra_u (Householder reflection vectors), hra_r (number of reflections), and hra_apply_GS (Gram-Schmidt flag). The update_layer() method initializes r reflection vectors with symmetric initialization (pairs of opposite-signed vectors for even r) or Kaiming initialization (for odd r). The reset_hra_parameters() and reset_hra_parameters_random() methods handle initialization. The apply_householder() method computes weight updates by iteratively applying reflections: W' = W - 2*u*(u^T*W)/(u^T*u), optionally applying Gram-Schmidt orthogonalization. HRALinear implements forward() for linear layers by computing base output + householder-transformed residual. HRAConv2d handles 2D convolutions by reshaping weights to 2D matrices, applying Householder reflections, and reshaping back. Both support merge/unmerge operations via get_delta_weight().

**Significance:** These classes implement HRA's core algorithm (https://huggingface.co/papers/2405.17484): using r Householder reflections H_i = I - 2*u_i*u_i^T/||u_i||^2 to transform weights orthogonally. Unlike low-rank methods that reduce dimensionality, Householder reflections preserve rank while rotating the weight space. This is geometrically elegant and parameter-efficient (only r vectors of size d). The method is particularly effective for vision models where preserving spatial structure matters, and the Conv2d support makes it unique among PEFT methods.
