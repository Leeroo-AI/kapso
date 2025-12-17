# File: `src/peft/tuners/ia3/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 330 |
| Classes | `IA3Layer`, `Linear`, `_ConvNd`, `Conv2d`, `Conv3d` |
| Imports | peft, torch, transformers, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Core IA3 layer implementations for linear and convolutional layers

**Mechanism:** IA3Layer maintains ia3_l scaling vectors ((1, in_features) for feedforward, (out_features, 1) for attention), initialized to 1.0. Linear applies element-wise scaling: feedforward scales inputs x*ia3_l before base layer, attention scales outputs after. Conv2d/Conv3d (_ConvNd) handle multi-dimensional kernels. Supports merge/unmerge by multiplying/dividing base weights

**Significance:** Implements minimal parameter adaptation - each layer adds only O(d) parameters vs O(d^2) for full fine-tuning, achieving competitive performance through strategic activation rescaling at key bottleneck points
