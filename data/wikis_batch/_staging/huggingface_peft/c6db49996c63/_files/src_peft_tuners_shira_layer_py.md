# File: `src/peft/tuners/shira/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 217 |
| Classes | `ShiraLayer`, `Linear` |
| Imports | copy, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Documented

**Purpose:** Implements SHiRA (Sparse High Rank Adapter) layers that update only a sparse subset of weight elements selected by a mask for Linear layers.

**Mechanism:** Stores trainable shira_weight vector (length r*(in_features + out_features)) and shira_indices tensor (2 × num_weights) specifying which matrix elements to update. During update_layer, computes indices from binary mask. The get_delta_weight method constructs a sparse_coo_tensor from indices and weights scaled by scaling factor. Forward pass creates new_weight = base_layer.weight + get_delta_weight() and applies F.linear. Supports merge/unmerge by directly adding/subtracting sparse delta weights. Uses vector parameters instead of sparse tensors directly due to PyTorch sparse tensor training issues.

**Significance:** Core SHiRA implementation enabling high-rank sparse adaptation. Unlike LoRA's low-rank dense updates, SHiRA trains sparse high-rank updates, potentially capturing more complex adaptations. The sparse representation keeps memory efficient despite high rank. Only supports Linear layers currently. The set_scale method allows inference-time scaling.
