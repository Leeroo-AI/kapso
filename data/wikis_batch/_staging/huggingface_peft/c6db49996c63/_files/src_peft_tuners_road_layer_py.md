# File: `src/peft/tuners/road/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 418 |
| Classes | `RoadLayer`, `Linear` |
| Functions | `_get_delta_weight`, `_prepare_cols`, `_apply_road`, `dispatch_default` |
| Imports | config, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Documented

**Purpose:** Implements RoAd (Rotation Adaptation) layers that apply learned 2D rotations to pairs of features using trainable angle (road_theta) and scale (road_alpha) parameters.

**Mechanism:** Stores road_theta and road_alpha parameters with sizes depending on variant (road_1: out_features/2, road_2: out_features, road_4: out_features*2). The _apply_road function splits features into groups of size group_size, pairs elements from first and second halves, and applies rotation: y0 = x0*alpha*cos(theta) - xn*alpha*sin(theta), yn = x0*alpha*sin(theta) + xn*alpha*cos(theta). Supports mixed batch inference with different adapters per sample via adapter_names argument. Merge/unmerge operations use full rotation matrix R @ W formula, with inverse computed via torch.linalg.inv for unmerge.

**Significance:** Core RoAd implementation using efficient element-wise rotations instead of full matrix multiplication. The rotation-based approach is parameter-efficient and offers different inductive biases than rank-based methods. The group_size parameter enables optimizations in inference frameworks like VLLM. Based on https://huggingface.co/papers/2409.00119.
