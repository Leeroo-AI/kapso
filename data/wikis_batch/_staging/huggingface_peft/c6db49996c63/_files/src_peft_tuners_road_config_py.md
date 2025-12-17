# File: `src/peft/tuners/road/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 126 |
| Classes | `RoadConfig` |
| Imports | __future__, dataclasses, peft, typing |

## Understanding

**Status:** ✅ Documented

**Purpose:** Configuration for RoAd (Rotation Adaptation) adapters, defining three variants (road_1, road_2, road_4) with different parameter counts and group_size for efficient rotation operations.

**Mechanism:** Extends PeftConfig with variant (Literal["road_1", "road_2", "road_4"]) controlling parameter count (road_1: out_features/2, road_2: out_features, road_4: out_features*2), and group_size (default 64) for grouping elements into 2D vectors for rotation. Validates that variant is valid and group_size is positive and even. The RoadVariant type alias ensures type safety.

**Significance:** RoAd adapts models through learned 2D rotations of feature pairs, offering a unique alternative to rank-based methods. The three variants trade off between parameter efficiency and expressiveness. The group_size affects inference speed but not model performance (elements are unordered). Requires hidden size divisible by group_size. Paper: https://huggingface.co/papers/2409.00119.
