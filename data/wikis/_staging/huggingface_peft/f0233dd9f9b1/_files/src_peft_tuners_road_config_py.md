# File: `src/peft/tuners/road/config.py`

**Category:** configuration

| Property | Value |
|----------|-------|
| Lines | 127 |
| Classes | `RoadConfig` |
| Types | `RoadVariant` |
| Imports | __future__, dataclasses, peft, typing |

## Understanding

**Status:** ✅ Fully explored

**Purpose:** Configuration class for RoAd (Rotation Adapter), defining hyperparameters for rotation-based parameter-efficient fine-tuning.

**Mechanism:**
- **RoadVariant type**: Literal type defining three variants: "road_1", "road_2", "road_4"

- **RoadConfig dataclass** extends `PeftConfig` with RoAd-specific parameters:
  - `variant` (RoadVariant, default="road_1"): Determines parameter count:
    - **road_1**: Same scale/angle for all pairs - stores output_size parameters (lowest)
    - **road_2**: Same scale/angle per element - 2x parameters vs road_1
    - **road_4**: Different scales/angles per element - 4x parameters vs road_1
  - `group_size` (int, default=64): Elements grouped for 2D rotation
    - Element i pairs with element i+group_size/2 within each group
    - Affects inference speed (larger = faster, minimum 32-64 recommended)
    - Hidden size must be divisible by group_size
  - `init_weights` (bool, default=True): Initialize with identity transformation (zeros for theta, ones for alpha)
  - `target_modules`: Module names to apply RoAd (supports 'all-linear')
  - `modules_to_save`: Additional modules to train and save

- **Validation in __post_init__:**
  - Sets `peft_type = PeftType.ROAD`
  - Validates variant is one of: road_1, road_2, road_4
  - Ensures group_size is positive and even (divisible by 2)
  - Converts target_modules list to set

**Significance:** Core configuration enabling RoAd's rotation-based adaptation. RoAd transforms hidden states by rotating 2D vectors formed from pairs of elements. The key insight is that 2D rotations require only 2 parameters (angle θ and scale α) per pair, making it extremely parameter-efficient. The variant controls granularity: road_1 shares parameters across pairs, road_2 per-element sharing, road_4 independent per-element. The group_size parameter balances flexibility with computational efficiency for inference deployment.
