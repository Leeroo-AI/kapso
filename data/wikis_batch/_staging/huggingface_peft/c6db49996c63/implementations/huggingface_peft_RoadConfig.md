# RoadConfig (RoAd Configuration)

## Implementation Overview

**File:** `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/road/config.py`
**Lines of Code:** 126
**Language:** Python

RoadConfig defines parameters for RoAd (Rotation Adaptation) adapters that apply learned 2D rotations to hidden representations using only rotation angles and scale factors.

## Core Configuration

```python
@dataclass
class RoadConfig(PeftConfig):
    """Configuration for RoadModel

    Paper: https://huggingface.co/papers/2409.00119

    Args:
        variant: 'road_1', 'road_2', or 'road_4'
        group_size: Grouping size for 2D rotations (default: 64)
        init_weights: Whether to initialize to identity (default: True)
        target_modules: Modules to adapt
        modules_to_save: Additional trainable modules
    """

    variant: Union[str, RoadVariant] = field(
        default="road_1",
        metadata={"help": "Variant: road_1 (minimal), road_2 (moderate), road_4 (maximum)"}
    )
    group_size: int = field(
        default=64,
        metadata={"help": "Group size for 2D rotation pairing. Must divide hidden size."}
    )
    init_weights: bool = field(
        default=True,
        metadata={"help": "Initialize to identity transformation (θ=0, α=1)"}
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "Module names or regex to adapt"}
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Additional trainable modules"}
    )
```

## Key Parameters

### Variant Selection

```python
RoadVariant = Literal["road_1", "road_2", "road_4"]

variant: Union[str, RoadVariant] = field(default="road_1")
```

**Variants:**

**road_1:** Minimal Parameters
- Params: `out_features / 2`
- Same θ and α for paired elements
- 32x fewer params than LoRA(r=8)
- Best for: Maximum efficiency

**road_2:** Moderate Parameters
- Params: `out_features`
- Unique θ and α per element
- 16x fewer params than LoRA(r=8)
- Best for: Balance efficiency/capacity

**road_4:** Maximum Parameters
- Params: `out_features * 2`
- Separate θ and α for cos/sin
- 8x fewer params than LoRA(r=8)
- Best for: Maximum adaptation capacity

### Group Size

```python
group_size: int = field(
    default=64,
    metadata={
        "help": "Elements grouped together for 2D rotation pairing. "
                "Must divide hidden size. Affects inference speed (larger = faster)."
    }
)
```

**Requirements:**
- Must divide `out_features`
- Must be even (divisible by 2)
- Recommended: ≥32 or 64 for speed

**Impact:**
- No effect on model performance (elements unordered)
- Affects inference speed in VLLM and similar
- Larger = better SIMD vectorization

**Examples:**
```python
# For 768-dim layers:
group_size=64  # OK: 768 % 64 == 0
group_size=32  # OK: 768 % 32 == 0
group_size=100 # Error: 768 % 100 != 0

# For tensor parallelism:
# If 768 split across 4 GPUs = 192 per GPU
group_size=64  # OK: 192 % 64 == 0
group_size=128 # Error: 192 % 128 != 0
```

### Initialization

```python
init_weights: bool = field(
    default=True,
    metadata={"help": "Initialize to identity (θ=0, α=1) vs random"}
)
```

**Options:**
- `True`: θ=0, α=1 (identity transformation, no change initially)
- `False`: θ~N(0, 0.5), α~N(1, 0.5) (random initialization)

**Recommendation:** Use `True` (default) for stable training

## Validation

**Method:** `__post_init__()`

```python
def __post_init__(self):
    super().__post_init__()
    self.peft_type = PeftType.ROAD
    self.target_modules = (
        set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
    )

    # Validate variant
    if self.variant not in ["road_1", "road_2", "road_4"]:
        raise ValueError(f"Invalid variant {self.variant}. Choose from road_1, road_2, road_4")

    # Validate group_size
    if self.group_size <= 0 or self.group_size % 2 != 0:
        raise ValueError(f"group_size must be positive and divisible by 2, got {self.group_size}")
```

## Configuration Examples

### Example 1: Maximum Efficiency

```python
config = RoadConfig(
    variant="road_1",        # Minimal params
    group_size=64,
    target_modules=['q_proj', 'v_proj'],
    init_weights=True
)
# For 768-dim: 384 params per layer
```

### Example 2: Balanced

```python
config = RoadConfig(
    variant="road_2",        # Moderate params
    group_size=64,
    target_modules='all-linear',
    init_weights=True
)
# For 768-dim: 768 params per layer
```

### Example 3: Maximum Capacity

```python
config = RoadConfig(
    variant="road_4",        # Maximum params
    group_size=32,           # Smaller groups
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    init_weights=True
)
# For 768-dim: 1,536 params per layer
```

### Example 4: Classification Task

```python
config = RoadConfig(
    variant="road_2",
    group_size=64,
    target_modules='all-linear',
    modules_to_save=['classifier'],  # Train classifier head
    init_weights=True
)
```

## Parameter Efficiency

For 768-dimensional layers:

| Variant | Params | vs LoRA(r=8) | vs LoRA(r=16) |
|---------|--------|--------------|---------------|
| road_1  | 384    | 32x fewer    | 64x fewer     |
| road_2  | 768    | 16x fewer    | 32x fewer     |
| road_4  | 1,536  | 8x fewer     | 16x fewer     |

## Common Pitfalls

### Wrong Group Size

```python
# Wrong: doesn't divide 768
config = RoadConfig(group_size=100)  # Error!

# Correct:
config = RoadConfig(group_size=64)  # 768 % 64 == 0
```

### Odd Group Size

```python
# Wrong: not divisible by 2
config = RoadConfig(group_size=65)  # Error!

# Correct:
config = RoadConfig(group_size=64)
```

### Wrong Variant String

```python
# Wrong: typo or invalid
config = RoadConfig(variant="road3")  # Error!

# Correct:
config = RoadConfig(variant="road_1")
```

## References

- **Paper**: https://huggingface.co/papers/2409.00119
- **Type**: `PeftType.ROAD`
- **Key Innovation**: 2D rotation-based adaptation
- **Efficiency**: 8-32x fewer parameters than LoRA
