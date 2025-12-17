# Implementation: boft/config.py

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/boft/config.py`
- **Size**: 160 lines
- **Description**: Butterfly Orthogonal Finetuning (BOFT) configuration

## Overview

BOFT is a parameter-efficient method that uses butterfly factorization to parameterize orthogonal transformations, providing better rotation expressivity than standard OFT while maintaining parameter efficiency.

**Reference**: [BOFT Paper](https://huggingface.co/papers/2311.06243)

## Core Configuration

### BOFTConfig

```python
@dataclass
class BOFTConfig(PeftConfig):
    boft_block_size: int = 4                # Block size (mutually exclusive with boft_block_num)
    boft_block_num: int = 0                 # Number of blocks (set one, not both)
    boft_n_butterfly_factor: int = 1        # Butterfly decomposition depth
    boft_dropout: float = 0.0               # Multiplicative dropout

    target_modules: Optional[Union[list[str], str]] = None
    exclude_modules: Optional[Union[list[str], str]] = None

    fan_in_fan_out: bool = False
    bias: str = "none"                      # "none", "all", or "boft_only"
    modules_to_save: Optional[list[str]] = None
    init_weights: bool = True

    layers_to_transform: Optional[Union[list[int], int]] = None
    layers_pattern: Optional[Union[list[str], str]] = None
```

### Key Constraints

**Mutual Exclusivity**:
```python
if not (self.boft_block_size != 0) ^ (self.boft_block_num != 0):
    raise ValueError("Specify either boft_block_size OR boft_block_num, not both")
```

**Constraint**: `boft_block_size × boft_block_num = layer_dimension`

### Butterfly Factorization

**n_butterfly_factor=1**: Standard OFT (single block rotation)
**n_butterfly_factor=2**: Effective block size doubles, num blocks halves
**Higher factors**: More expressive, fewer blocks

## Cross-References

- **Model**: `boft/model.py`
- **Paper**: [BOFT: Butterfly Finetuning](https://huggingface.co/papers/2311.06243)
