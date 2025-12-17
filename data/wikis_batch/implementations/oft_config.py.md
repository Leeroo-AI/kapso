# Implementation: oft/config.py

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/oft/config.py`
- **Size**: 213 lines
- **Description**: Orthogonal Finetuning (OFT) configuration

## Overview

OFT applies orthogonal transformations to weight matrices, preserving norms and angles while adapting the model. This configuration supports both standard OFT and constrained OFT (COFT) variants.

**Reference**: [OFT Paper](https://huggingface.co/papers/2306.07280)

## Core Configuration

### OFTConfig

```python
@dataclass
class OFTConfig(PeftConfig):
    r: int = 0                              # Number of blocks (mutually exclusive with oft_block_size)
    oft_block_size: int = 32                # Block size
    module_dropout: float = 0.0             # Multiplicative dropout

    # Variant selection
    coft: bool = False                      # Constrained OFT
    eps: float = 6e-5                       # COFT freedom of rotation
    block_share: bool = False               # Share parameters across blocks

    # Parameterization
    use_cayley_neumann: bool = True         # Use Cayley-Neumann formulation
    num_cayley_neumann_terms: int = 5       # Approximation accuracy

    target_modules: Optional[Union[list[str], str]] = None
    exclude_modules: Optional[Union[list[str], str]] = None

    fan_in_fan_out: bool = False
    bias: Literal["none", "all", "oft_only"] = "none"
    init_weights: bool = True

    layers_to_transform: Optional[Union[list[int], int]] = None
    layers_pattern: Optional[Union[list[str], str]] = None
    modules_to_save: Optional[list[str]] = None
```

### Key Constraints

**Mutual Exclusivity**:
```python
if not (self.r != 0) ^ (self.oft_block_size != 0):
    raise ValueError("Specify either r OR oft_block_size, not both")
```

**Constraint**: `r × oft_block_size = layer_dimension`

### Parameterization Options

**Cayley-Neumann Formulation**:
- **Enabled**: Faster, approximate orthogonality
- **Disabled**: Slower, exact orthogonality via matrix exponential
- **Terms**: More terms = better approximation

**Version Compatibility**:
```python
@classmethod
def check_kwargs(cls, **kwargs):
    if kwargs.get("use_cayley_neumann", False):
        peft_version = kwargs.get("peft_version", "0.0.0")
        if packaging.version.parse(peft_version) < packaging.version.Version("0.18.0"):
            warnings.warn("Cayley-Neumann parameterization changed in 0.18.0. Retrain or downgrade.")
```

## COFT Variant

**Constrained OFT**:
- Limits rotation freedom via `eps` parameter
- More stable training
- Less expressive than full OFT

**Usage**:
```python
config = OFTConfig(
    r=8,
    coft=True,
    eps=6e-5  # Controls rotation freedom
)
```

## Cross-References

- **Model**: `oft/model.py`
- **Paper**: [Controlling Text-to-Image Diffusion via OFT](https://huggingface.co/papers/2306.07280)
