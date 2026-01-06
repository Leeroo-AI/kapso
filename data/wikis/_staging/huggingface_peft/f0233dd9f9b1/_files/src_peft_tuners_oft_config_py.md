# File: `src/peft/tuners/oft/config.py`

**Category:** Configuration

| Property | Value |
|----------|-------|
| Lines | 213 |
| Classes | `OFTConfig` |
| Imports | __future__, dataclasses, packaging, peft, typing, warnings |

## Understanding

**Status:** Fully explored

**Purpose:** Defines `OFTConfig` class for Orthogonal Finetuning adapter configuration with comprehensive parameter control and validation.

**Mechanism:**

### Core Parameters:
- **r** (int, default=0): Number of OFT blocks per layer (mutually exclusive with oft_block_size)
- **oft_block_size** (int, default=32): Size of each OFT block (mutually exclusive with r)
  - Constraint: `r * oft_block_size == in_features`
- **module_dropout** (float, default=0.0): Multiplicative dropout - randomly sets blocks to identity during training
- **coft** (bool, default=False): Use Constrained OFT variant
- **eps** (float, default=6e-5): Control strength for COFT (freedom of rotation)
- **block_share** (bool, default=False): Share OFT parameters between blocks
- **use_cayley_neumann** (bool, default=True): Use Cayley-Neumann formulation for efficiency
- **num_cayley_neumann_terms** (int, default=5): Number of terms in approximation

### Module Selection:
- **target_modules**: Modules to adapt (regex, list, or 'all-linear')
- **exclude_modules**: Modules to exclude
- **fan_in_fan_out** (bool): Whether layer stores weights as (fan_in, fan_out)
- **bias** (Literal["none", "all", "oft_only"]): Bias training strategy
- **init_weights** (bool, default=True): Initialize to identity (zeros)

### Validation Logic:
1. Validates that either r or oft_block_size is specified (not both, not neither)
2. Checks for valid layers_pattern/layers_to_transform combination
3. Version checking for use_cayley_neumann (PEFT 0.18.0+ required)
4. Version checking for oft_block_size parameter (PEFT 0.14.0+ required)

**Significance:** OFT uses orthogonal transformations to adapt models while preserving activation norms. The configuration provides fine control over the orthogonal structure:
- **Block Structure**: Divides weight matrices into blocks, each with orthogonal transformation
- **Cayley-Neumann**: Efficient parameterization using Cayley transform with Neumann series approximation
- **COFT**: Constrained variant that limits rotation freedom for stability
- **Block Sharing**: Reduces parameters by using same transformation across blocks

The orthogonal constraint ensures stable training and has theoretical benefits for generalization.

## Key Features

- **Mutual Exclusivity**: Either r or oft_block_size, never both
- **Multiplicative Dropout**: Block-level dropout for regularization
- **Cayley Parameterization**: Efficient orthogonal matrix representation
- **Version Validation**: Ensures compatibility with saved checkpoints
- **Constrained Variant**: COFT for controlled adaptation

## References

- Paper: https://huggingface.co/papers/2306.07280
- Method: Orthogonal Finetuning
- Cayley-Neumann: PEFT 0.18.0+ uses improved numerical stability
