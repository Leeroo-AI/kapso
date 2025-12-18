# File: `src/peft/tuners/oft/layer.py`

**Category:** Layer Implementation

| Property | Value |
|----------|-------|
| Lines | 950 |
| Classes | `MultiplicativeDropoutLayer`, `OFTRotationModule`, `OFTLayer`, `Linear`, `Conv2d` |
| Functions | `dispatch_default` |
| Imports | __future__, config, peft, torch, typing, warnings |

## Understanding

**Status:** Fully explored

**Purpose:** Implements core OFT layer logic including orthogonal transformation modules, dropout, and layer-specific implementations for Linear and Conv2d.

**Mechanism:**

### 1. MultiplicativeDropoutLayer
Block-level dropout for OFT:
- Randomly replaces entire OFT blocks with identity matrices during training
- Probability `p` of dropping each block
- Shape: `(D, H, H)` where D is number of blocks, H is block size
- Skips dropout if only 1 block (block_share mode)

### 2. OFTRotationModule
Core orthogonal transformation module:

**Parameters**:
- `weight`: `(r, n_elements)` skew-symmetric matrix elements
- `n_elements = block_size * (block_size - 1) // 2` (upper triangle)

**Key Methods**:
- `_pytorch_skew_symmetric()`: Constructs skew-symmetric matrix from vector
- `_cayley_batch()`: Cayley transform to orthogonal matrix
  - With Cayley-Neumann: Efficient Neumann series approximation
  - Without: Direct matrix inversion (more accurate, slower)
- `_project_batch()`: COFT projection (constrains rotation freedom)
- `_block_diagonal()`: Assembles block-diagonal orthogonal matrix
- `_unfold()/_fold()`: Handle Conv2d spatial dimensions

**Cayley Transform**:
```python
R = (I - Q)(I + Q)^(-1)  # Without Cayley-Neumann
R = I + 2Q + 2Q^2 + ... + Q^(n-1)  # With Cayley-Neumann (n terms)
```

### 3. OFTLayer (Base Class)
Manages OFT adapters:

**Adapter Parameters**:
- `oft_R`: ModuleDict of OFTRotationModule per adapter
- `oft_block_size`: Block size per adapter
- `r`: Number of blocks per adapter
- `oft_dropout`: Multiplicative dropout layers

**Key Methods**:
- `update_layer()`: Creates OFT adapter with specified parameters
- `adjust_oft_parameters()`: Finds divisible block size/r if invalid
- `reset_oft_parameters()`: Initializes weights to zeros (identity)

### 4. Linear (OFT for Linear layers)
Applies orthogonal transformations to Linear layers:

**Forward**:
```python
for adapter in active_adapters:
    x = oft_R(x)  # Apply orthogonal transformation
result = base_layer(x)
```

**Merge/Unmerge**:
- Merge: `W_new = R @ W_old` (pre-multiply)
- Unmerge: `W_old = R^(-1) @ W_new` (requires matrix inverse)

### 5. Conv2d (OFT for Conv2d layers)
Applies OFT to convolutional layers:

**Handling**:
- Reshapes conv weights to 2D: `(out_ch, in_ch * kH * kW)`
- Applies OFT as for Linear
- Reshapes back to 4D: `(out_ch, in_ch, kH, kW)`
- Constraint: dilation > 1 not supported

**Significance:** This implements orthogonal finetuning's core mathematics. The Cayley parameterization ensures orthogonality by construction: skew-symmetric Q → orthogonal R via Cayley transform. The Neumann series approximation trades perfect orthogonality for computational efficiency (adjustable via num_cayley_neumann_terms). Block-diagonal structure reduces parameters while maintaining expressiveness. Multiplicative dropout provides regularization at the block level. The implementation carefully handles both standard and effective Conv2d modes.

## Key Technical Details

- **Skew-Symmetric Parameterization**: Only store upper triangle
- **Cayley Transform**: Maps skew-symmetric → orthogonal
- **Neumann Series**: O(n*d^3) vs O(d^3) for direct inversion
- **COFT Projection**: Constrains rotation magnitude for stability
- **Block-Diagonal**: Reduces O(d^2) to O((d/b)^2 * b) parameters
- **Multiplicative Dropout**: Block-level regularization

## Constraints

- **Divisibility**: in_features must be divisible by (r or oft_block_size)
- **Conv2d Dilation**: Only dilation=1 supported
- **No Nested Adapters**: OFT on OFT not recommended

## References

- Paper: https://huggingface.co/papers/2306.07280
- Cayley Transform: Standard technique for orthogonal matrices
- COFT: Constrained variant for stability
