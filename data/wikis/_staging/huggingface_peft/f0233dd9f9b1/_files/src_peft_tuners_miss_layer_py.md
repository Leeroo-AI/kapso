# File: `src/peft/tuners/miss/layer.py`

**Category:** Layer Implementation

| Property | Value |
|----------|-------|
| Lines | 393 |
| Classes | `MissLayer`, `MissLinear` |
| Imports | __future__, math, peft, torch, typing, warnings |

## Understanding

**Status:** Fully explored

**Purpose:** Implements core MiSS layer logic using Householder reflections for parameter-efficient adaptation of neural network layers.

**Mechanism:**

### 1. MissLayer (Base Class)

Manages MiSS adapter parameters and state:

**Adapter Parameters**:
- `miss_block`: Main parameter tensor with shape depending on init mode:
  - Balance mode: `(r, out_features)` - standard shape
  - Bat mode: `(out_features // r, r, r)` - block structure for nonlinear updates
  - Mini mode: `(r, mini_r)` - compact representation
- `miss_dropout`: Dropout layers per adapter
- `miss_r`: Rank values per adapter
- `miss_mini_r`: Mini rank values per adapter

**Initialization Modes** (`update_layer`):
- **Balance** (`reset_miss_parameters`): Zeros initialization, general purpose
- **Bat** (`reset_bat_parameters`): Block structure, requires divisibility constraints
- **Mini** (`reset_mini_parameters`): Compact, requires out_features % mini_r == 0
- **Random** (`reset_miss_parameters_random`): Kaiming uniform init

### 2. MissLinear (Linear Layer Implementation)

Implements MiSS adaptation for `nn.Linear` layers:

**Forward Pass Logic**:
- **Bat mode**: Modifies weight matrix directly using block operations
  ```python
  delta = (orig_weight @ miss_block) + miss_block
  result = F.linear(x, delta, bias)
  ```
- **Balance/Mini modes**: Adds delta to activations
  ```python
  result = base_layer(x)
  # Sum over blocks: x.reshape(..., r) @ miss_block
  result += sum(dropout(x).reshape(..., r), dim=-2) @ miss_block
  ```

**Weight Delta Computation**:
- **Bat mode** (`get_delta_weight`): Block-wise transformations with inverse for unmerging
- **Balance/Mini** (`get_delta_weight_miss`): Adds miss_block to reshaped weights
- Handles non-divisible dimensions by splitting into blocks + remainder

**Merge/Unmerge**:
- **Merge**: Folds adapter weights into base layer for inference speedup
- **Unmerge**: Extracts adapter weights back out (with potential numerical errors)
- **Safe merge**: Checks for NaN/Inf before committing changes
- Supports bat, mini, and balance modes with different merge strategies

**Significance:** MissLinear implements Householder reflection-based adaptation, which differs fundamentally from LoRA:

1. **Householder Reflections**: Uses orthogonal transformations instead of rank-limited matrices
2. **Three Modes**:
   - **Balance**: Standard approach, most flexible
   - **Bat**: Block-wise nonlinear updates, enables more complex adaptations
   - **Mini**: Minimal parameters via repeated blocks
3. **Block-wise Processing**: Divides features into r-sized blocks for efficient computation
4. **Padding Handling**: Automatically handles non-divisible feature dimensions

The method achieves competitive performance with potentially fewer parameters than LoRA by leveraging orthogonal structure and block-wise operations.

## Key Technical Details

- **Block-wise Operations**: Reshapes inputs into (batch, ..., n_blocks, r) for efficient processing
- **Automatic Padding**: Handles non-divisible dimensions transparently
- **Three Forward Paths**: Different computation for bat vs balance/mini modes
- **Merge Support**: All three modes support weight merging with appropriate inverse operations
- **Dropout Integration**: Per-adapter dropout for regularization
- **Scaling Warnings**: Scale operations not supported, automatically set to 1

## Constraints

- **Bat Mode**: Requires in_features % r == 0 and out_features % r == 0
- **Mini Mode**: Requires out_features % mini_r == 0
- **Linear Only**: Currently only supports nn.Linear layers
- **No Scaling**: Scale/unscale operations emit warnings and do nothing

## References

- Paper: https://huggingface.co/papers/2409.15371
- Method: Householder reflection adaptation
- Inspired by matrix-free orthogonal transformations
