# VBLoRA Layer Implementation

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/vblora/layer.py`
- **Lines**: 251
- **Purpose**: VBLoRA (Vector Bank LoRA) adapter layers using top-k vector selection

## Overview

This module implements VBLoRA adapter layers, which use a shared vector bank with top-k selection to construct low-rank adaptation matrices. Instead of storing full matrices, VBLoRA learns selection logits that choose and combine vectors from a shared bank, dramatically reducing storage requirements while maintaining adaptation quality.

## Key Components

### VBLoRALayer (Base Class)

**Purpose**: Base layer for VBLoRA adapters managing vector bank and selection logits

**Attributes**:
- `adapter_layer_names`: `("vblora_logits_A", "vblora_logits_B", "vblora_vector_bank")` - Trainable parameters
- `r`: Dict mapping adapter names to their ranks
- `topk`: Dict mapping adapter names to top-k values
- `vblora_logits_A`: ParameterDict storing input selection logits (shape: r × in_tiles × num_vectors)
- `vblora_logits_B`: ParameterDict storing output selection logits (shape: out_tiles × r × num_vectors)
- `vblora_vector_bank`: ParameterDict storing shared vector bank (shape: num_vectors × vector_length)
- `vblora_dropout`: ModuleDict for dropout layers per adapter

**Key Methods**:

1. **`__init__(base_layer, **kwargs)`**
   - Initializes base layer wrapper
   - Extracts in_features and out_features from base layer (Linear or Conv1D)
   - Sets up adapter parameter dictionaries

2. **`update_layer(adapter_name, vblora_vector_bank, r, topk, num_vectors, vector_length, vblora_dropout=0.0, init_logits_std=0.01, inference_mode=False)`**
   - Adds or updates a VBLoRA adapter
   - **Parameters**:
     - `r`: Rank of the adapter (must be positive)
     - `topk`: Number of vectors to select per position (typically 2)
     - `num_vectors`: Size of vector bank (e.g., 60, 256)
     - `vector_length`: Length of each vector (must divide in/out features)
     - `vblora_dropout`: Dropout probability
     - `init_logits_std`: Standard deviation for logit initialization
   - **Validation**:
     - in_features must be divisible by vector_length
     - out_features must be divisible by vector_length
   - **Creates Trainable Parameters**:
     - `vblora_logits_A[adapter_name]`: Shape (r, in_features // vector_length, num_vectors)
     - `vblora_logits_B[adapter_name]`: Shape (out_features // vector_length, r, num_vectors)
   - **Vector Bank**: Reference to shared ParameterDict

3. **`reset_vblora_logits(adapter_name, init_logits_std)`**
   - Initializes logits with normal distribution
   - Mean: 0, Std: init_logits_std
   - Applied to both logits_A and logits_B

### Linear (Implementation Class)

**Purpose**: VBLoRA implementation for Linear layers (inherits from nn.Linear and VBLoRALayer)

**Constructor Parameters**:
- `base_layer`: Original Linear layer to wrap
- `vblora_vector_bank`: Shared vector bank (ParameterDict)
- `adapter_name`: Name of the adapter
- `r`: Rank dimension
- `num_vectors`: Number of vectors in bank
- `vector_length`: Length of each vector
- `topk`: Number of vectors to select (default: 2)
- `vblora_dropout`: Dropout probability
- `init_logits_std`: Logit initialization std dev
- `fan_in_fan_out`: Whether layer stores weights as (fan_in, fan_out)
- `is_target_conv_1d_layer`: Flag for Conv1D layers

**Key Methods**:

1. **`merge(safe_merge=False, adapter_names=None)`**
   - Merges active adapter weights into base weights
   - Constructs full delta weight from vector bank and logits
   - **Parameters**:
     - `safe_merge`: If True, checks for NaNs before committing
     - `adapter_names`: List of adapters to merge (None = all active)
   - Updates `merged_adapters` list

2. **`unmerge()`**
   - Removes merged adapter weights from base weights
   - Reverses the merge operation

3. **`_get_low_rank_matrix(logits, vblora_vector_bank, topk)`**
   - Core vector selection and combination logic
   - **Algorithm**:
     ```python
     1. Select top-k indices: top_k_logits, indices = logits.topk(topk, dim=-1)
     2. Compute softmax weights: topk_weights = F.softmax(top_k_logits, dim=-1)
     3. Gather vectors: vectors = vblora_vector_bank[indices]
     4. Weighted combination: result = (topk_weights.unsqueeze(-1) * vectors).sum(-2)
     ```
   - Returns: Combined vector for each position

4. **`_get_lora_matrices(adapter, cast_to_fp32=False)`**
   - Constructs full A and B matrices from vector bank
   - **Algorithm**:
     ```python
     1. Get logits and vector bank
     2. Validate logits (check for infinity from save_only_topk_weights)
     3. Construct A matrix:
        - Apply _get_low_rank_matrix to logits_A
        - Shape: (rank, in_tile, vector_length) → (rank, in_tile × vector_length)
     4. Construct B matrix:
        - Apply _get_low_rank_matrix to logits_B
        - Shape: (out_tile, rank, vector_length) → (out_tile × vector_length, rank)
        - Includes transpose: (out_tile, rank, vector_length) → (out_tile, vector_length, rank)
     5. Return (A, B)
     ```
   - **Training Check**: Raises error if infinity values found (indicates resumed from save_only_topk_weights)
   - **CPU Float16 Handling**: Casts to float32 if needed

5. **`get_delta_weight(adapter)`**
   - Computes delta weight for an adapter
   - **Formula**: `delta = B @ A`
   - Handles fan_in_fan_out transpose
   - CPU float16 optimization

6. **`forward(x, *args, **kwargs)`**
   - Forward pass with VBLoRA adaptation
   - **Logic Flow**:
     ```python
     if disable_adapters:
         if merged: unmerge()
         return base_layer(x)
     elif merged:
         return base_layer(x)  # Already includes adapter
     else:
         result = base_layer(x)
         for each active adapter:
             A, B = _get_lora_matrices(adapter)
             x_adapted = x.to(vector_bank.dtype)
             result += F.linear(F.linear(dropout(x_adapted), A), B)
         return result
     ```
   - Preserves input dtype

## Mathematical Formulation

VBLoRA constructs low-rank matrices from vector bank:

```
For each position i,j in matrix A or B:
1. logits[i,j] → top-k vector indices
2. weights = softmax(top-k logits)
3. matrix_element[i,j] = Σ(weights[k] * vector_bank[indices[k]])

Final adaptation:
output = base_layer(x) + F.linear(F.linear(x, A), B)
where A and B are constructed from vector bank
```

### Tiling Structure

**Input Dimension Tiling**:
- in_features = in_tiles × vector_length
- logits_A: (rank, in_tiles, num_vectors)
- Each tile selects top-k vectors independently

**Output Dimension Tiling**:
- out_features = out_tiles × vector_length
- logits_B: (out_tiles, rank, num_vectors)
- Each tile selects top-k vectors independently

**Example** (in_features=768, out_features=768, vector_length=256, rank=4, topk=2):
```
in_tiles = 768 / 256 = 3
out_tiles = 768 / 256 = 3

logits_A: (4, 3, num_vectors) → A: (4, 768)
logits_B: (3, 4, num_vectors) → B: (768, 4)

Final: B @ A = (768, 4) @ (4, 768) = (768, 768)
```

## Top-K Selection Mechanism

### Selection Process
```python
# For each logit position
logits: [num_vectors]  # e.g., [0.5, -0.2, 0.8, -0.1, 0.3, ...]

# 1. Select top-k
topk_logits: [0.8, 0.5]  # if topk=2
indices: [2, 0]

# 2. Softmax for weights
weights = softmax([0.8, 0.5]) = [0.57, 0.43]

# 3. Weighted combination
vectors = vector_bank[[2, 0]]  # Shape: (2, vector_length)
result = 0.57 * vectors[0] + 0.43 * vectors[1]
```

### Benefits
- **Sparse Selection**: Only topk vectors contribute per position
- **Differentiable**: Softmax allows gradient flow
- **Flexible**: Different positions can select different vectors

## Storage Optimization

### save_only_topk_weights

When `config.save_only_topk_weights = True`:

**Standard Storage**:
- Logits: All num_vectors values per position
- Size: O(rank × tiles × num_vectors)

**Optimized Storage**:
- Store only top-k indices and weights
- Indices: Integer type (uint8, uint16, or uint32 depending on num_vectors)
- Weights: topk-1 values (softmax last can be inferred)
- Size reduction: ~90% for topk=2, num_vectors=256

**Limitation**: Models saved this way cannot resume training (only inference/merge)

### Inference Detection
```python
if self.training and vblora_logits_A[0, 0].isinf().any():
    raise RuntimeError(
        "Found infinity values in VB-LoRA logits. "
        "Ensure training was not resumed from a `save_only_topk_weights` model."
    )
```

## Parameter Efficiency

For a layer with dimensions d_in × d_out:

**Standard LoRA**:
- Parameters: r(d_in + d_out)

**VBLoRA (per adapter)**:
- Logits_A: r × (d_in // vector_length) × num_vectors
- Logits_B: (d_out // vector_length) × r × num_vectors
- Vector Bank: num_vectors × vector_length (shared across layers)

**Example** (d_in=d_out=4096, r=4, vector_length=256, num_vectors=60):
- LoRA: 4 × (4096 + 4096) = 32,768 params/layer
- VBLoRA Logits: 4 × 16 × 60 + 16 × 4 × 60 = 7,680 params/layer
- VBLoRA Bank: 60 × 256 = 15,360 params (shared)

**With save_only_topk_weights (topk=2)**:
- Stored: ~1,920 params/layer (4x reduction)

## Design Patterns

### 1. Vector Bank Sharing
Single vector bank shared across all adapted layers:
```python
self.vblora_vector_bank[adapter_name]  # Same for all layers
```

### 2. Lazy Matrix Construction
Matrices constructed on-demand from logits:
```python
A, B = self._get_lora_matrices(adapter)  # Built during forward pass
```

### 3. Top-K Differentiable Selection
Combines discrete selection (top-k) with differentiable combination (softmax)

### 4. Tiled Architecture
Divides large dimensions into manageable tiles for vector selection

## Integration Points

- Imports `BaseTunerLayer` from `peft.tuners.tuners_utils`
- Uses `check_adapters_to_merge` for merge validation
- Supports Conv1D layers from transformers.pytorch_utils
- PyTorch F.softmax and tensor operations

## Usage Example

```python
# Initialize VBLoRA layer
vblora_layer = Linear(
    base_layer=nn.Linear(768, 768),
    vblora_vector_bank=vector_bank,
    adapter_name="default",
    r=4,
    num_vectors=60,
    vector_length=256,
    topk=2,
    vblora_dropout=0.0,
    init_logits_std=0.1
)

# Forward pass
output = vblora_layer(input_tensor)

# Merge for inference
vblora_layer.merge(safe_merge=True)
```

## Implementation Notes

1. **Initialization**: Logits initialized with normal(0, init_logits_std)
2. **Vector Length Constraint**: Must divide both in_features and out_features
3. **Top-K Value**: Paper recommends topk=2 for best performance/efficiency
4. **Dropout**: Applied before first linear transformation
5. **Device Management**: Vector bank moved to logits device during forward pass
6. **Training Resumption**: Cannot resume from save_only_topk_weights checkpoints

## References

- **Paper**: https://huggingface.co/papers/2405.15179
- **Key Innovation**: Vector bank with learned top-k selection for parameter-efficient adaptation
- **Recommended Settings**: topk=2, num_vectors=60-256, vector_length=256
