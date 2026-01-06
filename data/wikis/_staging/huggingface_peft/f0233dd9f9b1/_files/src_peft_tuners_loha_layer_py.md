# File: `src/peft/tuners/loha/layer.py`

**Category:** Layer Implementation

| Property | Value |
|----------|-------|
| Lines | 444 |
| Classes | `LoHaLayer`, `Linear`, `Conv2d`, `Conv1d`, `HadaWeight`, `HadaWeightCP` |
| Functions | `make_weight`, `make_weight_cp` |
| Imports | math, peft, torch, typing |

## Understanding

**Status:** Fully explored

**Purpose:** Implements the core LoHa (Low-Rank Hadamard Product) layer logic and custom autograd functions for efficient gradient computation.

**Mechanism:**

### 1. LoHaLayer (Base Class)
The foundational layer that manages LoHa adapter parameters:

**Adapter Parameters** (6 weight matrices per adapter):
- `hada_w1_a`, `hada_w1_b`: First low-rank decomposition (A @ B)
- `hada_w2_a`, `hada_w2_b`: Second low-rank decomposition (C @ D)
- `hada_t1`, `hada_t2`: Tucker decomposition tensors for Conv layers (optional)

**Parameter Creation** (`create_adapter_parameters`):
- **Conv2d/Conv1d**: Creates 4D/3D tensors with Tucker decomposition support
- **Linear**: Creates 2D matrices for standard low-rank factorization

**Weight Initialization** (`reset_adapter_parameters`):
- Uses He initialization (Kaiming uniform) for stability
- Initializes `hada_w2_b` to zeros so adapter starts as identity
- Alternative: `reset_adapter_parameters_random` initializes all weights non-zero

**Delta Weight Computation** (`get_delta_weight`):
- Computes: Δ = (W1a @ W1b) ⊙ (W2a @ W2b) × scaling
- For Conv: Uses CP decomposition with Tucker tensors
- Applies rank dropout during training (drops rows randomly)

### 2. Custom Autograd Functions

**HadaWeight** - Standard Hadamard product with custom backward:
```python
forward: (w1a @ w1b) * (w2a @ w2b) * scale
backward: Efficient gradient computation avoiding redundant operations
```

**HadaWeightCP** - Tucker/CP decomposition for Conv layers:
```python
forward: einsum-based reconstruction with Tucker tensors
backward: Custom gradient computation for tensor decomposition
```

### 3. Layer Implementations

**Linear** (`LoHaLayer` for Linear layers):
- Applies LoHa to `nn.Linear` layers
- Uses `F.linear` for delta activations
- Supports merging/unmerging adapters

**Conv2d** (`LoHaLayer` for Conv2d layers):
- Handles 2D convolutions with optional effective decomposition
- Optimizes 1x1 convolutions by treating them like Linear layers
- Uses `F.conv2d` with base layer's stride, padding, dilation, groups

**Conv1d** (`LoHaLayer` for Conv1d layers):
- Handles 1D convolutions (e.g., for sequence modeling)
- Similar optimizations for kernel_size=1
- Uses `F.conv1d` for efficient computation

### 4. Forward Pass Logic

All layer implementations follow this pattern:
1. Check if adapters are disabled → run base layer only
2. Check if merged → run base layer (adapters already in weights)
3. Otherwise: Run base layer + apply each active adapter's delta
4. Apply module dropout (randomly skip adapters during training)

**Significance:** This file implements the mathematical core of LoHa adapters. The Hadamard product (element-wise multiplication) of two low-rank decompositions provides more expressiveness than single low-rank factorizations (like LoRA) while maintaining parameter efficiency. The custom autograd functions enable efficient backpropagation through the decomposition. The Tucker/CP decomposition support for convolutional layers (from the FedPara paper) allows effective adaptation of vision models with minimal parameters.

## Key Technical Details

- **Hadamard Product**: ⊙ denotes element-wise multiplication
- **Rank Dropout**: Randomly drops output channels during training for regularization
- **Module Dropout**: Randomly disables entire adapters during training
- **Effective Conv2d**: Uses Tucker decomposition for kernel_size > 1
- **Memory Efficiency**: Creates parameters on demand, moves to correct device
- **Gradient Optimization**: Custom backward passes avoid redundant computation

## References

- Based on LyCORIS implementation: https://github.com/KohakuBlueleaf/LyCORIS
- Paper: https://huggingface.co/papers/2108.06098
- FedPara paper: Proposition 3 for effective Conv2d decomposition
