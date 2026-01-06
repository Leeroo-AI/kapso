# src/peft/tuners/lokr/layer.py

## Overview
Implementation of LoKr (Low-Rank Kronecker Product) layers that apply Kronecker product factorization to enable parameter-efficient weight updates. This file contains the base LoKrLayer class and specialized implementations for Linear, Conv2d, and Conv1d layers, along with utility functions for factorization and Kronecker products.

## Class: LoKrLayer

Base class for all LoKr adapter layers. Inherits from both `nn.Module` and `LycorisLayer`.

### Adapter Layer Names
The class defines which parameter dictionaries contain adapter weights:
- `lokr_w1`: Full left Kronecker matrix
- `lokr_w1_a`, `lokr_w1_b`: Low-rank factorization of left matrix
- `lokr_w2`: Full right Kronecker matrix
- `lokr_w2_a`, `lokr_w2_b`: Low-rank factorization of right matrix
- `lokr_t2`: Effective convolution tensor (for Conv2d/Conv1d with use_effective_conv2d)

### Initialization
Creates empty ParameterDict instances for all possible LoKr components.

### Methods

#### create_adapter_parameters()
Creates the actual parameter tensors for a specific adapter.

**Parameters:**
- **adapter_name**: Name of the adapter
- **r**: Rank for low-rank factorization
- **shape**: Tuple describing weight dimensions
- **use_w1**: Whether to use full w1 matrix (vs. factorized)
- **use_w2**: Whether to use full w2 matrix (vs. factorized)
- **use_effective_conv2d**: Whether to use effective convolution decomposition

**Weight Creation Logic:**

**Left Kronecker Matrix (w1):**
- If `use_w1=True`: Create full matrix of shape `(shape[0][0], shape[1][0])`
- If `use_w1=False`: Create factorized matrices:
  - `w1_a`: shape `(shape[0][0], r)`
  - `w1_b`: shape `(r, shape[1][0])`

**Right Kronecker Matrix (w2) - Linear:**
- If `use_w2=True`: Create full matrix of shape `(shape[0][1], shape[1][1])`
- If `use_w2=False`: Create factorized matrices:
  - `w2_a`: shape `(shape[0][1], r)`
  - `w2_b`: shape `(r, shape[1][1])`

**Right Kronecker Matrix (w2) - Conv2d:**
- If `use_w2=True`: Full matrix `(shape[0][1], shape[1][1], kernel_h, kernel_w)`
- If `use_effective_conv2d=True`: CP decomposition with three components:
  - `t2`: Core tensor `(r, r, kernel_h, kernel_w)`
  - `w2_a`: Mode-1 factor `(r, shape[0][1])`
  - `w2_b`: Mode-2 factor `(r, shape[1][1])`
- If neither: Flattened factorization:
  - `w2_a`: shape `(shape[0][1], r)`
  - `w2_b`: shape `(r, shape[1][1] * kernel_h * kernel_w)`

**Right Kronecker Matrix (w2) - Conv1d:**
Similar to Conv2d but with single kernel dimension.

#### reset_adapter_parameters()
Default initialization strategy (recommended):
- **w1 or w1_a**: Zeros (no initial contribution)
- **w1_b**: Kaiming uniform
- **w2 or w2_a**: Kaiming uniform
- **w2_b**: Kaiming uniform
- **t2**: Kaiming uniform

This ensures the adapter starts with zero contribution and learns from there.

#### reset_adapter_parameters_random()
Random initialization for all parameters using Kaiming uniform. Not recommended for typical use.

#### reset_adapter_parameters_lycoris_way()
Initialization matching LyCORIS repository style:
- **w1**: Kaiming uniform
- **w1_a**: Kaiming uniform
- **w1_b**: Kaiming uniform
- **w2**: Zeros
- **w2_a**: Kaiming uniform
- **w2_b**: Zeros
- **t2**: Kaiming uniform

#### update_layer()
Main method to create and initialize a LoKr adapter for a layer.

**Process:**
1. Validate rank is positive
2. Store rank, alpha, scaling factor (alpha/r)
3. Store dropout parameters
4. Determine layer dimensions and compute factorization
5. Determine whether to use full matrices vs. factorized based on rank and dimension sizes
6. Create adapter parameters
7. Initialize weights according to init_weights setting
8. Move to device of base layer
9. Set adapter as active

**Factorization Decisions:**
- **use_w1**: Use full matrix if `decompose_both=False` OR `r >= max(left dimensions) / 2`
- **use_w2**: Use full matrix if `r >= max(right dimensions) / 2`
- **use_effective_conv2d**: Only for Conv2d/Conv1d with kernel_size > 1

**Special Handling for 1x1 Convolutions:**
For Conv2d/Conv1d with kernel_size=1:
- Disables effective_conv2d (no benefit for pointwise convolutions)
- Treats as essentially a Linear layer
- Avoids unnecessary tensor reshaping overhead

#### get_delta_weight()
Computes the weight update (delta) for a specific adapter.

**Process:**
1. Reconstruct w1:
   - If full w1 exists: Use directly
   - Else: Compute `w1_a @ w1_b`
2. Reconstruct w2:
   - If full w2 exists: Use directly
   - If t2 exists: Use `make_weight_cp()` for CP reconstruction
   - Else: Compute `w2_a @ w2_b`
3. Compute Kronecker product: `kron(w1, w2) * scaling`
4. Reshape to match base layer weight shape
5. Apply rank dropout if training:
   - Randomly drop entire output dimensions
   - Optionally scale remaining dimensions to compensate

**Returns:** Delta weight tensor matching base layer shape

#### forward()
Main forward pass that applies LoKr adaptation.

**Process:**
1. Store input dtype
2. Handle disable_adapters or merged state
3. Run base layer forward
4. For each active adapter:
   - Check module dropout (stochastic depth)
   - Get delta activations via `_get_delta_activations()`
   - Add to result
5. Restore original dtype
6. Return result

---

## Class: Linear(LoKrLayer)

LoKr implementation for `nn.Linear` layers.

### _get_delta_activations()
Computes LoKr contribution for linear layer:
1. Get delta weight from `get_delta_weight()`
2. Cast input to delta weight dtype
3. Apply `F.linear(input, delta_weight)` (no bias - already in base layer)
4. Return result

---

## Class: Conv2d(LoKrLayer)

LoKr implementation for `nn.Conv2d` layers.

### _get_delta_activations()
Computes LoKr contribution for 2D convolution:
1. Get delta weight from `get_delta_weight()`
2. Cast input to delta weight dtype
3. Apply `F.conv2d()` with base layer's stride, padding, dilation, groups
4. Return result

---

## Class: Conv1d(LoKrLayer)

LoKr implementation for `nn.Conv1d` layers.

### _get_delta_activations()
Computes LoKr contribution for 1D convolution:
1. Get delta weight from `get_delta_weight()`
2. Cast input to delta weight dtype
3. Apply `F.conv1d()` with base layer's stride, padding, dilation, groups
4. Return result

---

## Utility Functions

### factorization(dimension, factor=-1)
Factorizes a dimension into product of two numbers.

**Algorithm:**
1. If factor divides dimension evenly: Return (factor, dimension/factor)
2. Otherwise: Find two factors whose sum is minimal and close to sqrt(dimension)
3. Ensure first factor ≤ second factor

**Examples:**
- `factorization(256, -1)` → `(16, 16)`
- `factorization(128, -1)` → `(8, 16)`
- `factorization(128, 4)` → `(4, 32)`
- `factorization(127, -1)` → `(1, 127)` (prime number)

**Purpose:** Determines how to split dimensions for Kronecker product

### make_weight_cp(t, wa, wb)
Reconstructs weight from CP (CANDECOMP/PARAFAC) decomposition.

**Formula:** `einsum('i j k l, i p, j r -> p r k l', t, wa, wb)`

**Purpose:** Used for effective Conv2d decomposition where the kernel is represented as a core tensor with mode factors.

### make_kron(w1, w2, scale=1.0)
Computes scaled Kronecker product of two tensors.

**Process:**
1. If w2 is 4D (conv weights): Unsqueeze w1 to match dimensions
2. Compute `torch.kron(w1, w2)`
3. Multiply by scale factor
4. Return result

**Purpose:** Core operation that combines factorized matrices into final weight update

---

## Kronecker Product Mathematics

### What is a Kronecker Product?
For matrices A (m×n) and B (p×q), the Kronecker product A ⊗ B is an (mp)×(nq) matrix:
```
A ⊗ B = [a₁₁B  a₁₂B  ...
         a₂₁B  a₂₂B  ...
         ...]
```

### Why Use It for Neural Networks?
1. **Structural Prior**: Assumes weight structure can be decomposed
2. **Parameter Efficiency**: Instead of m×n parameters, use (m₁×n₁ + m₂×n₂) where m=m₁×m₂, n=n₁×n₂
3. **Further Factorization**: Each Kronecker factor can be low-rank factorized for even more efficiency

### Example
Weight matrix of size 256×256:
- Full rank: 65,536 parameters
- Kronecker (16×16) ⊗ (16×16): 512 parameters
- With rank-8 factorization: 256 parameters (16×8 + 8×16 for each factor)
- Compression ratio: ~256x fewer parameters!

---

## Dropout Mechanisms

### Rank Dropout
- Applied to output dimensions of the delta weight
- Randomly zeros out entire rows of the weight update
- Optional scaling to maintain expected magnitude
- Only active during training

### Module Dropout
- Stochastic depth style regularization
- Entire adapter either active or completely disabled for a forward pass
- Random decision per forward call during training
- Probability controlled by `module_dropout` config

---

## Design Decisions

### Why Three Initialization Methods?
- **Standard** (reset_adapter_parameters): Best for most cases, zero initial contribution
- **Random** (reset_adapter_parameters_random): For experimental use, non-zero start
- **LyCORIS** (reset_adapter_parameters_lycoris_way): Compatibility with LyCORIS repository

### Why Effective Conv2d?
For convolutional layers with spatial kernels, effective_conv2d uses CP decomposition which is more parameter-efficient than flattening the kernel dimensions. However, for 1×1 convolutions (pointwise), this adds unnecessary overhead with no benefit.

### Why Factorization Thresholds?
The decisions to use full vs. factorized matrices (based on r vs. dimension/2) balance:
- Small ranks: Factorization saves parameters
- Large ranks: Full matrix is simpler and may be more stable
- Threshold of dimension/2 is empirically reasonable

## Integration
These layers are automatically created and injected by LoKrModel when applying LoKr to a base model. The layer type (Linear, Conv2d, Conv1d) is selected based on the base layer type.
