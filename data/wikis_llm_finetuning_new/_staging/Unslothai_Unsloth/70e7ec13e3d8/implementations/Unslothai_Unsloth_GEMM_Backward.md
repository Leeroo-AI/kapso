# GEMM Backward Pass Kernels

[[domain::Kernels]] [[domain::MoE]] [[domain::GPU_Optimization]]

## Overview

The GEMM Backward module (`unsloth/kernels/moe/grouped_gemm/kernels/backward.py`) implements Triton kernels for computing gradients during backpropagation through grouped GEMM operations. It provides two kernels: one for input gradients (dX) and one for weight gradients (dW).

## Purpose

These backward kernels enable training of MoE models by:

1. **Computing input gradients (dX)** that flow back through the expert routing
2. **Computing weight gradients (dW)** for updating expert parameters
3. **Supporting permutation fusion** to match the forward pass memory layout
4. **Leveraging TMA** for efficient memory access where applicable

## Key Components

### dX Backward Kernel

#### `_grouped_gemm_dX_kernel`

Computes gradients with respect to input activations:

```python
@triton.jit
def _grouped_gemm_dX_kernel(
    dY_ptr,                    # [M_total, N] - gradient of output
    w_ptr,                     # [E, N, K] - expert weights
    dX_ptr,                    # [M_total, K] - gradient of input (output)
    gather_indices_ptr,        # Token-to-expert mapping
    m_sizes_ptr,               # Tokens per expert
    # Problem sizes
    NUM_EXPERTS: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    # Tuning parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    PERMUTE_X: tl.constexpr = False,
    PERMUTE_Y: tl.constexpr = False,
    USE_TMA_LOAD_W: tl.constexpr = False,
    USE_TMA_LOAD_dY: tl.constexpr = False,
    USE_TMA_STORE: tl.constexpr = False,
    FLATTEN: tl.constexpr = True,
) -> None
```

**Mathematical Operation:**
```
dX = dY @ W
# Where: dY is [M, N], W is [N, K], dX is [M, K]
```

**Shape Transformations:**
- Forward input X: `[NUM_TOKENS, K]` if `permute_x` else `[NUM_TOKENS * TOPK, K]`
- Forward output y: `[NUM_TOKENS * TOPK, N]`
- Backward input dY: `[NUM_TOKENS * TOPK, N]`
- Backward output dX: `[NUM_TOKENS * TOPK, K]` (always expanded)

**Permutation Handling:**

For `PERMUTE_X`:
- Forward: permuted load, contiguous store
- Backward: contiguous load, permuted store

```python
if PERMUTE_X:
    load_a_idx = indices_to_gather[:, None] * N  # Contiguous load
    store_idx = expert_token_offsets * K         # Permuted store
else:  # PERMUTE_Y case
    load_a_idx = expert_token_offsets * N        # Permuted load
    store_idx = indices_to_gather[:, None] * K   # Contiguous store
```

**TMA Descriptors:**

```python
if USE_TMA_LOAD_dY:
    dY_desc = tl._experimental_make_tensor_descriptor(
        dY_ptr,
        shape=[TOTAL_TOKENS, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

if USE_TMA_LOAD_W:
    w_desc = tl._experimental_make_tensor_descriptor(
        w_ptr,
        shape=[NUM_EXPERTS, N, K],
        strides=[expert_stride, K, 1],
        block_shape=[1, BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
```

**Main Loop Structure:**

```python
for expert_idx in range(NUM_EXPERTS, flatten=FLATTEN):
    m_start = m_end
    m_size = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
    m_end = m_start + m_size

    if m_size > 0:
        num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
        num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

        while tidx >= processed_tiles and tidx < (processed_tiles + num_tiles_per_expert):
            # Compute tile indices
            tile_m_idx = group_index % num_m_tiles
            tile_k_idx = group_index // num_m_tiles

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

            # GEMM reduction over N dimension
            for n_offset in range(0, N, BLOCK_SIZE_N):
                dY = load_dY_tile(...)
                w = load_weight_tile(...)
                accumulator += tl.dot(dY, w)  # [M, N] @ [N, K]

            store_dX_tile(accumulator.to(output_dtype), ...)
            tidx += NUM_SMS

        processed_tiles += num_tiles_per_expert
```

### dW Backward Kernel

#### `_grouped_gemm_dW_kernel`

Computes gradients with respect to expert weights:

```python
@triton.jit
def _grouped_gemm_dW_kernel(
    x_ptr,                     # [M, K] - input activations
    dY_ptr,                    # [M, N] - gradient of output
    dW_ptr,                    # [E, N, K] - gradient of weights (output)
    m_sizes_ptr,               # Tokens per expert
    gather_indices_ptr,        # Token-to-expert mapping
    # Problem sizes
    NUM_TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    PERMUTE_X: tl.constexpr = False,
    PERMUTE_Y: tl.constexpr = False,
    USE_TMA_LOAD_dY: tl.constexpr = False,
    USE_TMA_LOAD_X: tl.constexpr = False,
    USE_TMA_STORE: tl.constexpr = False,
    FLATTEN: tl.constexpr = True,
    acc_dtype: tl.constexpr = tl.float32,
) -> None
```

**Mathematical Operation:**
```
dW[e] = dY[e].T @ X[e]
# For each expert e: dY is [M_e, N], X is [M_e, K], dW is [N, K]
```

**Output Tiling:**

Unlike dX which tiles over M and K, dW tiles over N and K:

```python
num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
output_tiles_per_expert = num_n_tiles * num_k_tiles

for tile_idx in range(tidx, output_tiles_per_expert, NUM_SMS):
    tile_n_idx = tile_idx % num_n_tiles
    tile_k_idx = tile_idx // num_n_tiles

    n_offset = tile_n_idx * BLOCK_SIZE_N
    k_offset = tile_k_idx * BLOCK_SIZE_K
```

**Reduction Over Tokens:**

The kernel accumulates over all tokens assigned to each expert:

```python
for expert_idx in range(NUM_EXPERTS):
    accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=acc_dtype)

    m_start = m_end
    m_size = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
    m_end = m_start + m_size

    if m_size > 0:
        for tile_m_idx in range(0, m_size, BLOCK_SIZE_M):
            x = load_x_tile(...)     # [BLOCK_M, BLOCK_K]
            dY = load_dY_tile(...)   # [BLOCK_M, BLOCK_N]

            accumulator += tl.dot(
                dY.T,  # [BLOCK_N, BLOCK_M]
                x,     # [BLOCK_M, BLOCK_K]
            )

        store_dW_tile(accumulator.to(output_dtype), expert_idx, n_offset, k_offset)
```

**Permutation Handling for dW:**

```python
if PERMUTE_X:
    # Forward permuted on load -> index from original token count
    x_row_load_idx = (expert_token_offsets // TOPK) * K
    dY_row_load_idx = m_offsets[:, None] * N  # Contiguous
elif PERMUTE_Y:
    # Forward permuted on store -> dY needs permuted load
    x_row_load_idx = indices_to_gather[:, None] * K  # Contiguous
    dY_row_load_idx = expert_token_offsets * N       # Permuted
else:
    # No permutation
    x_row_load_idx = m_offsets[:, None] * K
    dY_row_load_idx = m_offsets[:, None] * N
```

**TMA Store for dW:**

```python
if USE_TMA_STORE:
    dW_desc = tl._experimental_make_tensor_descriptor(
        dW_ptr,
        shape=[NUM_EXPERTS, N, K],
        strides=[N * K, K, 1],
        block_shape=[1, BLOCK_SIZE_N, BLOCK_SIZE_K],
    )

    # Store requires expanding dims to match 3D shape
    y = tl.expand_dims(accumulator.to(output_dtype), 0)
    dW_desc.store([expert_idx, n_offset, k_offset], y)
```

### Autotuned Variants

Both kernels have autotuned versions:

```python
_autotuned_grouped_gemm_dX_kernel = triton.autotune(
    configs=get_dX_kernel_configs(),
    prune_configs_by={"early_config_prune": prune_dX_configs},
    key=["NUM_EXPERTS", "NUM_TOKENS", "N", "K", "PERMUTE_X", "PERMUTE_Y"],
)(_grouped_gemm_dX_kernel)

_autotuned_grouped_gemm_dW_kernel = triton.autotune(
    configs=get_dW_kernel_configs(),
    prune_configs_by={"early_config_prune": prune_kernel_configs_backward_dW},
    key=["NUM_EXPERTS", "NUM_TOKENS", "N", "K", "PERMUTE_X", "PERMUTE_Y"],
)(_grouped_gemm_dW_kernel)
```

## TMA Constraints

**dX Kernel:**
- `USE_TMA_LOAD_dY` incompatible with `PERMUTE_Y` (scattered access pattern)
- `USE_TMA_STORE` incompatible with `PERMUTE_X` (scattered store pattern)
- `USE_TMA_LOAD_W` always safe (contiguous expert weight access)

**dW Kernel:**
- `USE_TMA_LOAD_dY` incompatible with `PERMUTE_Y`
- `USE_TMA_LOAD_X` incompatible with `PERMUTE_X`
- `USE_TMA_STORE` always safe (writing to contiguous weight tensor)

## Static Assertions

```python
# dX kernel
tl.static_assert(N % BLOCK_SIZE_N == 0, "N must be divisible by BLOCK_SIZE_N")
tl.static_assert(K % BLOCK_SIZE_K == 0, "K must be divisible by BLOCK_SIZE_K")

# dW kernel (for TMA store)
tl.static_assert(N % BLOCK_SIZE_N == 0, "N must be divisible by BLOCK_SIZE_N")
tl.static_assert(K % BLOCK_SIZE_K == 0, "K must be divisible by BLOCK_SIZE_K")
```

## Post-Processing Requirements

After the dX kernel completes, additional processing may be needed:

```python
# In the autograd backward function:
if topk > 1 and permute_x:
    # dX shape is [NUM_TOKENS * TOPK, K]
    # Need to reduce across topk dimension
    dX = dX.view(X.shape[0], topk, -1).sum(dim=1)
```

## Implementation Notes

1. **Persistent Thread Blocks**: Both kernels use a persistent approach where thread blocks iterate over tiles rather than launching one block per tile

2. **Expert Loop Flattening**: The `FLATTEN` parameter enables Triton's loop flattening optimization for better instruction scheduling

3. **Dynamic TMA Descriptors**: When using TMA with expert boundaries, descriptors are created inside the loop to handle variable M sizes:
   ```python
   if TMA_LOAD_BOTH:
       dY_desc = tl._experimental_make_tensor_descriptor(
           dY_ptr,
           shape=[m_end, N],  # Uses current expert's end position
           ...
       )
   ```

4. **Index Optimization**: Uses `tl.max_contiguous` and `tl.multiple_of` hints for better memory coalescing

## Dependencies

- `torch`: Tensor operations
- `triton`, `triton.language`: Kernel JIT compilation
- `grouped_gemm.kernels.autotuning`: Configuration generators and pruning functions

## Source File

`unsloth/kernels/moe/grouped_gemm/kernels/backward.py` (502 lines)

## License

GNU Affero General Public License v3.0 - Copyright 2023-present the Unsloth team.
