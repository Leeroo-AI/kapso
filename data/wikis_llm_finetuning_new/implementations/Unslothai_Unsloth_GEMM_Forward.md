# GEMM Forward Pass Kernel

[[domain::Kernels]] [[domain::MoE]] [[domain::GPU_Optimization]]

## Overview

The GEMM Forward module (`unsloth/kernels/moe/grouped_gemm/kernels/forward.py`) implements the Triton kernel for the forward pass of grouped GEMM operations in MoE layers. This kernel computes batched matrix multiplications where tokens are routed to different experts.

## Purpose

This forward kernel enables efficient MoE computation by:

1. **Computing grouped matrix multiplications** for expert layers
2. **Fusing token permutation** with GEMM to reduce memory traffic
3. **Supporting topk weight multiplication** for inference optimization
4. **Leveraging TMA** for high-bandwidth memory access on Hopper GPUs

## Key Components

### Main Kernel

#### `_grouped_gemm_forward_kernel`

The core forward pass implementation:

```python
@triton.jit
def _grouped_gemm_forward_kernel(
    x_ptr,                     # Input activations
    w_ptr,                     # Expert weights [E, N, K]
    y_ptr,                     # Output
    # Variable depending on routed probs
    m_sizes_ptr,               # Tokens per expert
    gather_indices_ptr,        # Token-to-expert mapping
    topk_weights_ptr,          # Routing weights (optional)
    # Constant problem shapes
    NUM_EXPERTS: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    N: tl.constexpr,           # Output dimension
    K: tl.constexpr,           # Reduction dimension
    NUM_SMS: tl.constexpr,
    # Tuning params
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    PERMUTE_X: tl.constexpr = False,
    PERMUTE_Y: tl.constexpr = False,
    FUSE_MUL_PRE: tl.constexpr = False,
    FUSE_MUL_POST: tl.constexpr = False,
    USE_FAST_ACCUM: tl.constexpr = False,
    USE_TMA_LOAD_W: tl.constexpr = False,
    USE_TMA_LOAD_X: tl.constexpr = False,
    USE_TMA_STORE: tl.constexpr = False,
    acc_dtype: tl.constexpr = tl.float32,
    FLATTEN: tl.constexpr = True,
) -> None
```

**Mathematical Operation:**
```
Y[e] = X[e] @ W[e].T
# For each expert e: X is [M_e, K], W is [N, K], Y is [M_e, N]
```

### Permutation Modes

#### `PERMUTE_X` (First GEMM in MoE)

Loads tokens in original order and outputs in expert-grouped order:

```python
if PERMUTE_X:
    # Load from token positions divided by TOPK (original token indices)
    load_idx = (expert_token_offsets // TOPK) * K
    # Store contiguously in expert-grouped order
    store_idx = indices_to_gather[:, None] * N
```

Use case: Gate/Up projection where input is in token order.

#### `PERMUTE_Y` (Second GEMM in MoE)

Loads tokens in expert-grouped order and outputs in original token order:

```python
if PERMUTE_Y:
    # Load contiguously from expert-grouped positions
    load_idx = indices_to_gather[:, None] * K
    # Store at original token positions
    store_idx = expert_token_offsets * N
```

Use case: Down projection where output needs to be in token order for residual addition.

### TMA Descriptors

For high-performance memory access on Hopper GPUs:

```python
# Input tensor descriptor (when not permuting)
if USE_TMA_LOAD_X:
    x_desc = tl._experimental_make_tensor_descriptor(
        x_ptr,
        shape=[TOTAL_TOKENS, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )

# Weight tensor descriptor (3D for expert indexing)
if USE_TMA_LOAD_W:
    expert_stride = N * K
    w_desc = tl._experimental_make_tensor_descriptor(
        w_ptr,
        shape=[NUM_EXPERTS, N, K],
        strides=[expert_stride, K, 1],
        block_shape=[1, BLOCK_SIZE_N, BLOCK_SIZE_K],
    )

# Output tensor descriptor (created per-expert for M predication)
if USE_TMA_STORE:
    y_desc = tl._experimental_make_tensor_descriptor(
        y_ptr,
        shape=[m_end, N],  # Dynamic based on expert
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
```

### Fused Weight Multiplication

For inference optimization:

```python
# Pre-multiplication (before GEMM - causes performance regression, not recommended)
if FUSE_MUL_PRE:
    topk_weights = tl.load(topk_weights_ptr + topk_load_idx, mask=row_mask)
    x *= topk_weights.to(x.dtype)

# Post-multiplication (after GEMM - recommended for inference)
if FUSE_MUL_POST:
    topk_weights = tl.load(topk_weights_ptr + topk_load_idx, mask=row_mask)
    y *= topk_weights.to(output_dtype)
```

### Kernel Structure

```python
# Static assertion for dimension divisibility
tl.static_assert(K % BLOCK_SIZE_K == 0)

TOTAL_TOKENS: tl.constexpr = NUM_TOKENS * TOPK
tidx = tl.program_id(0)

m_end = 0
processed_tiles = 0
m_block_range = tl.arange(0, BLOCK_SIZE_M)

# Iterate over experts
for expert_idx in tl.range(NUM_EXPERTS, flatten=FLATTEN):
    m_start = m_end
    m_size = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
    m_end = m_start + m_size

    if m_size > 0:
        n_start = expert_idx * N

        num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
        num_tiles_per_expert = num_m_tiles * num_n_tiles

        # Persistent thread block approach
        while tidx >= processed_tiles and tidx < processed_tiles + num_tiles_per_expert:
            tile_idx = tidx - processed_tiles

            # M-major tile ordering for L2 cache reuse
            tile_m_idx = tile_idx % num_m_tiles
            tile_n_idx = tile_idx // num_m_tiles

            # Setup load/store indices based on permutation mode
            ...

            # Initialize accumulator
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

            # GEMM reduction over K dimension
            for k_offset in range(0, K, BLOCK_SIZE_K):
                x = load_x_tile(...)

                if FUSE_MUL_PRE:
                    x *= topk_weights

                w = load_w_tile(...)

                # Matrix multiply: [M, K] @ [K, N] -> [M, N]
                accumulator += tl.dot(x, w.T)

            y = accumulator.to(output_dtype)

            if FUSE_MUL_POST:
                y *= topk_weights

            # Store output tile
            store_y_tile(y, ...)

            tidx += NUM_SMS

        processed_tiles += num_tiles_per_expert
```

### Autotuned Variant

```python
_autotuned_grouped_gemm_forward_kernel = triton.autotune(
    configs=get_forward_configs(),
    prune_configs_by={"early_config_prune": prune_kernel_configs_fwd},
    key=[
        "NUM_EXPERTS",
        "NUM_TOKENS",
        "N",
        "K",
        "PERMUTE_X",
        "PERMUTE_Y",
        "FUSE_MUL_POST",
    ],
)(_grouped_gemm_forward_kernel)
```

Autotuning keys ensure different configurations are cached for:
- Different model sizes (N, K dimensions)
- Different batch sizes (NUM_TOKENS)
- Different expert counts (NUM_EXPERTS)
- Different fusion modes

## Memory Access Patterns

### Coalesced Access Optimization

```python
# Use contiguous/multiple_of hints for better coalescing
offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
offs_bn = tl.max_contiguous(
    tl.multiple_of(offs_bn % N, BLOCK_SIZE_N), BLOCK_SIZE_N
)
```

### Gather Index Optimization

```python
indices_to_gather = m_start + tl.max_contiguous(
    tl.multiple_of(gather_offsets % m_size, BLOCK_SIZE_M),
    BLOCK_SIZE_M,
)
```

## TMA Constraints

| Option | Constraint |
|--------|------------|
| `USE_TMA_LOAD_X` | Incompatible with `PERMUTE_X` (requires gather) |
| `USE_TMA_LOAD_W` | Always valid (contiguous expert weights) |
| `USE_TMA_STORE` | Incompatible with `PERMUTE_Y` (requires scatter) |

Note: Future Blackwell GPUs with TMA gather/scatter will remove these constraints.

## Tile Ordering

The kernel uses M-major tile ordering:
```python
tile_m_idx = tile_idx % num_m_tiles
tile_n_idx = tile_idx // num_m_tiles
```

This prioritizes L2 cache reuse for the weight matrix when processing multiple M tiles before moving to the next N tile.

## Performance Considerations

1. **Block Size Selection**:
   - Larger blocks improve arithmetic intensity but may exceed shared memory
   - `BLOCK_SIZE_K` should divide K evenly for optimal performance

2. **TMA Usage**:
   - Enable TMA loads when not using permutation for 2-3x memory bandwidth improvement
   - TMA store requires creating descriptors per-expert due to variable M sizes

3. **Fusion**:
   - `FUSE_MUL_POST` is preferred over `FUSE_MUL_PRE` as pre-multiplication interrupts the GEMM mainloop
   - Only use fusion for inference; training requires separate gradient computation

4. **Loop Flattening**:
   - `FLATTEN=True` enables Triton's loop optimization for better instruction-level parallelism

## Dependencies

- `torch`: Tensor operations
- `triton`, `triton.language`: Kernel JIT compilation
- `grouped_gemm.kernels.autotuning`: Configuration generators and pruning

## Source File

`unsloth/kernels/moe/grouped_gemm/kernels/forward.py` (265 lines)

## License

GNU Affero General Public License v3.0 - Copyright 2023-present the Unsloth team.
