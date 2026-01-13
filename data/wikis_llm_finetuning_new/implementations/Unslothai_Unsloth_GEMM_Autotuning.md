# GEMM Autotuning Infrastructure

[[domain::Kernels]] [[domain::MoE]] [[domain::GPU_Optimization]]

## Overview

The GEMM Autotuning module (`unsloth/kernels/moe/grouped_gemm/kernels/autotuning.py`) provides infrastructure for automatically tuning grouped GEMM kernel parameters. It generates configuration spaces, applies pruning heuristics, and manages the autotuning process for forward and backward kernels.

## Purpose

This module enables optimal kernel performance by:

1. **Generating configuration search spaces** for block sizes, warp counts, and pipeline stages
2. **Pruning invalid or suboptimal configurations** based on hardware constraints and problem characteristics
3. **Integrating with Triton's autotuning framework** for runtime kernel selection
4. **Estimating shared memory requirements** to avoid resource exhaustion

## Key Components

### Default Configuration Constants

```python
DEFAULT_M_BLOCK_SIZES = [64, 128]
DEFAULT_N_BLOCK_SIZES = [64, 128, 256]
DEFAULT_K_BLOCK_SIZES = [64, 128, 256]
DEFAULT_NUM_CTAS = 1
DEFAULT_NUM_WARPS = [4, 8]
DEFAULT_NUM_STAGES = [3, 4, 5]
BOOLS = [True, False]
```

### Configuration Generators

#### `get_forward_configs`

Generates Triton configurations for the forward pass kernel:

```python
def get_forward_configs(
    BLOCK_M = DEFAULT_M_BLOCK_SIZES,
    BLOCK_N = DEFAULT_N_BLOCK_SIZES,
    BLOCK_K = DEFAULT_K_BLOCK_SIZES,
    TMA_LOAD_X = True,
    TMA_LOAD_W = True,
    TMA_STORE = False,  # Disabled by default
    num_warps = DEFAULT_NUM_WARPS,
    num_stages = DEFAULT_NUM_STAGES,
    num_ctas = DEFAULT_NUM_CTAS,
):
    # Returns list of triton.Config objects
```

Each configuration specifies:
- `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`: Tile dimensions
- `USE_TMA_LOAD_X`, `USE_TMA_LOAD_W`, `USE_TMA_STORE`: TMA usage flags
- `num_warps`: Number of warps per thread block
- `num_stages`: Software pipelining stages
- `num_ctas`: Cooperative Thread Arrays count

#### `get_dX_kernel_configs`

Generates configurations for the dX backward kernel:

```python
def get_dX_kernel_configs(
    BLOCK_M = DEFAULT_M_BLOCK_SIZES,
    BLOCK_N = DEFAULT_N_BLOCK_SIZES,
    BLOCK_K = DEFAULT_K_BLOCK_SIZES,
    TMA_LOAD_dY = True,
    TMA_LOAD_W = True,
    TMA_STORE = False,
    num_warps = DEFAULT_NUM_WARPS,
    num_stages = DEFAULT_NUM_STAGES,
    num_ctas = DEFAULT_NUM_CTAS,
):
    # Returns configurations for input gradient computation
```

#### `get_dW_kernel_configs`

Generates configurations for the dW backward kernel:

```python
def get_dW_kernel_configs(
    BLOCK_M = DEFAULT_M_BLOCK_SIZES,
    BLOCK_N = DEFAULT_N_BLOCK_SIZES,
    BLOCK_K = DEFAULT_K_BLOCK_SIZES,
    num_warps = DEFAULT_NUM_WARPS,
    num_stages = DEFAULT_NUM_STAGES,
    num_ctas = DEFAULT_NUM_CTAS,
    TMA_LOAD_dY = True,
    TMA_LOAD_X = True,
    TMA_STORE = False,
):
    # Returns configurations for weight gradient computation
```

### Shared Memory Estimation

#### `estimate_smem_reqs`

Calculates shared memory requirements for a given configuration:

```python
def estimate_smem_reqs(
    num_stages: int,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    dtype: torch.dtype,
):
    num_bytes = dtype.itemsize
    return (
        num_stages * BLOCK_SIZE_K * (BLOCK_SIZE_M + BLOCK_SIZE_N)
        + BLOCK_SIZE_M * BLOCK_SIZE_N
    ) * num_bytes
```

This accounts for:
- Double/multi-buffered input tiles: `num_stages * BLOCK_SIZE_K * (BLOCK_SIZE_M + BLOCK_SIZE_N)`
- Accumulator tile: `BLOCK_SIZE_M * BLOCK_SIZE_N`

#### `exceeds_smem_capacity`

Checks if a configuration exceeds available shared memory:

```python
def exceeds_smem_capacity(
    num_stages: int,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    dtype: torch.dtype,
    smem_size: int,
    slack: float = 50000,  # Safety margin in bytes
):
    smem_reqs = estimate_smem_reqs(...)
    return smem_reqs > smem_size + slack
```

### Configuration Pruning

#### `common_prune_criteria`

Shared pruning logic applied to all kernel types:

```python
def common_prune_criteria(config: triton.Config, kwargs: dict, dtype):
    # Get device shared memory capacity
    smem_size = get_device_properties().SIZE_SMEM

    # Extract config parameters
    num_stages = config.num_stages
    BLOCK_SIZE_M = config.kwargs["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = config.kwargs["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = config.kwargs["BLOCK_SIZE_K"]

    # Extract problem parameters
    num_tokens = kwargs["NUM_TOKENS"]
    num_experts = kwargs["NUM_EXPERTS"]
    permute_x = kwargs["PERMUTE_X"]
    permute_y = kwargs["PERMUTE_Y"]
    tokens_per_expert = num_tokens // num_experts

    # Prune criteria:
    # 1. Exceeds shared memory capacity
    if exceeds_smem_capacity(...):
        return True

    # 2. Block size too large for problem size
    MIN_BLOCK_SIZE_M = DEFAULT_M_BLOCK_SIZES[0]
    if BLOCK_SIZE_M > tokens_per_expert * 2 and tokens_per_expert > MIN_BLOCK_SIZE_M:
        return True

    # 3. Invalid permutation combination
    if permute_x and permute_y:
        return True

    return False
```

#### `maybe_disable_tma`

Dynamically disables TMA on unsupported hardware:

```python
def maybe_disable_tma(config: triton.Config):
    tma_keys = [k for k in config.kwargs.keys() if k.startswith("USE_TMA_")]
    if not supports_tma():
        logger.info("Disabling TMA")
        for k in tma_keys:
            config.kwargs[k] = False
```

#### `prune_kernel_configs_fwd`

Forward-specific pruning:

```python
def prune_kernel_configs_fwd(configs: list[triton.Config], args, **kwargs):
    x = kwargs["x_ptr"]
    dtype = x.dtype

    pruned_configs = []
    for config in configs:
        maybe_disable_tma(config)

        if common_prune_criteria(config, kwargs, dtype):
            continue

        # Disable TMA load for permuted X (TMA doesn't support gather)
        if config.kwargs["USE_TMA_LOAD_X"] and kwargs["PERMUTE_X"]:
            config.kwargs["USE_TMA_LOAD_X"] = False

        # Skip TMA store with permuted Y
        if config.kwargs["USE_TMA_STORE"] and kwargs["PERMUTE_Y"]:
            continue

        pruned_configs.append(config)

    return pruned_configs
```

#### `prune_dX_configs`

Backward dX kernel pruning:

```python
def prune_dX_configs(configs: List[triton.Config], args, **kwargs):
    dtype = kwargs["w_ptr"].dtype

    pruned_configs = []
    for config in configs:
        if common_prune_criteria(config, kwargs, dtype):
            continue

        # Disable TMA load for permuted Y (need to load in scattered order)
        if config.kwargs["USE_TMA_LOAD_dY"] and kwargs["PERMUTE_Y"]:
            config.kwargs["USE_TMA_LOAD_dY"] = False

        # Skip TMA store with permuted X (need to scatter on store)
        if config.kwargs["USE_TMA_STORE"] and kwargs["PERMUTE_X"]:
            continue

        pruned_configs.append(config)

    return pruned_configs
```

#### `prune_kernel_configs_backward_dW`

Backward dW kernel pruning:

```python
def prune_kernel_configs_backward_dW(configs: list[triton.Config], args, **kwargs):
    dtype = kwargs["x_ptr"].dtype

    pruned_configs = []
    for config in configs:
        if common_prune_criteria(config, kwargs, dtype):
            continue

        # Disable TMA loads for permuted inputs
        if config.kwargs["USE_TMA_LOAD_dY"] and kwargs["PERMUTE_Y"]:
            config.kwargs["USE_TMA_LOAD_dY"] = False
        if config.kwargs["USE_TMA_LOAD_X"] and kwargs["PERMUTE_X"]:
            config.kwargs["USE_TMA_LOAD_X"] = False

        pruned_configs.append(config)

    return pruned_configs
```

### Helper Functions

#### `val_to_list`

Normalizes configuration values to lists:

```python
def val_to_list(val):
    if val is None:
        return None
    elif isinstance(val, list):
        return val
    else:
        return [val]
```

#### `convert_args_to_list`

Batch conversion for configuration generators:

```python
def convert_args_to_list(args):
    return [val_to_list(arg) for arg in args]
```

## Usage in Kernels

The autotuning configurations are used with Triton's `@triton.autotune` decorator:

```python
# Forward kernel
_autotuned_grouped_gemm_forward_kernel = triton.autotune(
    configs=get_forward_configs(),
    prune_configs_by={"early_config_prune": prune_kernel_configs_fwd},
    key=["NUM_EXPERTS", "NUM_TOKENS", "N", "K", "PERMUTE_X", "PERMUTE_Y", "FUSE_MUL_POST"],
)(_grouped_gemm_forward_kernel)

# dX backward kernel
_autotuned_grouped_gemm_dX_kernel = triton.autotune(
    configs=get_dX_kernel_configs(),
    prune_configs_by={"early_config_prune": prune_dX_configs},
    key=["NUM_EXPERTS", "NUM_TOKENS", "N", "K", "PERMUTE_X", "PERMUTE_Y"],
)(_grouped_gemm_dX_kernel)

# dW backward kernel
_autotuned_grouped_gemm_dW_kernel = triton.autotune(
    configs=get_dW_kernel_configs(),
    prune_configs_by={"early_config_prune": prune_kernel_configs_backward_dW},
    key=["NUM_EXPERTS", "NUM_TOKENS", "N", "K", "PERMUTE_X", "PERMUTE_Y"],
)(_grouped_gemm_dW_kernel)
```

## Configuration Space Size

The total number of configurations before pruning:

```
Forward: |BLOCK_M| * |BLOCK_N| * |BLOCK_K| * |num_warps| * |num_stages| * |TMA_LOAD_X| * |TMA_LOAD_W| * |TMA_STORE| * |num_ctas|
       = 2 * 3 * 3 * 2 * 3 * 2 * 2 * 2 * 1 = 864 configurations
```

After pruning based on:
- Shared memory constraints
- Problem size vs block size
- TMA compatibility with permutation
- Hardware capability

Typically reduces to 10-50 viable configurations per problem instance.

## Performance Impact

1. **Early Pruning**: Reduces autotuning time by eliminating invalid configurations before benchmarking
2. **Problem-Aware**: Adapts configuration space based on actual tensor dimensions
3. **Hardware-Aware**: Respects shared memory limits and TMA availability
4. **Cache-Friendly**: Autotuning results are cached by Triton based on the `key` parameters

## Dependencies

- `torch`: For tensor dtype information
- `triton`: For Config objects and autotuning integration
- `grouped_gemm.interface`: For `supports_tma()` function
- `grouped_gemm.kernels.tuning`: For `get_device_properties()`

## Source File

`unsloth/kernels/moe/grouped_gemm/kernels/autotuning.py` (396 lines)

## License

GNU Affero General Public License v3.0 - Copyright 2023-present the Unsloth team.
