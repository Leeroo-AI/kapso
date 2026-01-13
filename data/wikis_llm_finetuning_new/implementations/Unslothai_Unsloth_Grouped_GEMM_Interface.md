# Grouped GEMM Interface

[[domain::Kernels]] [[domain::MoE]] [[domain::GPU_Optimization]]

## Overview

The Grouped GEMM Interface (`unsloth/kernels/moe/grouped_gemm/interface.py`) provides the main API for grouped General Matrix Multiply (GEMM) operations used in Mixture of Experts (MoE) layers. This module serves as the primary entry point for efficient batched matrix multiplications where different groups of tokens are processed by different expert networks.

## Purpose

This implementation enables high-performance MoE computations by:

1. **Fusing permutation operations** with GEMM computations to reduce memory bandwidth requirements
2. **Supporting TMA (Tensor Memory Accelerator)** for efficient memory access on Hopper (sm90+) and newer GPUs
3. **Providing both autotuned and manually configured kernel variants** for flexibility between ease of use and fine-grained control
4. **Implementing full autograd support** for training through the `GroupedGemm` class

## Key Components

### Core Functions

#### `grouped_gemm_forward`

The forward pass implementation for grouped GEMM in MoE MLPs.

```python
def grouped_gemm_forward(
    X: torch.Tensor,           # (M, K) hidden states
    W: torch.Tensor,           # (E, N, K) expert weights
    topk: int,                 # number of experts per token
    m_sizes: torch.Tensor,     # tokens assigned to each expert
    gather_indices: torch.Tensor = None,  # token-to-expert mapping
    topk_weights: torch.Tensor = None,    # routing weights
    # Fusions
    permute_x: bool = False,   # fuse input permutation
    permute_y: bool = False,   # fuse output permutation
    fuse_mul_post: bool = False,  # fuse weight multiplication (inference only)
    # Autotuning
    autotune: bool = False,
    # Manual kernel params
    BLOCK_SIZE_M: int = 32,
    BLOCK_SIZE_N: int = 32,
    BLOCK_SIZE_K: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
    use_tma_load_w: bool = False,
    use_tma_load_x: bool = False,
    use_tma_store: bool = False,
    ...
) -> torch.Tensor
```

**Fusion Options:**
- `permute_x`: Fuses permutation from token order to grouped expert order on load (first GEMM in MoE MLP)
- `permute_y`: Fuses permutation from expert order back to token order on store (second GEMM in MoE MLP)
- `fuse_mul_post`: Fuses multiplication with topk weights (inference only, not for training)

#### `grouped_gemm_dX`

Backward kernel for computing gradients with respect to input activations.

```python
def grouped_gemm_dX(
    dY: torch.Tensor,          # gradient of output
    W: torch.Tensor,           # expert weights
    gather_indices: torch.Tensor,
    m_sizes: torch.Tensor,
    topk: int,
    ...
) -> torch.Tensor  # returns dX gradient
```

Key characteristics:
- Output shape is `[NUM_TOKENS * TOPK, K]` even when `permute_x` was used in forward
- Requires post-processing reduction step to accumulate gradients across experts

#### `grouped_gemm_dW`

Backward kernel for computing gradients with respect to expert weights.

```python
def grouped_gemm_dW(
    X: torch.Tensor,           # input activations
    dY: torch.Tensor,          # gradient of output
    m_sizes: torch.Tensor,
    gather_indices: torch.Tensor,
    topk: int,
    ...
) -> torch.Tensor  # returns dW gradient shape (E, N, K)
```

### Autograd Integration

#### `GroupedGemm` Class

A `torch.autograd.Function` that provides seamless integration with PyTorch's automatic differentiation:

```python
class GroupedGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, m_sizes, topk, gather_indices, permute_x, permute_y,
                topk_weights, fuse_mul_post, kernel_config_fwd,
                kernel_config_bwd_dX, kernel_config_bwd_dW, autotune, dX_only, dW_only):
        # Saves tensors for backward, calls grouped_gemm_forward
        ...

    @staticmethod
    def backward(ctx, dY):
        # Computes dX and dW using grouped_gemm_dX and grouped_gemm_dW
        ...
```

#### `grouped_gemm` Function

The main user-facing API that wraps the autograd function:

```python
def grouped_gemm(
    X: torch.Tensor,
    W: torch.Tensor,
    m_sizes: torch.Tensor,
    topk: int,
    gather_indices: torch.Tensor = None,
    permute_x: bool = False,
    permute_y: bool = False,
    topk_weights = None,
    fuse_mul_post = False,
    kernel_config_fwd: KernelConfigForward = None,
    kernel_config_bwd_dX: KernelConfigBackward_dX = None,
    kernel_config_bwd_dW: KernelConfigBackward_dW = None,
    autotune: bool = False,
    is_first_gemm: bool = True,
    ...
)
```

### TMA Support

The module includes TMA (Tensor Memory Accelerator) support for sm90+ GPUs:

```python
def supports_tma():
    global _SUPPORTS_TMA
    if _SUPPORTS_TMA is None:
        _SUPPORTS_TMA = torch.cuda.get_device_capability()[0] >= 9
    return _SUPPORTS_TMA
```

TMA constraints:
- `use_tma_load_x` is incompatible with `permute_x` (until Blackwell+ with TMA gather/scatter)
- `use_tma_store` is incompatible with `permute_y` (until Blackwell+ with TMA gather/scatter)
- `use_tma_load_w` should generally be enabled when TMA is supported for better performance

### Configuration Validation

Helper functions validate kernel configurations:

```python
def check_valid_config_fwd(permute_x, permute_y, use_tma_load_x, use_tma_load_w,
                            use_tma_store, fuse_mul_post, is_first_gemm):
    # Validates forward pass configuration

def check_valid_config_bwd_dW(permute_x, permute_y, use_tma_load_dY,
                               use_tma_load_x, use_tma_store, fuse_mul_post, is_first_gemm):
    # Validates dW backward configuration

def check_valid_config_bwd_dX(permute_x, permute_y, use_tma_load_dY,
                               use_tma_load_w, use_tma_store, fuse_mul_post, is_first_gemm):
    # Validates dX backward configuration
```

## Memory Management

The module manages GPU memory allocation for TMA operations:

```python
def get_per_device_per_stream_alloc_fn(device):
    # Returns per-device, per-stream allocation function
    # Maintains tensor cache to avoid repeated allocations
    # Uses 128-byte alignment for TMA compatibility
```

## Usage Patterns

### First GEMM in MoE MLP (Gate/Up Projection)

```python
# permute_x=True: Load tokens in original order, output in expert-grouped order
y = grouped_gemm(
    X=hidden_states,        # (num_tokens, hidden_dim)
    W=expert_weights,       # (num_experts, intermediate_dim, hidden_dim)
    m_sizes=tokens_per_expert,
    topk=top_k,
    gather_indices=sorted_indices,
    permute_x=True,
    is_first_gemm=True,
    kernel_config_fwd=fwd_config,
    kernel_config_bwd_dX=bwd_dX_config,
    kernel_config_bwd_dW=bwd_dW_config,
)
```

### Second GEMM in MoE MLP (Down Projection)

```python
# permute_y=True: Load in expert-grouped order, output in original token order
y = grouped_gemm(
    X=intermediate_states,  # (num_tokens * topk, intermediate_dim)
    W=down_proj_weights,    # (num_experts, hidden_dim, intermediate_dim)
    m_sizes=tokens_per_expert,
    topk=top_k,
    gather_indices=sorted_indices,
    permute_y=True,
    is_first_gemm=False,
    kernel_config_fwd=fwd_config,
    kernel_config_bwd_dX=bwd_dX_config,
    kernel_config_bwd_dW=bwd_dW_config,
)
```

## Implementation Details

### Grid Configuration

The kernels use a persistent thread block approach:

```python
NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

def grid(META):
    return (NUM_SMS,)  # Launch one block per SM
```

### Kernel Logging

Debug information is available through logging:

```python
def log_kernel_info(compiled_kernel, best_config=None):
    # Logs register usage, spills, and metadata
    # Reports autotuned best_config when available
```

## Dependencies

- `torch`: PyTorch tensor operations
- `triton`: Triton compiler for GPU kernels
- Forward kernel: `grouped_gemm.kernels.forward`
- Backward kernels: `grouped_gemm.kernels.backward`
- Tuning configurations: `grouped_gemm.kernels.tuning`

## Performance Considerations

1. **Autotuning vs Manual Config**: Use `autotune=True` for automatic optimization, or provide manual `KernelConfig*` objects for reproducible performance
2. **TMA Usage**: Enable TMA loads/stores when supported (sm90+) for better memory bandwidth
3. **Block Sizes**: Adjust `BLOCK_SIZE_M/N/K` based on problem dimensions and GPU architecture
4. **Fusion**: Use `permute_x`/`permute_y` to avoid separate permutation kernels

## Source File

`unsloth/kernels/moe/grouped_gemm/interface.py` (968 lines)

## License

GNU Affero General Public License v3.0 - Copyright 2023-present the Unsloth team.
