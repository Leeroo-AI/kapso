# File: `unsloth/kernels/moe/grouped_gemm/interface.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 968 |
| Classes | `GroupedGemm` |
| Functions | `supports_tma`, `get_per_device_per_stream_alloc_fn`, `log_kernel_info`, `grouped_gemm_forward`, `grouped_gemm_dX`, `grouped_gemm_dW`, `check_valid_config_fwd`, `check_valid_config_bwd_dW`, `... +2 more` |
| Imports | dataclasses, grouped_gemm, logging, torch, triton, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** High-level Python interface for grouped GEMM operations in MoE layers, providing forward and backward passes with support for permutations, fusions, and TMA (Tensor Memory Accelerator) optimizations.

**Mechanism:**
- `grouped_gemm_forward()`: Executes Y = X @ W for tokens grouped by expert assignment, with optional token permutation (expert grouping), output permutation (restore token order), and topk weight multiplication
- `grouped_gemm_dX()` / `grouped_gemm_dW()`: Backward pass kernels computing gradients w.r.t. inputs and weights
- `GroupedGemm` autograd.Function: Wraps forward/backward operations for PyTorch autodifferentiation
- `supports_tma()`: Detects GPU compute capability >= 9.0 (Hopper+) for TMA support
- Configuration validation functions ensure valid combinations of permutations and TMA loads/stores
- Handles expert-specific matrix sizes via m_sizes tensor and gather_indices for token routing
- Supports both manual kernel configuration and Triton autotuning

**Significance:** Core abstraction layer that exposes optimized grouped GEMM operations to higher-level MoE implementations. Enables efficient MoE computation by batching matrix multiplications across experts while maintaining flexibility for different MoE architectures (first vs second GEMM in MLP, with/without permutations). The interface balances performance (TMA, autotuning) with usability (PyTorch integration, configuration validation).
