# File: `unsloth/kernels/moe/tests/common.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 336 |
| Classes | `DataConfig`, `ModelConfig`, `GroupedGEMMTestConfig` |
| Functions | `print_delimiter`, `delimiter_context`, `make_inputs`, `assert_equal`, `assert_close`, `assert_indx_equal`, `get_kernel_test_configs`, `remove_feature_flags` |
| Imports | contextlib, dataclasses, grouped_gemm, itertools, torch |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides shared test infrastructure including configuration classes, input generation, assertion utilities, and test parameter generation for MoE grouped GEMM kernel tests.

**Mechanism:** Implements comprehensive test support:
- Configuration classes: `DataConfig` (sequence length, dtype), `ModelConfig` (hidden size, intermediate size, experts, topk, activation), `GroupedGEMMTestConfig` (combines data and model configs)
- Input generation: `make_inputs()` creates random tensors (X1, X2, W1, W2, scores) with proper shapes for first/second GEMM testing
- Assertion utilities: `assert_equal()` for exact matches, `assert_close()` for numerical tolerance with detailed error reporting (max/RMS relative error), `assert_indx_equal()` for index comparisons
- Test configuration generation: `get_kernel_test_configs()` generates all valid combinations of kernel parameters (block sizes, warps, stages, TMA flags, permutation flags), with pruning functions to remove invalid combinations
- Predefined test configs: Small model sizes (32x32 to 512x512), Llama/Qwen model configs, sequence lengths (128, 1024), and kernel configurations

Tolerances are defined per dtype: bfloat16 (1e-3), float16 (1e-4), float32 (1e-5).

**Significance:** Central test infrastructure that standardizes testing across all grouped GEMM kernel tests. Ensures consistent configuration, input generation, and validation across forward/backward passes with various permutation and TMA settings.
