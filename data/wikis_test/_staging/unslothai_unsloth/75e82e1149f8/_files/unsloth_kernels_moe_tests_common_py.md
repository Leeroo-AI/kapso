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

**Purpose:** Provides shared test infrastructure, configuration dataclasses, and assertion utilities for MoE kernel testing.

**Mechanism:**

**Configuration classes**:
- `DataConfig`: Test data parameters (seq_len, dtype, device, batch size)
- `ModelConfig`: Model architecture (hidden_size, intermediate_size, num_experts, topk, activation type)
- `GroupedGEMMTestConfig`: Combines data and model configs with test name

**Test utilities**:
- `make_inputs`: Generates random tensors for X, W1, W2, and routing scores
- `assert_close`: Validates tensor equality with configurable tolerances, detailed error reporting
- `get_kernel_test_configs`: Generates all valid kernel config combinations (forward, dX, dW)
- `remove_feature_flags`: Filters configs to exclude specific features (permutations, TMA)

**Constants**:
- Defines small/medium model sizes for quick testing
- Llama4 and Qwen3 model configs matching actual architectures
- Tolerance thresholds by dtype

**Significance:** Core testing infrastructure that ensures consistent test configuration across all test files. The detailed assertion functions with tolerance tracking help debug numerical differences. Essential for maintaining correctness as kernels evolve.