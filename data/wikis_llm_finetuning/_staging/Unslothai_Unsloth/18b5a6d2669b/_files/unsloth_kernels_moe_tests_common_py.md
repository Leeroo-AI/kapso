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

**Purpose:** Shared test infrastructure: configurations, fixtures, and assertion utilities for MoE kernel testing.

**Mechanism:** Defines DataConfig and ModelConfig dataclasses encapsulating test parameters (sequence lengths, dtypes, model dimensions, expert counts, topk). Provides SMALL_MODEL_CONFIGS, LLAMA_MODEL_CONFIG, QWEN_MODEL_CONFIG test suites. Implements make_inputs() for random test data generation. Contains assert_close() for numerical comparison with adaptive tolerances, handling infinities and computing max/RMS relative errors. get_kernel_test_configs() generates comprehensive kernel configuration matrices with pruning.

**Significance:** Centralized test infrastructure that ensures consistency across all MoE kernel tests. The assert_close() implementation is particularly important for catching numerical issues in different dtypes (bf16/fp16/fp32). The configuration generators enable exhaustive testing of kernel variants (permutations, TMA, block sizes) without code duplication.
