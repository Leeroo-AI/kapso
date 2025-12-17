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

**Purpose:** Common test configurations, data structures, and utility functions for grouped GEMM testing.

**Mechanism:** Defines test configs (DataConfig, ModelConfig), input generation, tolerance thresholds by dtype, kernel test config generation with pruning, assertion helpers.

**Significance:** Shared infrastructure for all MoE test suites - ensures consistent test setup and validation.
