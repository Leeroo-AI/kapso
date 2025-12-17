# File: `benchmarks/kernels/benchmark_mla_k_concat.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 150 |
| Functions | `cat_method`, `direct_copy_method`, `benchmark_method`, `run_benchmark`, `main` |
| Imports | collections, time, torch |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarks k_nope/k_pe concatenation methods for MLA

**Mechanism:** Compares torch.cat with expand vs direct copy for concatenating k_nope and k_pe tensors in Multi-head Latent Attention (MLA) prefill. Tests both methods across batch sizes 32-65536 for DeepSeek-V3 dimensions (128 heads, 128 nope dim, 64 pe dim). Runs warmup iterations followed by timed benchmarks using CUDA synchronization. Reports timing, speedup, and performance reduction percentages. Tests both bfloat16 and float8_e4m3fn dtypes.

**Significance:** Validates the optimization from commit 8d4142bd that replaces torch.cat with direct copy. Demonstrates performance benefits across various batch sizes, particularly for large batches typical in prefill. Essential for MLA performance optimization in DeepSeek-V3 and similar architectures. Shows that eliminating expand+cat overhead provides consistent speedup for tensor concatenation.
