# File: `benchmarks/kernels/benchmark_rmsnorm.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 255 |
| Classes | `HuggingFaceRMSNorm` |
| Functions | `rmsnorm_naive`, `rmsnorm_flashinfer`, `rmsnorm_vllm`, `calculate_diff`, `get_benchmark` |
| Imports | flashinfer, itertools, torch, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Compares RMSNorm implementations (PyTorch/FlashInfer/vLLM)

**Mechanism:** Comprehensive 3-way RMSNorm benchmark using Triton's perf_report framework. Implements reference HuggingFace RMSNorm and compares against FlashInfer (fused_add_rmsnorm/rmsnorm) and vLLM (fused_add_rms_norm) implementations. Tests both standalone RMSNorm and fused add+RMSNorm variants. Benchmarks across different batch sizes (1-64), sequence lengths (64-512), and head counts (32/48). Measures performance at multiple quantiles (20%, 50%, 80%). Generates comparison plots showing relative performance across implementations.

**Significance:** Essential for validating RMSNorm kernel optimizations. RMSNorm is used in every transformer layer, making it a critical operation for overall inference performance. Benchmarking helps choose between implementations (naive PyTorch for correctness vs optimized FlashInfer/vLLM for speed). Fused add+RMSNorm significantly reduces overhead by eliminating intermediate buffers. Key component of transformer layer optimization.
