# File: `benchmarks/kernels/benchmark_fused_collective.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1129 |
| Classes | `FlashInferFusedAllReduceParams`, `VllmFusedAllreduce` |
| Functions | `setup_flashinfer_workspace`, `cleanup_flashinfer_workspace`, `flashinfer_fused_allreduce_rmsnorm`, `flashinfer_fused_allreduce_rmsnorm_fp8_quant`, `flashinfer_fused_allreduce_rmsnorm_fp4_quant`, `create_test_tensors`, `benchmark_operation`, `run_benchmarks`, `... +5 more` |
| Imports | argparse, itertools, os, pandas, time, torch, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarks FlashInfer fused allreduce+rmsnorm+quantization operations

**Mechanism:** Compares FlashInfer's trtllm_allreduce_fusion (fused allreduce+rmsnorm with optional FP8/FP4 quantization) against standard tensor_model_parallel_all_reduce followed by separate rmsnorm/quant operations. Tests multiple configurations: oneshot vs twoshot modes, with/without residual connections, custom vs native ops, and torch.compile variants. Uses CUDA graphs for precise timing, supports multiple token counts and hidden dimensions, outputs results as markdown tables with speedup calculations.

**Significance:** Essential for evaluating fused collective operation performance in tensor parallel setups. Validates that kernel fusion can significantly reduce overhead vs sequential operations. Critical for optimizing distributed transformer inference, particularly for models using RMSNorm and quantization. Helps justify using FlashInfer's optimized kernels over standard implementations.
