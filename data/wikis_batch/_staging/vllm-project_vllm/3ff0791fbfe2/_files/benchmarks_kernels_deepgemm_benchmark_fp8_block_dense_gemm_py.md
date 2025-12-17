# File: `benchmarks/kernels/deepgemm/benchmark_fp8_block_dense_gemm.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 435 |
| Functions | `benchmark_shape`, `format_table_row`, `print_table`, `format_speedup`, `run_benchmarks` |
| Imports | time, torch, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarks DeepGEMM FP8 block GEMM

**Mechanism:** Comprehensive benchmark for DeepGEMM's FP8 block-quantized dense matrix multiplication. Tests cutlass_scaled_mm_dq and w8a8_block_fp8_matmul across various matrix shapes, block sizes (128x128, 256x256), and batch sizes. Creates random FP8 quantized weights and activations with block-specific scales. Measures execution time using CUDA synchronization. Prints formatted performance tables showing latency, throughput (TFLOPS), and speedup comparisons. Tests both CUTLASS and Triton implementations of block FP8 GEMM.

**Significance:** Validates DeepGEMM performance for block-quantized FP8 inference. Block quantization provides better accuracy than per-tensor quantization while maintaining FP8 performance. Essential for evaluating DeepGEMM competitiveness against standard FP8 GEMM implementations. Critical for models using block-wise FP8 quantization schemes. Helps determine optimal block size and validate kernel efficiency.
