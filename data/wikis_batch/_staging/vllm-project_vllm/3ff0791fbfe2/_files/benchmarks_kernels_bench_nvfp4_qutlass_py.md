# File: `benchmarks/kernels/bench_nvfp4_qutlass.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 207 |
| Functions | `get_hadamard_matrix`, `build_nvfp4_runner`, `benchmark`, `prepare_shapes` |
| Imports | argparse, compressed_tensors, copy, itertools, torch, vllm, weight_shapes |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmark NVFP4 with Hadamard transform

**Mechanism:** Similar to MXFP4 but using NVIDIA's FP4 format. Applies Hadamard transforms (16/32/64/128 sizes) via fusedQuantizeNv before CUTLASS GEMM. Tests Llama-3.3-70B shapes with extended batch range (1-32K). Uses blocked scales for per-group quantization (16-element groups).

**Significance:** Advanced 4-bit quantization benchmark combining Hadamard rotations with NVFP4. Hadamard transforms improve quantization quality by redistributing outliers. Critical for evaluating production-ready 4-bit inference on Blackwell hardware.
