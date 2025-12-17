{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::Quantization]], [[domain::GEMM]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==
Benchmark for block-quantized FP8 GEMM operations (W8A8 with 128x128 block size).

=== Description ===
This benchmark evaluates block-quantized FP8 matrix multiplication performance, specifically designed for DeepSeek-V3 model shapes. Block quantization divides tensors into fixed-size blocks (128x128) and applies separate scaling factors to each block, providing finer granularity than per-tensor quantization while maintaining efficiency. The script compares BF16 baseline against Triton and CUTLASS implementations, measuring TFLOP/s across batch sizes from 1 to 16384. It explicitly disables DeepGEMM to isolate CUTLASS performance characteristics.

=== Usage ===
Use this benchmark to validate block-wise FP8 quantization performance for models using grouped quantization schemes like DeepSeek-V3. Essential for understanding the trade-offs between quantization granularity and computational efficiency.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/bench_block_fp8_gemm.py#L1-L160 benchmarks/kernels/bench_block_fp8_gemm.py]
* '''Lines:''' 1-160

=== Signature ===
<syntaxhighlight lang="python">
def build_w8a8_block_fp8_runner(M, N, K, block_size, device, use_cutlass):
    """Build runner function for w8a8 block fp8 matmul."""
    # Returns a callable that performs block-quantized FP8 GEMM

@triton.testing.perf_report(...)
def benchmark_tflops(batch_size, provider, N, K, block_size=(128, 128)):
    """Benchmark TFLOP/s for different providers."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# This is a standalone benchmark script
# Must be run on CUDA devices
python benchmarks/kernels/bench_block_fp8_gemm.py
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| DEEPSEEK_V3_SHAPES || list[tuple[int,int]] || Internal || Pre-defined N, K shapes from DeepSeek-V3 model layers
|-
| batch_size || int || Internal || M dimension (batch/sequence length): [1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
|-
| block_size || tuple[int,int] || Internal || Block dimensions for quantization (fixed at 128x128)
|-
| provider || str || Internal || Implementation: "torch-bf16", "w8a8-block-fp8-triton", "w8a8-block-fp8-cutlass"
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| tflops || tuple[float,float,float] || TFLOP/s (median, 80th percentile, 20th percentile)
|-
| plots || files || PNG plots comparing providers across batch sizes
|-
| stdout || text || Tabular performance comparison
|}

== Usage Examples ==
<syntaxhighlight lang="python">
# Example 1: Run full benchmark on all DeepSeek-V3 shapes
# Requires CUDA device
python benchmarks/kernels/bench_block_fp8_gemm.py

# Output for each shape:
# Benchmarking DeepSeek-V3, N=7168 K=16384
# TFLOP/s comparison (block_size=(128, 128)):
#
# batch_size        1      16      64     128     256     512    1024    2048    4096    8192   16384
# ---------------------------------------------------------------------------------------------------------------
# torch-bf16    X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX
# w8a8-triton   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX
# w8a8-cutlass  X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX

# Example 2: Programmatic usage
from benchmarks.kernels.bench_block_fp8_gemm import (
    build_w8a8_block_fp8_runner,
    DEEPSEEK_V3_SHAPES,
)
import torch
from vllm.triton_utils import triton

# Test a specific configuration
M, N, K = 512, 7168, 16384
block_size = (128, 128)

# Build runner for CUTLASS
run_cutlass = build_w8a8_block_fp8_runner(
    M, N, K, block_size, "cuda", use_cutlass=True
)

# Benchmark with Triton's do_bench_cudagraph
ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
    lambda: run_cutlass(), quantiles=[0.5, 0.2, 0.8]
)

tflops = (2 * M * N * K) * 1e-12 / (ms * 1e-3)
print(f"CUTLASS Block FP8: {tflops:.2f} TFLOP/s")

# Example 3: Compare Triton vs CUTLASS
run_triton = build_w8a8_block_fp8_runner(
    M, N, K, block_size, "cuda", use_cutlass=False
)

triton_ms, _, _ = triton.testing.do_bench_cudagraph(
    lambda: run_triton(), quantiles=[0.5, 0.2, 0.8]
)
cutlass_ms, _, _ = triton.testing.do_bench_cudagraph(
    lambda: run_cutlass(), quantiles=[0.5, 0.2, 0.8]
)

triton_tflops = (2 * M * N * K) * 1e-12 / (triton_ms * 1e-3)
cutlass_tflops = (2 * M * N * K) * 1e-12 / (cutlass_ms * 1e-3)

print(f"Triton: {triton_tflops:.2f} TFLOP/s")
print(f"CUTLASS: {cutlass_tflops:.2f} TFLOP/s")
print(f"CUTLASS speedup: {cutlass_tflops/triton_tflops:.2f}x")

# Example 4: Verify correctness
# The runner includes built-in weight quantization and scale generation
output = run_cutlass()
print(f"Output shape: {output.shape}")  # Should be (M, N)
print(f"Output dtype: {output.dtype}")   # Should be bfloat16

# DeepSeek-V3 layer shapes tested:
shapes_tested = [
    (576, 7168),      # Small projection
    (2112, 7168),     # Medium projection
    (24576, 7168),    # Large projection (MoE)
    (24576, 512),     # Expert to hidden
    (7168, 16384),    # FFN expansion
    (7168, 18432),    # Larger FFN
    (36864, 7168),    # FFN contraction
    (24576, 1536),    # Router/gating
    (12288, 7168),    # Half expansion
    (4096, 7168),     # Attention projection
    (7168, 2048),     # Down projection
]
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_CUDA_Environment]]
