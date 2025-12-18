{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::GEMM]], [[domain::Quantization]], [[domain::Kernels]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmark comparing BF16 and W8A8 block FP8 GEMM kernels for DeepSeek-V3 layer shapes.

=== Description ===
This benchmark evaluates blockwise FP8 quantized GEMM operations with 128x128 block scaling for weight-activation matrix multiplications. It compares three implementations: native PyTorch BF16 matmul, Triton-based block FP8 GEMM, and CUTLASS-based block FP8 GEMM. The W8A8BlockFp8LinearOp performs per-token per-group quantization on activations and uses pre-quantized weights with block-granular scaling factors.

The benchmark specifically targets DeepSeek-V3 model layer dimensions and measures TFLOP/s across different batch sizes (1 to 16384). It disables DeepGEMM to use CUTLASS implementations and requires CUDA platform support. Results demonstrate the performance tradeoffs between precision (BF16) and quantized compute with various kernel backends.

=== Usage ===
Run when evaluating blockwise FP8 quantization performance for large language models, particularly for DeepSeek-V3 architecture optimization.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/bench_block_fp8_gemm.py benchmarks/kernels/bench_block_fp8_gemm.py]
* '''Lines:''' 1-160

=== Signature ===
<syntaxhighlight lang="python">
def build_w8a8_block_fp8_runner(
    M: int,
    N: int,
    K: int,
    block_size: tuple[int, int],
    device: str,
    use_cutlass: bool
) -> Callable

@triton.testing.perf_report(...)
def benchmark_tflops(
    batch_size: int,
    provider: str,
    N: int,
    K: int,
    block_size: tuple[int, int] = (128, 128)
)
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Run benchmark with default DeepSeek-V3 shapes
python benchmarks/kernels/bench_block_fp8_gemm.py

# Benchmark iterates through:
# - (512+64, 7168)
# - (2112, 7168)
# - ((128+64)*128, 7168)
# - (128*(128+128), 512)
# - (7168, 16384)
# - ... and more DeepSeek-V3 layer shapes

# Results show TFLOP/s for:
# - torch-bf16 (baseline)
# - w8a8-block-fp8-triton
# - w8a8-block-fp8-cutlass (if CUTLASS_BLOCK_FP8_SUPPORTED)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| batch_size || int || Number of tokens (M dimension)
|-
| N || int || Output feature dimension
|-
| K || int || Input feature dimension
|-
| block_size || tuple[int, int] || Block dimensions for scaling (default: 128x128)
|-
| provider || str || Backend ("torch-bf16", "w8a8-block-fp8-triton", "w8a8-block-fp8-cutlass")
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| tflops || float || TeraFLOPS per second (median)
|-
| tflops_min || float || TeraFLOPS per second (20th percentile)
|-
| tflops_max || float || TeraFLOPS per second (80th percentile)
|}

== Usage Examples ==

<syntaxhighlight lang="python">
import torch
from benchmarks.kernels.bench_block_fp8_gemm import (
    build_w8a8_block_fp8_runner,
    DEEPSEEK_V3_SHAPES
)

# Build Triton runner for specific shape
M, N, K = 2048, 7168, 16384
block_size = (128, 128)

triton_runner = build_w8a8_block_fp8_runner(
    M=M, N=N, K=K,
    block_size=block_size,
    device="cuda",
    use_cutlass=False
)

# Execute benchmark
output = triton_runner()  # Returns quantized matmul result

# Compare against CUTLASS
cutlass_runner = build_w8a8_block_fp8_runner(
    M=M, N=N, K=K,
    block_size=block_size,
    device="cuda",
    use_cutlass=True
)
output_cutlass = cutlass_runner()

# Benchmark all DeepSeek-V3 shapes
for N, K in DEEPSEEK_V3_SHAPES:
    print(f"Benchmarking N={N}, K={K}")
    # benchmark_tflops.run(...) handles batch size sweep
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[uses::Library:CUTLASS]]
* [[related::Concept:BlockwiseQuantization]]
