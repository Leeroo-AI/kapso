{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Kernel Benchmarking]], [[domain::GEMM]], [[domain::Quantization]], [[domain::DeepGEMM]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
A three-way benchmark comparing DeepGEMM, vLLM Triton, and vLLM CUTLASS implementations for FP8 block-wise quantized matrix multiplication.

=== Description ===
This comprehensive benchmark evaluates three FP8 GEMM implementations (DeepGEMM, vLLM Triton, vLLM CUTLASS) across realistic model weight shapes, primarily targeting DeepSeek-V3 architectures. It measures execution time, TFLOPS, memory bandwidth utilization, and speedup ratios for block-wise FP8 quantized matrix multiplication with 128x128 block sizes. The benchmark pre-quantizes weights and activations to FP8 with per-block scales, then compares performance across various M (batch), N (output), K (hidden) dimensions. Results include detailed accuracy comparisons against BF16 reference and cross-implementation validation.

=== Usage ===
Use this benchmark to select the best FP8 GEMM kernel for your hardware, validate custom FP8 implementations, or optimize inference performance for DeepSeek-V3 and similar large models with block-wise quantization.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/deepgemm/benchmark_fp8_block_dense_gemm.py benchmarks/kernels/deepgemm/benchmark_fp8_block_dense_gemm.py]

=== Signature ===
<syntaxhighlight lang="python">
def benchmark_shape(m: int, n: int, k: int,
                    warmup: int = 100, repeat: int = 10000,
                    verbose: bool = False) -> dict

def run_benchmarks(verbose: bool = False) -> None
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Run as script
python benchmarks/kernels/deepgemm/benchmark_fp8_block_dense_gemm.py
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| m || int || Batch dimension (number of tokens)
|-
| n || int || Output dimension (weight rows)
|-
| k || int || Hidden dimension (weight columns)
|-
| warmup || int || Warmup iterations (default 100)
|-
| repeat || int || Benchmark iterations (default 10000)
|-
| verbose || bool || Print detailed timing per shape
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| benchmark_results || dict || Per-implementation timing, TFLOPS, GB/s, speedups
|-
| accuracy_diffs || dict || Numerical differences vs reference and between implementations
|-
| summary_tables || stdout || Formatted performance comparison tables
|-
| average_metrics || dict || Mean TFLOPS, bandwidth, speedup across all shapes
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Run full benchmark suite
python benchmarks/kernels/deepgemm/benchmark_fp8_block_dense_gemm.py

# Benchmark specific shape with verbose output
from benchmark_fp8_block_dense_gemm import benchmark_shape

result = benchmark_shape(
    m=64, n=7168, k=18432,
    warmup=100, repeat=10000,
    verbose=True
)

print(f"DeepGEMM: {result['implementations']['DeepGEMM']['tflops']:.1f} TFLOPS")
print(f"vLLM CUTLASS speedup vs DeepGEMM: "
      f"{result['implementations']['vLLM CUTLASS']['speedup_vs_deepgemm']:.2f}x")

# Use individual implementations
from vllm.utils.deep_gemm import fp8_gemm_nt, per_block_cast_to_fp8
import torch

A = torch.randn((64, 7168), dtype=torch.bfloat16, device="cuda")
B = torch.randn((18432, 7168), dtype=torch.bfloat16, device="cuda")

# Quantize to FP8 with block-wise scales
A_fp8, A_scale = per_block_cast_to_fp8(A, [128, 128])
B_fp8, B_scale = per_block_cast_to_fp8(B, [128, 128])

C = torch.empty((64, 18432), dtype=torch.bfloat16, device="cuda")
fp8_gemm_nt((A_fp8, A_scale), (B_fp8, B_scale), C)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[Implementation:DeepGEMM_Kernels]]
* [[Implementation:CUTLASS_Integration]]
* [[Concept:Block_Wise_FP8]]
* [[Benchmark:GEMM_Performance]]
