{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Kernel Benchmarking]], [[domain::Quantization]], [[domain::Activation Functions]], [[domain::MoE]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
A comprehensive benchmark suite comparing SiLU activation with FP8 quantization implementations for Mixture-of-Experts (MoE) models.

=== Description ===
This benchmark performs a 3-way comparison of fused SiLU-multiply-FP8 quantization kernels, critical for efficient MoE inference. It tests SiLU V2 (CUDA), Triton, and generates detailed performance metrics including memory bandwidth utilization, speedup ratios, and TFLOPS across various expert configurations, token distributions, and batch sizes. The benchmark supports multiple token distribution strategies (uniform, imbalanced random, even assignment) to simulate real-world MoE workloads. It uses group-wise FP8 quantization with configurable block sizes (typically 128x128).

=== Usage ===
Use this benchmark to select optimal SiLU-multiply-quant kernels for MoE models, tune kernel configurations for specific hardware (especially DeepSeek-V3 architectures), or validate custom kernel implementations against production baselines.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_silu_mul_fp8_quant.py benchmarks/kernels/benchmark_silu_mul_fp8_quant.py]

=== Signature ===
<syntaxhighlight lang="python">
def silu_mul_fp8_quant_deep_gemm_triton(
    y: torch.Tensor,  # (E, T, 2*H)
    tokens_per_expert: torch.Tensor,  # (E,)
    num_parallel_tokens: int,
    group_size: int = 128,
    eps: float = 1e-10,
    expert_offsets: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor]

def benchmark(kernel: Callable, E: int, T: int, H: int,
              total_tokens: int, num_parallel_tokens: int = 64,
              G: int = 128, runs: int = 200,
              gen_strategy: str = "default") -> tuple
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Run as script
python benchmarks/kernels/benchmark_silu_mul_fp8_quant.py
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| y || torch.Tensor || Input activations, shape (E, T, 2*H) - first H for gate, second H for up projection
|-
| tokens_per_expert || torch.Tensor || Number of valid tokens per expert, shape (E,)
|-
| num_parallel_tokens || int || Parallelism parameter for kernel
|-
| group_size || int || Group size for FP8 quantization (default 128)
|-
| E || int || Number of experts
|-
| T || int || Max tokens per expert
|-
| H || int || Hidden dimension (7168 for DeepSeek-V3)
|-
| total_tokens || int || Total active tokens across all experts
|-
| gen_strategy || str || Token distribution: "uniform", "random_imbalanced", "max_t", "first_t"
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| y_q || torch.Tensor || FP8 quantized activations, shape (E, T, H)
|-
| y_s || torch.Tensor || Per-group quantization scales, shape (E, T, H//group_size)
|-
| benchmark_plots || File || Performance comparison plots (speedup and bandwidth utilization)
|-
| timing_metrics || dict || Median time (ms), TFLOPS, GB/s, bandwidth %
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Run comprehensive benchmark with all strategies
python benchmarks/kernels/benchmark_silu_mul_fp8_quant.py

# Use the Triton kernel directly
from benchmark_silu_mul_fp8_quant import silu_mul_fp8_quant_deep_gemm_triton
import torch

E, T, H = 256, 1024, 7168  # DeepSeek-V3 config
y = torch.rand((E, T, 2*H), dtype=torch.bfloat16, device="cuda")
tokens_per_expert = torch.randint(1, T, (E,), dtype=torch.int32, device="cuda")

y_q, y_s = silu_mul_fp8_quant_deep_gemm_triton(
    y, tokens_per_expert,
    num_parallel_tokens=64,
    group_size=128
)

# Results: y_q is FP8 (E, T, H), y_s is scales (E, T, H//128)
print(f"Quantized output: {y_q.shape}, dtype: {y_q.dtype}")
print(f"Scales: {y_s.shape}, dtype: {y_s.dtype}")
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[Implementation:DeepGEMM_MoE]]
* [[Concept:FP8_Quantization]]
* [[Concept:Mixture_of_Experts]]
* [[Benchmark:MoE_Kernel_Performance]]
