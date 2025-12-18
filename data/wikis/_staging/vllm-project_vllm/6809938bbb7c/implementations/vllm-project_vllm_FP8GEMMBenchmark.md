{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::GEMM]], [[domain::Quantization]], [[domain::FP8]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmark comparing BF16 baseline against various FP8 GEMM quantization strategies.

=== Description ===
This benchmark evaluates FP8 quantized matrix multiplication performance across different scaling granularities and quantization strategies. It compares tensor-wise scaling (single scale per tensor), channel-wise scaling (per-output-channel scales), and token-wise scaling (per-input-row scales) for both weights and activations. The benchmark also tests whether activations are pre-quantized or dynamically quantized during the operation.

The benchmark measures TFLOP/s across batch sizes from 1 to 16384 using real model layer dimensions from transformers like Llama-3.1. It compares vLLM's CUTLASS FP8 kernels against PyTorch native implementations and supports tensor parallelism configurations. Results help determine optimal quantization strategies for different workload characteristics.

=== Usage ===
Run when evaluating FP8 quantization schemes for LLM inference, particularly for selecting scaling granularity and quantization approach.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/bench_fp8_gemm.py benchmarks/kernels/bench_fp8_gemm.py]
* '''Lines:''' 1-159

=== Signature ===
<syntaxhighlight lang="python">
def build_fp8_runner(
    cfg: dict,
    a: torch.Tensor,
    b: torch.Tensor,
    dtype: torch.dtype,
    device: str
) -> Callable

@triton.testing.perf_report(...)
def benchmark(
    batch_size: int,
    provider: str,
    N: int,
    K: int
)
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Benchmark Llama-3.1-8B-Instruct with default settings
python benchmarks/kernels/bench_fp8_gemm.py \
    --models meta-llama/Llama-3.1-8B-Instruct \
    --tp-sizes 1

# Benchmark multiple models with tensor parallelism
python benchmarks/kernels/bench_fp8_gemm.py \
    --models meta-llama/Llama-2-7b-hf meta-llama/Llama-3.1-8B-Instruct \
    --tp-sizes 1 2 4

# Results saved to bench_fp8_res_n{N}_k{K}.png/pkl
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| batch_size || int || Number of tokens (M dimension)
|-
| provider || str || Quantization strategy (e.g., "fp8-tensor-w-tensor-a", "fp8-channel-w-token-a")
|-
| N || int || Output dimension from model layer
|-
| K || int || Input dimension from model layer
|-
| models || list[str] || Model names to extract layer dimensions
|-
| tp_sizes || list[int] || Tensor parallel sizes to test
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
|-
| plots || .png files || Performance visualization plots
|}

== Usage Examples ==

<syntaxhighlight lang="python">
import torch
from benchmarks.kernels.bench_fp8_gemm import build_fp8_runner

device = "cuda"
M, K, N = 1024, 4096, 4096
a = torch.randn((M, K), device=device, dtype=torch.bfloat16)
b = torch.randn((N, K), device=device, dtype=torch.bfloat16)

# Build tensor-wise scaling runner
cfg_tensor = {"w": "tensor", "a": "tensor", "no_a_quant": False}
runner_tensor = build_fp8_runner(cfg_tensor, a, b, torch.bfloat16, device)

# Build channel-wise weight, token-wise activation runner
cfg_mixed = {"w": "channel", "a": "token", "no_a_quant": False}
runner_mixed = build_fp8_runner(cfg_mixed, a, b, torch.bfloat16, device)

# Execute
output_tensor = runner_tensor()
output_mixed = runner_mixed()

# Provider configurations available:
# - "torch-bf16" (baseline)
# - "fp8-tensor-w-tensor-a" (both tensor-wise)
# - "fp8-channel-w-token-a" (channel weight, token activation)
# - "fp8-tensor-w-tensor-a-noquant" (pre-quantized activation)
# - "fp8-channel-w-token-a-noquant" (pre-quantized activation)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[uses::Library:CUTLASS]]
* [[related::Concept:FP8Quantization]]
