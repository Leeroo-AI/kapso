{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::GEMM]], [[domain::Quantization]], [[domain::INT8]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmark comparing BF16 baseline against various INT8 GEMM quantization strategies.

=== Description ===
This benchmark evaluates INT8 quantized matrix multiplication performance with different scaling approaches. Similar to the FP8 benchmark, it tests tensor-wise, channel-wise, and token-wise scaling for both weights and activations. The benchmark measures the performance of vLLM's CUTLASS INT8 kernels against PyTorch BF16/FP16 baselines with and without dynamic quantization overhead.

The benchmark sweeps batch sizes from 1 to 16384 using real transformer model layer dimensions and supports tensor parallelism. It provides insights into the tradeoff between quantization overhead and compute speedup, helping users select optimal INT8 quantization strategies for their deployment scenarios.

=== Usage ===
Run when evaluating INT8 quantization for LLM inference to determine optimal scaling granularity and quantization timing.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/bench_int8_gemm.py benchmarks/kernels/bench_int8_gemm.py]
* '''Lines:''' 1-169

=== Signature ===
<syntaxhighlight lang="python">
def build_int8_runner(
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
# Benchmark Llama-3.1-8B-Instruct
python benchmarks/kernels/bench_int8_gemm.py \
    --models meta-llama/Llama-3.1-8B-Instruct \
    --tp-sizes 1

# Benchmark multiple models and TP configs
python benchmarks/kernels/bench_int8_gemm.py \
    --models meta-llama/Llama-2-7b-hf meta-llama/Llama-3.1-8B-Instruct \
    --tp-sizes 1 2 4

# Results saved to bench_int8_res_n{N}_k{K}.png/pkl
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| batch_size || int || Number of tokens (M dimension)
|-
| provider || str || Quantization config (e.g., "int8-tensor-w-tensor-a", "int8-channel-w-token-a")
|-
| N || int || Output dimension from model layer
|-
| K || int || Input dimension from model layer
|-
| models || list[str] || Model names for layer dimensions
|-
| tp_sizes || list[int] || Tensor parallel sizes
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
| plots || .png files || Performance charts
|}

== Usage Examples ==

<syntaxhighlight lang="python">
import torch
from benchmarks.kernels.bench_int8_gemm import build_int8_runner

device = "cuda"
M, K, N = 1024, 4096, 4096
a = torch.randn((M, K), device=device, dtype=torch.bfloat16)
b = torch.randn((N, K), device=device, dtype=torch.bfloat16)

# Build tensor-wise scaling runner (dynamic quant)
cfg_tensor = {"w": "tensor", "a": "tensor", "no_a_quant": False}
runner_tensor = build_int8_runner(cfg_tensor, a, b, torch.bfloat16, device)

# Build channel-wise weight, token-wise activation (dynamic quant)
cfg_mixed = {"w": "channel", "a": "token", "no_a_quant": False}
runner_mixed = build_int8_runner(cfg_mixed, a, b, torch.bfloat16, device)

# Build with pre-quantized activations (no dynamic quant overhead)
cfg_noquant = {"w": "channel", "a": "token", "no_a_quant": True}
runner_noquant = build_int8_runner(cfg_noquant, a, b, torch.bfloat16, device)

# Execute
output_tensor = runner_tensor()
output_mixed = runner_mixed()
output_noquant = runner_noquant()

# Available providers:
# - "torch-bf16" (baseline FP16/BF16)
# - "int8-tensor-w-tensor-a" (both tensor-wise, dynamic quant)
# - "int8-channel-w-token-a" (channel weight, token activation)
# - "int8-tensor-w-tensor-a-noquant" (pre-quantized)
# - "int8-channel-w-token-a-noquant" (pre-quantized)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[uses::Library:CUTLASS]]
* [[related::Concept:INT8Quantization]]
