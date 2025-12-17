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
Comprehensive benchmark comparing BF16 versus FP8 quantized GEMM variants.

=== Description ===
This benchmark provides detailed performance analysis of FP8 quantized matrix multiplication across different quantization configurations. It tests weight quantization granularities (per-tensor using single scale, per-channel using row-wise scales) and activation quantization granularities (per-tensor for entire batch, per-token for row-wise scales). Each configuration can run with or without dynamic quantization. The script uses real model weight shapes (default Llama-3.1-8B) across various tensor parallel sizes and batch sizes, measuring TFLOP/s to compare PyTorch and vLLM CUTLASS implementations.

=== Usage ===
Use this benchmark to understand FP8 quantization performance trade-offs, choose optimal quantization strategies for specific models, and validate that FP8 implementations provide expected speedups over BF16 baselines.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/bench_fp8_gemm.py#L1-L159 benchmarks/kernels/bench_fp8_gemm.py]
* '''Lines:''' 1-159

=== Signature ===
<syntaxhighlight lang="python">
def build_fp8_runner(cfg, a, b, dtype, device):
    """Build FP8 quantization runner based on configuration."""
    # Returns callable that quantizes and performs FP8 GEMM

@triton.testing.perf_report(...)
def benchmark(batch_size, provider, N, K):
    """Benchmark different FP8 configurations."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# This is a standalone benchmark script
python benchmarks/kernels/bench_fp8_gemm.py \
    --models meta-llama/Llama-3.1-8B-Instruct \
    --tp-sizes 1
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| --models || list[str] || No || Model names to test (default: ["meta-llama/Llama-3.1-8B-Instruct"])
|-
| --tp-sizes || list[int] || No || Tensor parallel sizes (default: [1])
|-
| batch_size || int || Internal || Batch sizes: [1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
|-
| provider || str || Internal || Configuration to test (see PROVIDER_CFGS)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| tflops || tuple[float,float,float] || TFLOP/s (median, 80th percentile, 20th percentile)
|-
| plots || files || PNG plots: bench_fp8_res_n{N}_k{K}.png
|-
| stdout || text || Tabular TFLOP/s comparison across batch sizes and providers
|}

== Usage Examples ==
<syntaxhighlight lang="python">
# Example 1: Default benchmark (Llama-3.1-8B)
python benchmarks/kernels/bench_fp8_gemm.py

# Tests these configurations (enabled by default):
# - torch-bf16: BF16 baseline
# - fp8-tensor-w-tensor-a: per-tensor weight + per-tensor activation
# - fp8-channel-w-token-a: per-channel weight + per-token activation
# - fp8-tensor-w-tensor-a-noquant: pre-quantized activation (tensor)
# - fp8-channel-w-token-a-noquant: pre-quantized activation (channel)

# Example 2: Multiple models with different TP sizes
python benchmarks/kernels/bench_fp8_gemm.py \
    --models meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.1-70B-Instruct \
    --tp-sizes 1 2 4 8

# Example 3: Programmatic usage
from benchmarks.kernels.bench_fp8_gemm import (
    build_fp8_runner,
    PROVIDER_CFGS,
)
import torch
from vllm.triton_utils import triton

M, N, K = 512, 4096, 4096
device = "cuda"
dtype = torch.bfloat16

a = torch.randn((M, K), device=device, dtype=dtype)
b = torch.randn((N, K), device=device, dtype=dtype)

# Test per-channel weight + per-token activation
cfg = PROVIDER_CFGS["fp8-channel-w-token-a"]
run_fp8 = build_fp8_runner(cfg, a, b, dtype, device)

# Benchmark
ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
    lambda: run_fp8(), quantiles=[0.5, 0.2, 0.8]
)

tflops = (2 * M * N * K) * 1e-12 / (ms * 1e-3)
print(f"FP8 channel-token: {tflops:.2f} TFLOP/s")

# Example 4: Compare all enabled configurations
for provider_name in ["torch-bf16", "fp8-tensor-w-tensor-a", "fp8-channel-w-token-a"]:
    if provider_name == "torch-bf16":
        ms, _, _ = triton.testing.do_bench_cudagraph(
            lambda: torch.nn.functional.linear(a, b),
            quantiles=[0.5, 0.2, 0.8]
        )
    else:
        cfg = PROVIDER_CFGS[provider_name]
        run_fn = build_fp8_runner(cfg, a, b, dtype, device)
        ms, _, _ = triton.testing.do_bench_cudagraph(
            lambda: run_fn(), quantiles=[0.5, 0.2, 0.8]
        )

    tflops = (2 * M * N * K) * 1e-12 / (ms * 1e-3)
    print(f"{provider_name:35s}: {tflops:6.2f} TFLOP/s ({ms:.4f} ms)")

# Example 5: Understanding the configuration space
# PROVIDER_CFGS defines:
# - w: "tensor" (single scale) or "channel" (per-row scales)
# - a: "tensor" (single scale) or "token" (per-row scales)
# - no_a_quant: False (dynamic) or True (pre-quantized)

# Typical results:
# torch-bf16:                          45.32 TFLOP/s (baseline)
# fp8-tensor-w-tensor-a:               78.45 TFLOP/s (1.73x speedup)
# fp8-channel-w-token-a:               72.18 TFLOP/s (1.59x speedup, better accuracy)
# fp8-tensor-w-tensor-a-noquant:       82.67 TFLOP/s (1.82x speedup)
# fp8-channel-w-token-a-noquant:       76.91 TFLOP/s (1.70x speedup)

# Trade-off: tensor quantization is faster but less accurate
#            channel/token quantization is slower but more accurate
#            pre-quantization (noquant) is fastest but requires static activations

# Output format:
# meta-llama/Llama-3.1-8B-Instruct, N=4096 K=4096, BF16 vs FP8 GEMMs TFLOP/s:
#
# batch_size            1      16      64     128     256     512    1024    2048    4096    8192   16384
# ---------------------------------------------------------------------------------------------------------------
# torch-bf16        X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX
# fp8-tensor-w-...  X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX
# fp8-channel-w-... X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_CUDA_Environment]]
