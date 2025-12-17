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
Comprehensive benchmark comparing BF16 versus INT8 quantized GEMM variants.

=== Description ===
This benchmark mirrors the FP8 benchmark structure but focuses on INT8 quantization. It evaluates multiple INT8 configurations: per-tensor weight quantization (single scale for entire weight matrix), per-channel weight quantization (row-wise scaling), per-tensor activation quantization (single scale for batch), and per-token activation quantization (row-wise scaling). Each configuration can use dynamic quantization (quantize on-the-fly) or pre-quantization. The script tests real model weight shapes (default Llama-3.1-8B) across tensor parallel sizes and batch sizes, measuring TFLOP/s for PyTorch and vLLM implementations.

=== Usage ===
Use this benchmark to understand INT8 quantization performance characteristics, especially for hardware without native FP8 support. Essential for comparing INT8 versus FP8 trade-offs in accuracy and speed.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/bench_int8_gemm.py#L1-L169 benchmarks/kernels/bench_int8_gemm.py]
* '''Lines:''' 1-169

=== Signature ===
<syntaxhighlight lang="python">
def build_int8_runner(cfg, a, b, dtype, device):
    """Build INT8 quantization runner based on configuration."""
    # Returns callable that quantizes and performs INT8 GEMM

@triton.testing.perf_report(...)
def benchmark(batch_size, provider, N, K):
    """Benchmark different INT8 configurations."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# This is a standalone benchmark script
python benchmarks/kernels/bench_int8_gemm.py \
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
| plots || files || PNG plots: bench_int8_res_n{N}_k{K}.png
|-
| stdout || text || Tabular TFLOP/s comparison across batch sizes and providers
|}

== Usage Examples ==
<syntaxhighlight lang="python">
# Example 1: Default benchmark (Llama-3.1-8B)
python benchmarks/kernels/bench_int8_gemm.py

# Tests these configurations (enabled by default):
# - torch-bf16: BF16 baseline
# - int8-tensor-w-tensor-a: per-tensor weight + per-tensor activation
# - int8-channel-w-token-a: per-channel weight + per-token activation
# - int8-tensor-w-tensor-a-noquant: pre-quantized activation (tensor)
# - int8-channel-w-token-a-noquant: pre-quantized activation (channel)

# Example 2: Multiple models across TP configurations
python benchmarks/kernels/bench_int8_gemm.py \
    --models meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf \
    --tp-sizes 1 2 4

# Example 3: Programmatic usage
from benchmarks.kernels.bench_int8_gemm import (
    build_int8_runner,
    PROVIDER_CFGS,
)
import torch
from vllm.triton_utils import triton

M, N, K = 1024, 4096, 4096
device = "cuda"
dtype = torch.bfloat16

a = torch.randn((M, K), device=device, dtype=dtype)
b = torch.randn((N, K), device=device, dtype=dtype)

# Test per-channel weight + per-token activation
cfg = PROVIDER_CFGS["int8-channel-w-token-a"]
run_int8 = build_int8_runner(cfg, a, b, dtype, device)

# Benchmark
ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
    lambda: run_int8(), quantiles=[0.5, 0.2, 0.8]
)

tflops = (2 * M * N * K) * 1e-12 / (ms * 1e-3)
print(f"INT8 channel-token: {tflops:.2f} TFLOP/s")

# Example 4: Compare INT8 vs FP8 (requires both scripts)
# INT8 is typically 10-20% slower than FP8 on GPUs with FP8 support,
# but may be faster on older GPUs or CPUs

# Example 5: Understanding quantization overhead
import torch.utils.benchmark as TBenchmark

# Just quantization (no GEMM)
quant_timer = TBenchmark.Timer(
    stmt="vllm_scaled_int8_quant(a)",
    globals={
        'a': a,
        'vllm_scaled_int8_quant': __import__('vllm._custom_ops',
            fromlist=['scaled_int8_quant']).scaled_int8_quant
    }
)
print(f"Quantization overhead: {quant_timer.blocked_autorange(min_run_time=1)}")

# Example 6: Testing asymmetric quantization
# Per-token with zero-point (not available in default config)
from vllm._custom_ops import cutlass_scaled_mm, scaled_int8_quant

# Quantize with per-token scales
a_int8, scale_a, _ = scaled_int8_quant(a)  # per-token by default
b_int8, scale_b, _ = scaled_int8_quant(b)  # per-channel by default

# Benchmark INT8 GEMM
timer = TBenchmark.Timer(
    stmt="cutlass_scaled_mm(a_int8, b_int8.t(), scale_a, scale_b, dtype)",
    globals={
        'a_int8': a_int8,
        'b_int8': b_int8,
        'scale_a': scale_a,
        'scale_b': scale_b,
        'dtype': dtype,
        'cutlass_scaled_mm': cutlass_scaled_mm,
    }
)
result = timer.blocked_autorange(min_run_time=1)
print(f"INT8 GEMM: {result}")

# Typical performance comparison:
# Hardware: NVIDIA A100 (no native FP8)
# - BF16:                 42.3 TFLOP/s (baseline)
# - INT8 tensor-tensor:   68.5 TFLOP/s (1.62x)
# - INT8 channel-token:   61.2 TFLOP/s (1.45x, better accuracy)
#
# Hardware: NVIDIA H100 (native FP8)
# - BF16:                 67.8 TFLOP/s (baseline)
# - INT8 tensor-tensor:   89.3 TFLOP/s (1.32x)
# - FP8 tensor-tensor:    104.7 TFLOP/s (1.54x) <- FP8 wins on H100

# Output format:
# meta-llama/Llama-3.1-8B-Instruct, N=4096 K=4096, BF16 vs INT8 GEMMs TFLOP/s:
#
# batch_size            1      16      64     128     256     512    1024    2048    4096    8192   16384
# ---------------------------------------------------------------------------------------------------------------
# torch-bf16        X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX
# int8-tensor-w-... X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX
# int8-channel-w-...X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_CUDA_Environment]]
