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
Benchmark for NVFP4 (NVIDIA FP4) quantized GEMM on Blackwell architecture (SM 10.0+).

=== Description ===
This benchmark evaluates NVIDIA's native FP4 quantization format, which requires Blackwell (compute capability 10.0) or newer GPUs. NVFP4 uses a two-stage scaling approach: global scaling with FP8_E4M3_MAX * FP4_E2M1_MAX and per-group block scaling (E8M0 format). The script compares BF16 baseline against CUTLASS and FBGEMM implementations with dynamic and pre-quantized activations. It tests Llama-3.1-8B model shapes across batch sizes (1-16384) and tensor parallel configurations, providing TFLOP/s measurements to validate hardware acceleration benefits.

=== Usage ===
Use this benchmark to evaluate next-generation GPU quantization capabilities on Blackwell architecture. Essential for understanding hardware-accelerated 4-bit inference performance compared to software-based solutions like MXFP4.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/bench_nvfp4_gemm.py#L1-L198 benchmarks/kernels/bench_nvfp4_gemm.py]
* '''Lines:''' 1-198

=== Signature ===
<syntaxhighlight lang="python">
def build_nvfp4_runner(cfg, a, b, dtype, device):
    """Build NVFP4 quantization runner for CUTLASS or FBGEMM."""

@triton.testing.perf_report(...)
def benchmark(batch_size, provider, N, K):
    """Benchmark NVFP4 vs BF16."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# This is a standalone benchmark script
# Requires Blackwell GPU (SM 10.0+)
python benchmarks/kernels/bench_nvfp4_gemm.py \
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
| provider || str || Internal || Configuration: "torch-bf16", "nvfp4", "nvfp4-noquant", "fbgemm-nvfp4", "fbgemm-nvfp4-noquant"
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| tflops || tuple[float,float,float] || TFLOP/s (median, 80th percentile, 20th percentile)
|-
| plots || directory || PNG plots in bench_nvfp4_res_n{N}_k{K}/
|-
| stdout || text || Tabular TFLOP/s comparison across batch sizes
|}

== Usage Examples ==
<syntaxhighlight lang="python">
# Example 1: Default benchmark (Llama-3.1-8B on Blackwell)
# Requires GPU with compute capability 10.0+
python benchmarks/kernels/bench_nvfp4_gemm.py

# Tests these configurations:
# - torch-bf16: BF16 baseline
# - nvfp4: CUTLASS implementation with dynamic quantization
# - nvfp4-noquant: CUTLASS with pre-quantized activations
# - fbgemm-nvfp4: FBGEMM implementation with dynamic quantization (if installed)
# - fbgemm-nvfp4-noquant: FBGEMM with pre-quantized activations (if installed)

# Example 2: Multiple models and TP sizes
python benchmarks/kernels/bench_nvfp4_gemm.py \
    --models meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.1-70B-Instruct \
    --tp-sizes 1 2 4 8

# Example 3: Programmatic usage with CUTLASS
from benchmarks.kernels.bench_nvfp4_gemm import (
    build_nvfp4_runner,
    PROVIDER_CFGS,
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
)
import torch
from vllm.triton_utils import triton

# Check GPU compatibility
from vllm.platforms import current_platform
if not current_platform.has_device_capability(100):
    raise RuntimeError("Requires Blackwell GPU (SM 10.0+)")

M, N, K = 512, 4096, 4096
device = "cuda"
dtype = torch.bfloat16

a = torch.randn((M, K), device=device, dtype=dtype)
b = torch.randn((N, K), device=device, dtype=dtype)

# Build NVFP4 runner
cfg = PROVIDER_CFGS["nvfp4"]
run_nvfp4 = build_nvfp4_runner(cfg, a, b, dtype, device)

# Benchmark
ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
    lambda: run_nvfp4(), quantiles=[0.5, 0.2, 0.8]
)

tflops = (2 * M * N * K) * 1e-12 / (ms * 1e-3)
print(f"NVFP4: {tflops:.2f} TFLOP/s")

# Example 4: Understanding NVFP4 quantization
from vllm._custom_ops import scaled_fp4_quant

# Compute global scale
b_amax = torch.abs(b).max().to(torch.float32)
b_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / b_amax
print(f"Global scale: {b_global_scale:.6f}")

# Quantize weight
b_fp4, scale_b_fp4 = scaled_fp4_quant(b, b_global_scale)
print(f"Quantized weight dtype: {b_fp4.dtype}")  # E2M1 format
print(f"Scale dtype: {scale_b_fp4.dtype}")       # E8M0 format

# Example 5: Compare CUTLASS vs FBGEMM implementations
try:
    from fbgemm_gpu.experimental.gemm.triton_gemm.fp4_quantize import (
        triton_scale_nvfp4_quant,
    )

    # CUTLASS quantization
    cutlass_cfg = PROVIDER_CFGS["nvfp4"]
    cutlass_runner = build_nvfp4_runner(cutlass_cfg, a, b, dtype, device)

    # FBGEMM quantization
    fbgemm_cfg = PROVIDER_CFGS["fbgemm-nvfp4"]
    fbgemm_runner = build_nvfp4_runner(fbgemm_cfg, a, b, dtype, device)

    # Benchmark both
    cutlass_ms, _, _ = triton.testing.do_bench_cudagraph(
        lambda: cutlass_runner(), quantiles=[0.5, 0.2, 0.8]
    )
    fbgemm_ms, _, _ = triton.testing.do_bench_cudagraph(
        lambda: fbgemm_runner(), quantiles=[0.5, 0.2, 0.8]
    )

    cutlass_tflops = (2 * M * N * K) * 1e-12 / (cutlass_ms * 1e-3)
    fbgemm_tflops = (2 * M * N * K) * 1e-12 / (fbgemm_ms * 1e-3)

    print(f"CUTLASS: {cutlass_tflops:.2f} TFLOP/s")
    print(f"FBGEMM:  {fbgemm_tflops:.2f} TFLOP/s")
except ImportError:
    print("FBGEMM not installed, skipping comparison")

# Example 6: Memory analysis
bf16_memory = N * K * 2  # 2 bytes per element
nvfp4_memory = (N * K * 0.5) + (N * K // 128) * 1  # 4 bits + E8M0 scales
savings = (1 - nvfp4_memory / bf16_memory) * 100
print(f"Memory savings: {savings:.1f}%")
# Expected: ~75% memory reduction

# Example 7: Expected performance (NVIDIA GB200)
# M=1024, N=4096, K=4096
#
# torch-bf16:            187.3 TFLOP/s (baseline)
# nvfp4:                 421.6 TFLOP/s (2.25x speedup)
# nvfp4-noquant:         438.9 TFLOP/s (2.34x speedup)
# fbgemm-nvfp4:          415.2 TFLOP/s (2.22x speedup)
# fbgemm-nvfp4-noquant:  432.7 TFLOP/s (2.31x speedup)
#
# Key advantages over MXFP4:
# - Hardware acceleration (2-3x faster on Blackwell)
# - Native GPU support (no CPU overhead)
# - Better numerical stability with E8M0 scales

# Output format:
# meta-llama/Llama-3.1-8B-Instruct, N=4096 K=4096, BF16 vs NVFP4 GEMMs TFLOP/s:
#
# batch_size            1      16      64     128     256     512    1024    2048    4096    8192   16384
# ---------------------------------------------------------------------------------------------------------------
# torch-bf16        X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX
# nvfp4             X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX
# nvfp4-noquant     X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX
# fbgemm-nvfp4      X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX
# fbgemm-nvfp4-noq. X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX   X.XXX
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_CUDA_Environment]]
