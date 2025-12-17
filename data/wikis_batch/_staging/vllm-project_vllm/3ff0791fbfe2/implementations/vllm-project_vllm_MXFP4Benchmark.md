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
Benchmark for MXFP4 (Microscaling FP4) quantized GEMM using CUTLASS kernels with Hadamard transforms.

=== Description ===
This benchmark evaluates extreme 4-bit quantization using the MX (Microscaling) format, which combines block-level exponents (E8M0) with low-precision mantissas (E2M1) for better accuracy than pure INT4. The implementation uses Hadamard transformations (sizes 32, 64, or 128) before quantization to spread information across elements, improving quantization quality. The script compares BF16 baseline against MXFP4 with dynamic and pre-quantized activations, testing Llama-3.3-70B model shapes across an extended batch size range (1-32768) to stress-test extreme quantization performance.

=== Usage ===
Use this benchmark to evaluate 4-bit inference feasibility and understand the performance characteristics of MXFP4 quantization. Critical for scenarios requiring maximum memory savings with acceptable accuracy degradation.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/bench_mxfp4_qutlass.py#L1-L191 benchmarks/kernels/bench_mxfp4_qutlass.py]
* '''Lines:''' 1-191

=== Signature ===
<syntaxhighlight lang="python">
def get_hadamard_matrix(group_size: int, dtype: torch.dtype, device: torch.device):
    """Generate deterministic Hadamard matrix for quantization."""

def build_mxfp4_runner(cfg, a, b, forward_hadamard_matrix, dtype, device):
    """Build MXFP4 quantization runner."""

@triton.testing.perf_report(...)
def benchmark(batch_size, provider, N, K, had_size):
    """Benchmark MXFP4 vs BF16."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# This is a standalone benchmark script
python benchmarks/kernels/bench_mxfp4_qutlass.py \
    --models meta-llama/Llama-3.3-70B-Instruct \
    --tp-sizes 1
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| --models || list[str] || No || Model names to test (default: ["meta-llama/Llama-3.3-70B-Instruct"])
|-
| --tp-sizes || list[int] || No || Tensor parallel sizes (default: [1])
|-
| batch_size || int || Internal || Batch sizes: [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 24576, 32768]
|-
| provider || str || Internal || Configuration: "torch-bf16", "mxfp4", "mxfp4-noquant"
|-
| had_size || int || Internal || Hadamard matrix size: [32, 64, 128]
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| tflops || tuple[float,float,float] || TFLOP/s (median, 80th percentile, 20th percentile)
|-
| plots || files || PNG plots: bench_mxfp4_res_n{N}_k{K}.png
|-
| stdout || text || Tabular TFLOP/s comparison across batch sizes
|}

== Usage Examples ==
<syntaxhighlight lang="python">
# Example 1: Default benchmark (Llama-3.3-70B)
# Tests with Hadamard sizes 32, 64, 128
python benchmarks/kernels/bench_mxfp4_qutlass.py

# Example 2: Custom model
python benchmarks/kernels/bench_mxfp4_qutlass.py \
    --models meta-llama/Llama-2-7b-hf \
    --tp-sizes 1 2

# Example 3: Programmatic usage
from benchmarks.kernels.bench_mxfp4_qutlass import (
    get_hadamard_matrix,
    build_mxfp4_runner,
    PROVIDER_CFGS,
)
import torch
from vllm.triton_utils import triton

M, N, K = 1024, 8192, 8192
device = "cuda"
dtype = torch.bfloat16
had_size = 64

a = torch.randn((M, K), device=device, dtype=dtype)
b = torch.randn((N, K), device=device, dtype=dtype)

# Generate Hadamard matrix
forward_hadamard = get_hadamard_matrix(had_size, dtype, device)

# Build MXFP4 runner with dynamic quantization
cfg = PROVIDER_CFGS["mxfp4"]
run_mxfp4 = build_mxfp4_runner(cfg, a, b, forward_hadamard, dtype, device)

# Benchmark
ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
    lambda: run_mxfp4(), rep=200, quantiles=[0.5, 0.2, 0.8]
)

tflops = (2 * M * N * K) * 1e-12 / (ms * 1e-3)
print(f"MXFP4 (HAD={had_size}): {tflops:.2f} TFLOP/s")

# Example 4: Compare different Hadamard sizes
for had_size in [32, 64, 128]:
    forward_hadamard = get_hadamard_matrix(had_size, dtype, device)
    cfg = PROVIDER_CFGS["mxfp4"]
    run_fn = build_mxfp4_runner(cfg, a, b, forward_hadamard, dtype, device)

    ms, _, _ = triton.testing.do_bench_cudagraph(
        lambda: run_fn(), rep=200, quantiles=[0.5, 0.2, 0.8]
    )

    tflops = (2 * M * N * K) * 1e-12 / (ms * 1e-3)
    print(f"HAD={had_size:3d}: {tflops:6.2f} TFLOP/s")

# Example 5: Understanding MXFP4 quantization
from vllm._custom_ops import fusedQuantizeMx
from vllm.model_executor.layers.quantization.qutlass_utils import to_blocked

# Step 1: Apply Hadamard transform and quantize
weight_e2m1, weight_e8m0 = fusedQuantizeMx(
    b, forward_hadamard, method="abs_max"
)
print(f"Quantized weight shape: {weight_e2m1.shape}")
print(f"Scale shape: {weight_e8m0.shape}")
print(f"Quantized dtype: {weight_e2m1.dtype}")  # E2M1 format

# Step 2: Convert scales to blocked format for efficient GPU access
weight_scale_blocked = to_blocked(weight_e8m0, backend="triton")
print(f"Blocked scale shape: {weight_scale_blocked.shape}")

# Example 6: Memory savings calculation
bf16_memory = M * K * 2  # 2 bytes per element
mxfp4_memory = (M * K * 0.5) + (M * K // had_size) * 1  # 4 bits + scale (E8M0)
savings = (1 - mxfp4_memory / bf16_memory) * 100
print(f"Memory savings: {savings:.1f}%")
# Typical: ~75% memory reduction

# Example 7: Performance expectations
# Hardware: NVIDIA H100
# M=1024, N=8192, K=8192
#
# torch-bf16:           142.3 TFLOP/s (baseline)
# mxfp4 (HAD=32):       287.6 TFLOP/s (2.02x, higher quantization error)
# mxfp4 (HAD=64):       271.4 TFLOP/s (1.91x, balanced)
# mxfp4 (HAD=128):      253.8 TFLOP/s (1.78x, lower quantization error)
# mxfp4-noquant (64):   295.1 TFLOP/s (2.07x, pre-quantized activations)
#
# Larger Hadamard = better accuracy, slightly slower
# Pre-quantization = fastest but requires static activations

# Output format:
# meta-llama/Llama-3.3-70B-Instruct, N=8192 K=8192, HAD=64, BF16 vs MXFP4 GEMMs TFLOP/s:
#
# batch_size     1     4     8    16    32    64   128   256   512  1024  2048  4096  8192 16384 24576 32768
# ----------------------------------------------------------------------------------------------------------------
# torch-bf16  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX
# mxfp4       X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX
# mxfp4-noq.  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX  X.XX
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_CUDA_Environment]]
