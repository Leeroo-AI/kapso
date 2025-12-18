{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::GEMM]], [[domain::Quantization]], [[domain::FP4]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmark for NVFP4 (NVIDIA FP4) quantized GEMM operations on Blackwell GPUs.

=== Description ===
This benchmark evaluates NVFP4 quantized matrix multiplication performance, which uses 4-bit floating point format (E2M1) with FP8 scaling factors. NVFP4 quantization requires NVIDIA Blackwell architecture (compute capability 10.0) and combines global scaling with per-block FP8 scales. The benchmark compares both CUTLASS and FBGEMM implementations of NVFP4 GEMM.

The benchmark measures TFLOP/s across batch sizes 1 to 16384 using real model layer dimensions. It tests dynamic quantization (on-the-fly activation quantization) versus pre-quantized activations. FBGEMM support is optional and requires fbgemm-gpu-genai installation. Global scales are computed based on tensor absolute maximum values and FP8/FP4 format ranges.

=== Usage ===
Run when evaluating extreme FP4 quantization on Blackwell GPUs, particularly for comparing CUTLASS vs FBGEMM NVFP4 implementations.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/bench_nvfp4_gemm.py benchmarks/kernels/bench_nvfp4_gemm.py]
* '''Lines:''' 1-198

=== Signature ===
<syntaxhighlight lang="python">
def build_nvfp4_runner(
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
# Requires Blackwell (compute capability 10.0)
# Benchmark Llama-3.1-8B
python benchmarks/kernels/bench_nvfp4_gemm.py \
    --models meta-llama/Llama-3.1-8B-Instruct \
    --tp-sizes 1

# Multiple models
python benchmarks/kernels/bench_nvfp4_gemm.py \
    --models meta-llama/Llama-2-7b-hf meta-llama/Llama-3.1-8B-Instruct \
    --tp-sizes 1 2

# Results saved to bench_nvfp4_res_n{N}_k{K}/
# Tests providers: torch-bf16, nvfp4, nvfp4-noquant,
#                  fbgemm-nvfp4, fbgemm-nvfp4-noquant (if available)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| batch_size || int || Number of tokens (M dimension)
|-
| provider || str || Backend ("torch-bf16", "nvfp4", "nvfp4-noquant", "fbgemm-*")
|-
| N || int || Output dimension
|-
| K || int || Input dimension
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
|}

== Usage Examples ==

<syntaxhighlight lang="python">
import torch
from vllm.scalar_type import scalar_types
from benchmarks.kernels.bench_nvfp4_gemm import build_nvfp4_runner

device = "cuda"
M, K, N = 1024, 4096, 4096
dtype = torch.bfloat16

# Create inputs
a = torch.randn((M, K), device=device, dtype=dtype)
b = torch.randn((N, K), device=device, dtype=dtype)

# Build CUTLASS NVFP4 runner (dynamic quant)
cfg_cutlass = {"no_a_quant": False, "fbgemm": False}
runner_cutlass = build_nvfp4_runner(cfg_cutlass, a, b, dtype, device)

# Build FBGEMM NVFP4 runner (if available)
cfg_fbgemm = {"no_a_quant": False, "fbgemm": True}
runner_fbgemm = build_nvfp4_runner(cfg_fbgemm, a, b, dtype, device)

# Build with pre-quantized activations
cfg_noquant = {"no_a_quant": True, "fbgemm": False}
runner_noquant = build_nvfp4_runner(cfg_noquant, a, b, dtype, device)

# Execute
output_cutlass = runner_cutlass()
output_fbgemm = runner_fbgemm()
output_noquant = runner_noquant()

# Quantization uses global scales:
FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
# global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
