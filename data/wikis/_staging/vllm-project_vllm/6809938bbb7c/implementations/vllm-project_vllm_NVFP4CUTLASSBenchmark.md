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
Benchmark for NVFP4 quantized GEMM with Hadamard transformation using CUTLASS backend.

=== Description ===
This benchmark evaluates NVFP4 quantized matrix multiplication combined with Hadamard transformations, using the CUTLASS FP4 GEMM kernel. Unlike the standard NVFP4 benchmark, this version applies Hadamard rotation before quantization to decorrelate features and improve numerical properties. The fusedQuantizeNv operation combines Hadamard transformation with NVFP4 quantization into E2M1 format with E8M0 block scales.

The benchmark tests various Hadamard matrix sizes (16, 32, 64, 128) and batch sizes (1 to 32768) using model layer dimensions. It compares dynamic quantization against pre-quantized activations, measuring TFLOP/s performance. The implementation uses CUTLASS scaled_fp4_mm kernels with blocked scale layout optimized for Triton backend.

=== Usage ===
Run when evaluating NVFP4 with Hadamard preprocessing on Blackwell GPUs, particularly for models like Llama-3.3-70B requiring extreme compression.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/bench_nvfp4_qutlass.py benchmarks/kernels/bench_nvfp4_qutlass.py]
* '''Lines:''' 1-207

=== Signature ===
<syntaxhighlight lang="python">
def build_nvfp4_runner(
    cfg: dict,
    a: torch.Tensor,
    b: torch.Tensor,
    forward_hadamard_matrix: torch.Tensor,
    dtype: torch.dtype,
    device: str,
    M: int,
    N: int,
    K: int
) -> Callable

@triton.testing.perf_report(...)
def benchmark(
    batch_size: int,
    provider: str,
    N: int,
    K: int,
    had_size: int
)
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Benchmark Llama-3.3-70B with various Hadamard sizes
python benchmarks/kernels/bench_nvfp4_qutlass.py \
    --models meta-llama/Llama-3.3-70B-Instruct \
    --tp-sizes 1

# Multiple models and TP configurations
python benchmarks/kernels/bench_nvfp4_qutlass.py \
    --models meta-llama/Llama-3.3-70B-Instruct \
    --tp-sizes 1 2 4 8

# Tests Hadamard sizes: 16, 32, 64, 128
# Results saved to bench_nvfp4_res_n{N}_k{K}.png/pkl
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| batch_size || int || Number of tokens (M dimension)
|-
| provider || str || Backend ("torch-bf16", "nvfp4", "nvfp4-noquant")
|-
| N || int || Output dimension
|-
| K || int || Input dimension (must be divisible by 16)
|-
| had_size || int || Hadamard matrix dimension (16, 32, 64, or 128)
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
from compressed_tensors.transform.utils.hadamard import deterministic_hadamard_matrix
from benchmarks.kernels.bench_nvfp4_qutlass import build_nvfp4_runner

device = "cuda"
M, K, N = 1024, 4096, 4096
had_size = 64
dtype = torch.bfloat16

# Create inputs
a = torch.randn((M, K), device=device, dtype=dtype)
b = torch.randn((N, K), device=device, dtype=dtype)

# Generate Hadamard matrix
forward_hadamard = (
    deterministic_hadamard_matrix(had_size, dtype=dtype, device=device)
    * had_size**-0.5
)

# Build NVFP4 runner with dynamic quantization
cfg = {"no_a_quant": False}
runner = build_nvfp4_runner(
    cfg, a, b, forward_hadamard, dtype, device, M, N, K
)

# Execute
output = runner()

# Build with pre-quantized activations
cfg_noquant = {"no_a_quant": True}
runner_noquant = build_nvfp4_runner(
    cfg_noquant, a, b, forward_hadamard, dtype, device, M, N, K
)
output_noquant = runner_noquant()

# The fusedQuantizeNv operation:
# 1. Applies Hadamard transformation
# 2. Quantizes to E2M1 format
# 3. Generates E8M0 block scales
# 4. Reshapes scales for Triton backend: (-1, K // 16)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
