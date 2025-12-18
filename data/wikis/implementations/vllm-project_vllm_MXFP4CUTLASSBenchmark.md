{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::GEMM]], [[domain::Quantization]], [[domain::MXFP4]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmark for MXFP4 (Microscaling FP4) quantized GEMM with Hadamard transformation and CUTLASS backend.

=== Description ===
This benchmark evaluates MXFP4 quantized matrix multiplication performance using microscaling FP4 format combined with Hadamard transformations. MXFP4 uses 4-bit floating point values (E2M1 format) with block-wise FP8 scaling factors (E8M0), providing better numerical stability than plain FP4. The Hadamard transformation decorrelates features before quantization to improve accuracy.

The benchmark compares MXFP4 against BF16 baseline across various Hadamard matrix sizes (32, 64, 128) and batch sizes (1 to 32768). It tests both pre-quantized weights with dynamic activation quantization and pre-quantized activations. The fusedQuantizeMx operation combines Hadamard transformation with quantization, and results are measured in TFLOP/s using CUTLASS-based GEMM kernels.

=== Usage ===
Run when evaluating MXFP4 quantization for extreme compression scenarios, particularly for large models like Llama-3.3-70B.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/bench_mxfp4_qutlass.py benchmarks/kernels/bench_mxfp4_qutlass.py]
* '''Lines:''' 1-191

=== Signature ===
<syntaxhighlight lang="python">
def build_mxfp4_runner(
    cfg: dict,
    a: torch.Tensor,
    b: torch.Tensor,
    forward_hadamard_matrix: torch.Tensor,
    dtype: torch.dtype,
    device: str
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
# Benchmark Llama-3.3-70B with default Hadamard sizes
python benchmarks/kernels/bench_mxfp4_qutlass.py \
    --models meta-llama/Llama-3.3-70B-Instruct \
    --tp-sizes 1

# Multiple models and TP configs
python benchmarks/kernels/bench_mxfp4_qutlass.py \
    --models meta-llama/Llama-3.3-70B-Instruct \
    --tp-sizes 1 2 4 8

# Tests Hadamard sizes: 32, 64, 128
# Results saved to bench_mxfp4_res_n{N}_k{K}.png/pkl
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| batch_size || int || Number of tokens (M dimension)
|-
| provider || str || Backend ("torch-bf16", "mxfp4", "mxfp4-noquant")
|-
| N || int || Output dimension
|-
| K || int || Input dimension
|-
| had_size || int || Hadamard matrix dimension (32, 64, or 128)
|-
| models || list[str] || Model names for layer shapes
|-
| tp_sizes || list[int] || Tensor parallelism sizes
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
from benchmarks.kernels.bench_mxfp4_qutlass import build_mxfp4_runner

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

# Build MXFP4 runner with dynamic activation quantization
cfg = {"no_a_quant": False}
runner = build_mxfp4_runner(cfg, a, b, forward_hadamard, dtype, device)

# Execute
output = runner()

# Build with pre-quantized activations
cfg_noquant = {"no_a_quant": True}
runner_noquant = build_mxfp4_runner(
    cfg_noquant, a, b, forward_hadamard, dtype, device
)
output_noquant = runner_noquant()

# Available providers:
# - "torch-bf16" (baseline)
# - "mxfp4" (dynamic quantization)
# - "mxfp4-noquant" (pre-quantized)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[uses::Library:CUTLASS]]
* [[related::Concept:MicroscalingQuantization]]
