{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::GEMM]], [[domain::Quantization]], [[domain::Kernels]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmark for W8A8 quantized GEMM operations with INT8 and FP8 support.

=== Description ===
This benchmark evaluates weight-8bit activation-8bit (W8A8) quantized GEMM kernels using CUTLASS and PyTorch implementations. It supports both INT8 and FP8 quantization schemes with various scaling strategies including tensor-wise, channel-wise, and per-token scaling. For FP8, it includes block-wise scaling with 128x128 block sizes and compares Triton and CUTLASS implementations.

The benchmark measures different quantization configurations: weight scaling (tensor vs channel), activation scaling (tensor vs token), with and without asymmetric zero-points (AZP), and whether activations are pre-quantized or dynamically quantized. It supports model-based benchmarks using real layer dimensions from transformer models and outputs detailed timing comparisons with optional kernel filtering.

=== Usage ===
Run when testing quantized inference kernels on NVIDIA GPUs, particularly for comparing different W8A8 quantization strategies and kernel implementations.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/cutlass_benchmarks/w8a8_benchmarks.py benchmarks/cutlass_benchmarks/w8a8_benchmarks.py]
* '''Lines:''' 1-372

=== Signature ===
<syntaxhighlight lang="python">
def bench_int8(
    dtype: torch.dtype,
    m: int,
    k: int,
    n: int,
    label: str,
    sub_label: str,
    bench_kernels: list[str] | None = None,
) -> Iterable[TMeasurement]

def bench_fp8(
    dtype: torch.dtype,
    m: int,
    k: int,
    n: int,
    label: str,
    sub_label: str,
    bench_kernels: list[str] | None = None,
) -> Iterable[TMeasurement]
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Square GEMMs
python3 benchmarks/cutlass_benchmarks/w8a8_benchmarks.py \
    --dtype fp8 square_bench \
    --dim-start 128 --dim-end 512 --dim-increment 64

# Sweep M with constant N and K
python3 benchmarks/cutlass_benchmarks/w8a8_benchmarks.py \
    --dtype fp8 range_bench \
    --dim-start 128 --dim-end 512 --dim-increment 64 \
    --n-constant 16384 --k-constant 16384

# Model-based dimensions
python3 benchmarks/cutlass_benchmarks/w8a8_benchmarks.py \
    --dtype fp8 model_bench \
    --models meta-llama/Llama-2-7b-hf \
    --batch-sizes 16 --tp-sizes 1

# Filter specific kernels
python3 benchmarks/cutlass_benchmarks/w8a8_benchmarks.py \
    --dtype int8 square_bench \
    --dim-start 256 --dim-end 1024 --dim-increment 256 \
    --kernels cutlass_i8_i8_bf16_scaled_mm cutlass_i8_i8_bf16_scaled_mm_azp

# FP8 blockwise scaling
python3 benchmarks/cutlass_benchmarks/w8a8_benchmarks.py \
    --dtype fp8 model_bench \
    --models meta-llama/Llama-3.1-8B-Instruct \
    --kernels triton_fp8_fp8_fp16_scaled_mm_blockwise \
             cutlass_fp8_fp8_fp16_scaled_mm_blockwise
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| dtype || torch.dtype || Quantization type (torch.int8 or torch.float8_e4m3fn)
|-
| m || int || Batch size / number of tokens
|-
| k || int || Input feature dimension
|-
| n || int || Output feature dimension
|-
| bench_kernels || list[str] (optional) || Specific kernel names to benchmark
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| measurements || list[TMeasurement] || Torch benchmark timing results for each kernel variant
|-
| pickle_file || .pkl file || Serialized results with timestamps
|}

== Usage Examples ==

<syntaxhighlight lang="python">
import torch
from benchmarks.cutlass_benchmarks.w8a8_benchmarks import bench_fp8, bench_int8

# Benchmark FP8 kernels
M, K, N = 1024, 4096, 4096
fp8_results = bench_fp8(
    dtype=torch.float8_e4m3fn,
    m=M, k=K, n=N,
    label="w8a8-fp8",
    sub_label=f"MKN=({M}x{K}x{N})",
    bench_kernels=None  # Run all kernels
)

# Benchmark specific INT8 kernels with AZP
int8_kernels = [
    "cutlass_i8_i8_bf16_scaled_mm_azp",
    "cutlass_i8_i8_bf16_scaled_mm_azp_bias"
]
int8_results = bench_int8(
    dtype=torch.int8,
    m=M, k=K, n=N,
    label="w8a8-int8-azp",
    sub_label=f"MKN=({M}x{K}x{N})",
    bench_kernels=int8_kernels
)

# Results include variants like:
# - pytorch_bf16_bf16_bf16_matmul-no-scales (baseline)
# - cutlass_fp8_fp8_bf16_scaled_mm (tensor scaling)
# - triton_fp8_fp8_fp16_scaled_mm_blockwise (block scaling)
# - cutlass_i8_i8_bf16_scaled_mm_azp (asymmetric zero-point)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[uses::Library:CUTLASS]]
* [[related::Concept:W8A8Quantization]]
