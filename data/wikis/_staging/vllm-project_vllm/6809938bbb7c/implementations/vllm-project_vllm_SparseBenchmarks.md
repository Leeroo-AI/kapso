{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::GEMM]], [[domain::Sparsity]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmark for 2:4 structured sparsity GEMM kernels with INT8 and FP8 quantization.

=== Description ===
This benchmark evaluates CUTLASS sparse GEMM kernels that leverage 2:4 structured sparsity patterns for accelerated matrix multiplication. Structured sparsity means every 4 consecutive values contain exactly 2 zeros, allowing specialized hardware acceleration on NVIDIA GPUs. The benchmark compares sparse implementations against dense PyTorch operations and regular CUTLASS scaled matrix multiplications for both INT8 and FP8 datatypes.

The benchmark supports three modes: square matrix benchmarks, range benchmarks with constant dimensions, and model-based benchmarks using real layer shapes from models like Llama-2. Results are serialized as pickle files containing torch.benchmark measurements with timing data and FLOP counts.

=== Usage ===
Run when evaluating structured sparse matrix multiplication performance on NVIDIA GPUs with tensor cores, particularly for quantized inference workloads.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/cutlass_benchmarks/sparse_benchmarks.py benchmarks/cutlass_benchmarks/sparse_benchmarks.py]
* '''Lines:''' 1-515

=== Signature ===
<syntaxhighlight lang="python">
def bench_int8(
    dtype: torch.dtype,
    m: int,
    k: int,
    n: int,
    label: str,
    sub_label: str
) -> Iterable[TMeasurement]

def bench_fp8(
    dtype: torch.dtype,
    m: int,
    k: int,
    n: int,
    label: str,
    sub_label: str
) -> Iterable[TMeasurement]
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Square GEMMs
python3 benchmarks/cutlass_benchmarks/sparse_benchmarks.py \
    --dtype fp8 square_bench \
    --dim-start 128 --dim-end 512 --dim-increment 64

# Constant N and K, sweep M
python3 benchmarks/cutlass_benchmarks/sparse_benchmarks.py \
    --dtype fp8 range_bench \
    --dim-start 128 --dim-end 512 --dim-increment 64 \
    --n-constant 16384 --k-constant 16384

# Model-based dimensions
python3 benchmarks/cutlass_benchmarks/sparse_benchmarks.py \
    --dtype fp8 model_bench \
    --models meta-llama/Llama-2-7b-hf \
    --batch-sizes 16 --tp-sizes 1

# INT8 benchmark
python3 benchmarks/cutlass_benchmarks/sparse_benchmarks.py \
    --dtype int8 square_bench \
    --dim-start 256 --dim-end 1024 --dim-increment 256
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| dtype || torch.dtype || Data type (torch.int8 or torch.float8_e4m3fn)
|-
| m || int || Number of rows in matrix A
|-
| k || int || Shared dimension (columns of A, rows of B)
|-
| n || int || Number of columns in matrix B
|-
| models || list[str] || Model names for model_bench mode
|-
| batch_sizes || list[int] || Batch sizes to benchmark
|-
| tp_sizes || list[int] || Tensor parallel sizes
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| measurements || list[TMeasurement] || Torch benchmark measurements with timing statistics
|-
| pickle_file || .pkl file || Serialized benchmark results with timestamps
|}

== Usage Examples ==

<syntaxhighlight lang="python">
import torch
from benchmarks.cutlass_benchmarks.sparse_benchmarks import bench_fp8, run

# Benchmark specific dimensions
M, K, N = 256, 4096, 4096
timers = bench_fp8(
    dtype=torch.float8_e4m3fn,
    m=M,
    k=K,
    n=N,
    label="sparse-fp8-gemm",
    sub_label=f"MKN=({M}x{K}x{N})"
)

# Run batch of benchmarks
MKNs = [(128, 4096, 4096), (256, 4096, 4096), (512, 4096, 4096)]
results = run(torch.float8_e4m3fn, MKNs)

# Results show comparisons like:
# - pytorch_bf16_bf16_bf16_matmul-no-scales (baseline)
# - cutlass_fp8_fp8_bf16_scaled_mm (dense)
# - cutlass_fp8_fp8_bf16_scaled_sparse_mm (2:4 sparse)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[uses::Library:CUTLASS]]
* [[related::Concept:StructuredSparsity]]
