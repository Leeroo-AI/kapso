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
Comprehensive benchmark for W8A8 (8-bit weight, 8-bit activation) quantized GEMM operations.

=== Description ===
This benchmark script provides extensive performance testing for INT8 and FP8 quantized dense matrix multiplication. It compares multiple quantization granularities: per-tensor (single scale for entire tensor), per-token (row-wise scaling), per-channel (column-wise scaling), and per-group/block (fine-grained scaling). Each configuration can be tested with or without dynamic activation quantization. The script supports square benchmarks, range sweeps, and real model layer shapes across different tensor parallel configurations, measuring TFLOP/s for PyTorch baselines and CUTLASS/Triton implementations.

=== Usage ===
Use this benchmark to validate W8A8 quantization performance against BF16 baselines, understand the trade-offs between quantization granularity and speed, and determine optimal quantization strategies for specific models and hardware configurations.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/cutlass_benchmarks/w8a8_benchmarks.py#L1-L372 benchmarks/cutlass_benchmarks/w8a8_benchmarks.py]
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

=== Import ===
<syntaxhighlight lang="python">
# This is a standalone benchmark script
# Run with one of three modes:

# Square GEMMs:
python benchmarks/cutlass_benchmarks/w8a8_benchmarks.py --dtype fp8 square_bench \
    --dim-start 128 --dim-end 512 --dim-increment 64

# Range sweep:
python benchmarks/cutlass_benchmarks/w8a8_benchmarks.py --dtype int8 range_bench \
    --dim-start 128 --dim-end 512 --dim-increment 64 \
    --n-constant 16384 --k-constant 16384

# Model shapes:
python benchmarks/cutlass_benchmarks/w8a8_benchmarks.py --dtype fp8 model_bench \
    --models meta-llama/Llama-2-7b-hf --batch-sizes 16 --tp-sizes 1
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| --dtype || str || Yes || Quantization dtype: 'int8' or 'fp8'
|-
| --kernels || list[str] || No || Specific kernel names to benchmark (default: all)
|-
| mode || str || Yes || Benchmark mode: square_bench, range_bench, or model_bench
|-
| --dim-start || int || Yes* || Starting dimension size (*required for square/range_bench)
|-
| --dim-end || int || Yes* || Ending dimension size (*required for square/range_bench)
|-
| --dim-increment || int || Yes* || Dimension increment step (*required for square/range_bench)
|-
| --m-constant || int || No || Fixed M dimension (range_bench only)
|-
| --n-constant || int || No || Fixed N dimension (range_bench only)
|-
| --k-constant || int || No || Fixed K dimension (range_bench only)
|-
| --models || list[str] || No || Model names (default: all in WEIGHT_SHAPES)
|-
| --tp-sizes || list[int] || No || Tensor parallel sizes (default: [1])
|-
| --batch-sizes || list[int] || No || Batch sizes (default: [1,16,32,64,128,256,512])
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| measurements || list[TMeasurement] || PyTorch benchmark measurements for each configuration
|-
| pkl_file || file || Pickled results: {mode}-{dtype}-{timestamp}.pkl
|-
| stdout || text || Formatted comparison tables showing timing and speedup
|}

== Usage Examples ==
<syntaxhighlight lang="python">
# Example 1: FP8 square benchmark with all kernels
python benchmarks/cutlass_benchmarks/w8a8_benchmarks.py \
    --dtype fp8 \
    square_bench \
    --dim-start 128 \
    --dim-end 512 \
    --dim-increment 64

# Tests these kernels:
# - pytorch_bf16_bf16_bf16_matmul-no-scales (baseline)
# - pytorch_fp8_fp8_bf16_scaled_mm (PyTorch FP8)
# - pytorch_fp8_fp8_bf16_scaled_mm_fast_accum (with fast accumulation)
# - cutlass_fp8_fp8_bf16_scaled_mm (CUTLASS implementation)
# - cutlass_fp8_fp8_bf16_scaled_mm_bias (with bias)
# - triton_fp8_fp8_fp16_scaled_mm_blockwise (Triton block-quantized)
# - cutlass_fp8_fp8_fp16_scaled_mm_blockwise (CUTLASS block-quantized)

# Example 2: INT8 with specific kernels only
python benchmarks/cutlass_benchmarks/w8a8_benchmarks.py \
    --dtype int8 \
    --kernels cutlass_i8_i8_bf16_scaled_mm cutlass_i8_i8_bf16_scaled_mm_azp \
    model_bench \
    --models meta-llama/Llama-2-7b-hf \
    --batch-sizes 16 64

# Example 3: Compare per-tensor vs per-channel quantization
python benchmarks/cutlass_benchmarks/w8a8_benchmarks.py \
    --dtype fp8 \
    --kernels pytorch_fp8_fp8_bf16_scaled_mm \
    range_bench \
    --dim-start 1 \
    --dim-end 1024 \
    --dim-increment 128 \
    --n-constant 4096 \
    --k-constant 4096

# Example 4: Full model benchmark across TP configurations
python benchmarks/cutlass_benchmarks/w8a8_benchmarks.py \
    --dtype fp8 \
    model_bench \
    --models meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf \
    --batch-sizes 1 8 16 32 64 128 \
    --tp-sizes 1 2 4 8

# Example 5: INT8 with asymmetric zero-point (AZP) testing
python benchmarks/cutlass_benchmarks/w8a8_benchmarks.py \
    --dtype int8 \
    --kernels cutlass_i8_i8_bf16_scaled_mm_azp cutlass_i8_i8_bf16_scaled_mm_azp_pt \
    square_bench \
    --dim-start 256 \
    --dim-end 2048 \
    --dim-increment 256

# Expected output format:
# ================================================================================
# [0]  scaled-torch.float8_e4m3fn-gemm | MKN=(128x128x128)
# ================================================================================
# Description                                      |   Time (mean +/- std)
# --------------------------------------------------------------------------------
# pytorch_bf16_bf16_bf16_matmul-no-scales         |   X.XXXe-XX +/- X.XXXe-XX
# pytorch_fp8_fp8_bf16_scaled_mm                  |   X.XXXe-XX +/- X.XXXe-XX
# pytorch_fp8_fp8_bf16_scaled_mm_fast_accum       |   X.XXXe-XX +/- X.XXXe-XX
# cutlass_fp8_fp8_bf16_scaled_mm                  |   X.XXXe-XX +/- X.XXXe-XX
# cutlass_fp8_fp8_bf16_scaled_mm_bias             |   X.XXXe-XX +/- X.XXXe-XX
# triton_fp8_fp8_fp16_scaled_mm_blockwise         |   X.XXXe-XX +/- X.XXXe-XX
# cutlass_fp8_fp8_fp16_scaled_mm_blockwise        |   X.XXXe-XX +/- X.XXXe-XX

# Kernel naming convention:
# {backend}_{weight_dtype}_{act_dtype}_{output_dtype}_scaled_mm[_variant]
# - backend: pytorch, cutlass, triton
# - variant: bias, azp (zero-point), blockwise, etc.
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_CUDA_Environment]]
