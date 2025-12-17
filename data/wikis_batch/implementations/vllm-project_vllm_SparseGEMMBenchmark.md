{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::Quantization]], [[domain::Sparsity]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==
Comprehensive benchmark for 2:4 structured sparse GEMM operations using CUTLASS kernels.

=== Description ===
This benchmark script validates the performance of 2:4 structured sparse matrix multiplication with INT8 and FP8 quantization. 2:4 structured sparsity ensures that in every group of 4 consecutive values, exactly 2 are zeros, enabling 2x theoretical speedup on Ampere+ GPUs. The script compares dense baselines (PyTorch BF16/FP16) against CUTLASS sparse implementations with various quantization schemes (per-tensor, per-token, per-channel), including optional bias addition. It supports square benchmarks, range sweeps, and real model shapes from Llama/Mistral with tensor parallelism.

=== Usage ===
Use this benchmark to validate that 2:4 sparse quantized operations provide expected performance gains (2x over dense) while maintaining accuracy. Critical for proving the viability of sparse quantized inference in production deployments.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/cutlass_benchmarks/sparse_benchmarks.py#L1-L515 benchmarks/cutlass_benchmarks/sparse_benchmarks.py]
* '''Lines:''' 1-515

=== Signature ===
<syntaxhighlight lang="python">
def bench_int8(
    dtype: torch.dtype, m: int, k: int, n: int, label: str, sub_label: str
) -> Iterable[TMeasurement]

def bench_fp8(
    dtype: torch.dtype, m: int, k: int, n: int, label: str, sub_label: str
) -> Iterable[TMeasurement]

def run(
    dtype: torch.dtype, MKNs: Iterable[tuple[int, int, int]]
) -> Iterable[TMeasurement]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# This is a standalone benchmark script
# Run with one of three modes:

# Square GEMMs:
python benchmarks/cutlass_benchmarks/sparse_benchmarks.py --dtype fp8 square_bench \
    --dim-start 128 --dim-end 512 --dim-increment 64

# Range sweep:
python benchmarks/cutlass_benchmarks/sparse_benchmarks.py --dtype fp8 range_bench \
    --dim-start 128 --dim-end 512 --dim-increment 64 \
    --n-constant 16384 --k-constant 16384

# Model shapes:
python benchmarks/cutlass_benchmarks/sparse_benchmarks.py --dtype fp8 model_bench \
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
| pkl_file || file || Pickled results file: {mode}-{dtype}-{timestamp}.pkl
|-
| stdout || text || Formatted comparison tables showing timing and TFLOP/s
|}

== Usage Examples ==
<syntaxhighlight lang="python">
# Example 1: Square GEMMs with INT8
# Tests MxMxM matrices from 128 to 512 in steps of 64
python benchmarks/cutlass_benchmarks/sparse_benchmarks.py \
    --dtype int8 \
    square_bench \
    --dim-start 128 \
    --dim-end 512 \
    --dim-increment 64

# Output includes:
# - pytorch_bf16_bf16_bf16_matmul-no-scales (baseline)
# - pytorch_fp16_fp16_fp16_matmul-no-scales (baseline)
# - cutlass_i8_i8_bf16_scaled_mm (dense)
# - cutlass_i8_i8_bf16_scaled_mm_bias (dense with bias)
# - cutlass_i8_i8_bf16_scaled_sparse_mm (2:4 sparse)
# - cutlass_i8_i8_bf16_scaled_sparse_mm_bias (2:4 sparse with bias)

# Example 2: FP8 range sweep with fixed N and K
# Sweeps M dimension while keeping N=16384, K=16384
python benchmarks/cutlass_benchmarks/sparse_benchmarks.py \
    --dtype fp8 \
    range_bench \
    --dim-start 128 \
    --dim-end 512 \
    --dim-increment 64 \
    --n-constant 16384 \
    --k-constant 16384

# Example 3: Model-specific benchmark (Llama-2-7b)
# Tests all linear layer shapes from the model
python benchmarks/cutlass_benchmarks/sparse_benchmarks.py \
    --dtype fp8 \
    model_bench \
    --models meta-llama/Llama-2-7b-hf \
    --batch-sizes 1 16 64 128 \
    --tp-sizes 1

# Example 4: Multi-model, multi-TP benchmark
python benchmarks/cutlass_benchmarks/sparse_benchmarks.py \
    --dtype fp8 \
    model_bench \
    --models meta-llama/Llama-2-7b-hf mistralai/Mistral-7B-v0.1 \
    --batch-sizes 16 32 64 \
    --tp-sizes 1 2 4

# Example 5: Load and analyze results
import pickle
with open('model_bench-torch.float8_e4m3fn-1234567890.pkl', 'rb') as f:
    measurements = pickle.load(f)

for m in measurements:
    print(f"{m.label}/{m.sub_label}: {m.mean:.6f}s")

# Expected output format:
# ================================================================================
# [0]  scaled-torch.float8_e4m3fn-gemm | MKN=(128x128x128)
# ================================================================================
# Description                                      |   Time (mean +/- std)
# --------------------------------------------------------------------------------
# pytorch_bf16_bf16_bf16_matmul-no-scales         |   X.XXXe-XX +/- X.XXXe-XX
# pytorch_fp8_fp8_bf16_scaled_mm                  |   X.XXXe-XX +/- X.XXXe-XX
# cutlass_fp8_fp8_bf16_scaled_mm                  |   X.XXXe-XX +/- X.XXXe-XX
# cutlass_fp8_fp8_bf16_scaled_sparse_mm           |   X.XXXe-XX +/- X.XXXe-XX
# cutlass_fp8_fp8_bf16_scaled_sparse_mm_bias      |   X.XXXe-XX +/- X.XXXe-XX
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_CUDA_Environment]]
