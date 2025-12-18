{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::Quantization]], [[domain::FP8]], [[domain::INT8]], [[domain::Dynamic_Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmarks per-token group quantization for FP8 and INT8 comparing CUDA and Triton implementations with various configuration options.

=== Description ===
This benchmark evaluates per-token group quantization performance for dynamic FP8 and INT8 quantization. Per-token group quantization divides the hidden dimension into groups and computes separate quantization scales for each group within each token, providing better accuracy than per-tensor quantization while maintaining reasonable compute overhead. The benchmark compares CUDA kernel implementations against Triton fallback implementations.

For FP8 quantization, it tests both row-major and column-major scale layouts, as well as UE8M0 (unsigned 8-bit exponent, 0-bit mantissa) scale format option. For INT8, it uses standard per-token group quantization without these additional options. The benchmark measures performance across different tensor shapes and group sizes (64, 128), reporting speedup of CUDA over Triton implementations to validate kernel optimizations.

=== Usage ===
Run this benchmark when validating per-token group quantization implementations, comparing CUDA vs Triton performance, evaluating different scale layout options for FP8, or determining optimal group sizes for dynamic quantization.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_per_token_group_quant.py benchmarks/kernels/benchmark_per_token_group_quant.py]

=== Signature ===
<syntaxhighlight lang="python">
def _run_single(
    shape: tuple[int, int],
    group_size: int,
    dtype: str,
    *,
    column_major: bool = False,
    scale_ue8m0: bool = False,
    warmup_iters: int,
    bench_iters: int,
) -> None
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Default benchmark (both FP8 and INT8)
python benchmarks/kernels/benchmark_per_token_group_quant.py

# FP8 only
python benchmarks/kernels/benchmark_per_token_group_quant.py \
    --dtype fp8 \
    --warmup-iters 10 \
    --bench-iters 100

# INT8 only
python benchmarks/kernels/benchmark_per_token_group_quant.py \
    --dtype int8 \
    --warmup-iters 5 \
    --bench-iters 50

# Custom iterations
python benchmarks/kernels/benchmark_per_token_group_quant.py \
    --warmup-iters 20 \
    --bench-iters 200
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| dtype || str || Quantization dtype: fp8, int8, or both
|-
| warmup_iters || int || Number of warmup iterations (default: 10)
|-
| bench_iters || int || Number of benchmark iterations (default: 100)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| configuration || str || Shape, group_size, column_major, ue8m0, dtype
|-
| cuda_time_ms || float || CUDA implementation time in milliseconds
|-
| triton_time_ms || float || Triton implementation time in milliseconds
|-
| speedup || float || CUDA speedup over Triton (triton_time / cuda_time)
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# FP8 per-token group quantization (CUDA)
def cuda_impl():
    return fp8_utils.per_token_group_quant_fp8(
        x,
        group_size,
        column_major_scales=column_major,
        use_ue8m0=scale_ue8m0
    )

# FP8 per-token group quantization (Triton)
def triton_impl():
    with _triton_mode():
        return fp8_utils.per_token_group_quant_fp8(
            x,
            group_size,
            column_major_scales=column_major,
            use_ue8m0=scale_ue8m0
        )

# INT8 per-token group quantization (CUDA)
def cuda_impl():
    return int8_utils.per_token_group_quant_int8(x, group_size)

# INT8 per-token group quantization (Triton)
def triton_impl():
    with _triton_mode():
        return int8_utils.per_token_group_quant_int8(x, group_size)

# Benchmark execution
x = torch.randn(num_tokens, hidden_dim,
               device="cuda", dtype=torch.bfloat16) * 8

cuda_ms = _time_cuda(cuda_impl, warmup_iters, bench_iters)
triton_ms = _time_cuda(triton_impl, warmup_iters, bench_iters)
speedup = triton_ms / cuda_ms
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[Implementation:benchmark_quant]]
* [[Concept:Dynamic_Quantization]]
* [[Concept:Per_Token_Quantization]]
* [[Concept:FP8_Quantization]]
