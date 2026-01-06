{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::Quantization]], [[domain::GEMM]], [[domain::Marlin]], [[domain::GPTQ]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmarks Marlin quantized GEMM kernels including GPTQ-Marlin, Marlin-24 (2:4 sparsity), AllSpark W8A16, and FP4/FP8-Marlin variants.

=== Description ===
This comprehensive benchmark evaluates multiple Marlin kernel variants for efficient quantized inference. It tests GPTQ-Marlin for 4-bit and 8-bit quantization with optional activation ordering and zero points, Marlin-24 for 2:4 structured sparsity patterns, AllSpark W8A16 for 8-bit weights with 16-bit activations on Ampere GPUs, and specialized FP4/FP8-Marlin kernels. The benchmark also measures gptq_marlin_repack performance for converting GPTQ weights to Marlin format.

The benchmark supports various configurations including different quantization types (uint4, uint4b8, uint8, float4_e2m1f, float8_e4m3fn), group sizes (16, 32, 64, 128, -1 for channel-wise), activation ordering options, and k_full modes. It compares Marlin implementations against pytorch baseline torch.matmul using float16 reference weights, helping identify optimal configurations for different model architectures and batch sizes.

=== Usage ===
Run this benchmark when evaluating Marlin for quantized inference, comparing different Marlin variants and quantization strategies, determining optimal group sizes and activation ordering, or measuring GPTQ weight repacking overhead.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_marlin.py benchmarks/kernels/benchmark_marlin.py]

=== Signature ===
<syntaxhighlight lang="python">
def bench_run(
    results: list[benchmark.Measurement],
    model: str,
    act_order: bool,
    is_k_full: bool,
    quant_type: ScalarType,
    group_size: int,
    size_m: int,
    size_k: int,
    size_n: int,
) -> None
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Quick benchmark with limited configurations
python benchmarks/kernels/benchmark_marlin.py \
    --batch-sizes 1 16 32 \
    --limit-k 4096 \
    --limit-n 4096 \
    --limit-group-size 128 \
    --limit-num-bits 4 \
    --limit-act-order 0 \
    --limit-k-full 1

# Full model benchmark
python benchmarks/kernels/benchmark_marlin.py \
    --models meta-llama/Llama-2-7b-hf/TP1 \
    --batch-sizes 1 16 32 64 128 256

# Test specific quantization type
python benchmarks/kernels/benchmark_marlin.py \
    --batch-sizes 16 32 64 \
    --limit-num-bits 8 \
    --limit-group-size -1 \
    --limit-act-order 0
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| models || list[str] || Model names from WEIGHT_SHAPES
|-
| batch_sizes || list[int] || Batch sizes to test (default: 1-8192)
|-
| limit_k || list[int] || Filter specific K dimensions
|-
| limit_n || list[int] || Filter specific N dimensions
|-
| limit_group_size || list[int] || Filter specific group sizes
|-
| limit_num_bits || list[int] || Filter specific quantization bit widths
|-
| limit_act_order || list[int] || Filter activation ordering (0/1)
|-
| limit_k_full || list[int] || Filter k_full mode (0/1)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| benchmark_results || Comparison || Formatted comparison of pytorch_gemm, gptq_marlin_gemm, gptq_marlin_gemm_fp32, gptq_marlin_24_gemm, gptq_marlin_repack, allspark_w8a16_gemm_fp32
|-
| execution_time || float || Time per operation in seconds
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# GPTQ-Marlin GEMM with activation ordering
w_ref, q_w, s, g_idx, sort_indices, _ = marlin_quantize(
    b, quant_type, group_size, act_order
)
q_w = ops.gptq_marlin_repack(
    q_w, sort_indices, size_k, size_n, quant_type.size_bits
)
w_s = marlin_permute_scales(s, size_k, size_n, group_size)

output = ops.gptq_marlin_gemm(
    a, None, q_w, w_s, None, None, w_zp,
    g_idx, sort_indices, workspace.scratch,
    quant_type, size_m, size_n, size_k,
    is_k_full, False, False, False
)

# Marlin-24 (2:4 sparsity)
marlin_24_w_ref, marlin_24_q_w_comp, marlin_24_meta, marlin_24_s = \
    marlin_24_quantize(b, quant_type, group_size)

output = ops.gptq_marlin_24_gemm(
    a, marlin_24_q_w_comp, marlin_24_meta, marlin_24_s,
    marlin_24_workspace.scratch, quant_type,
    size_m, size_n, size_k
)

# AllSpark W8A16
qw_reorder, s_reorder, zp_reorder = ops.allspark_repack_weight(
    qw, s, zp, has_zp
)
output = ops.allspark_w8a16_gemm(
    a, qw_reorder, s_reorder, zp_reorder,
    size_n, group_size, sm_count, sm_version,
    CUBLAS_M_THRESHOLD, False, True
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[Implementation:benchmark_machete]]
* [[Concept:Marlin]]
* [[Concept:GPTQ]]
* [[Concept:Structured_Sparsity]]
