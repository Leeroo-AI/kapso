{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::MoE]], [[domain::CUTLASS]], [[domain::Quantization]], [[domain::FP4]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmarks CUTLASS FP4 MoE kernels against Triton FP8 MoE kernels for mixture-of-experts models using FP4 quantized weights.

=== Description ===
This benchmark compares the performance of NVIDIA's CUTLASS FP4 MoE implementation against the Triton FP8 MoE kernel. The CUTLASS kernel uses FP4 quantized weights with block-scaled quantization (blocksize=16) and 16-bit activations, while the Triton kernel uses FP8 tensor-scaled weights with 16-bit activations. The benchmark is specifically designed for models like DeepSeek-R1-FP4 that use FP4 quantization for mixture-of-experts layers.

The benchmark tests both standard execution and CUDA graph execution modes, measuring performance across different batch sizes and configurations. It captures 10 invocations per CUDA graph and measures latency with warmup and multiple trials. Results help determine when FP4 quantization provides performance benefits over FP8 quantization for MoE workloads.

=== Usage ===
Run this benchmark when evaluating FP4 quantization for MoE models, comparing CUTLASS FP4 vs Triton FP8 performance, or tuning MoE inference for models like DeepSeek-R1-FP4 across different batch sizes and tensor parallel configurations.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_cutlass_fp4_moe.py benchmarks/kernels/benchmark_cutlass_fp4_moe.py]

=== Signature ===
<syntaxhighlight lang="python">
def bench_run(
    results: list[benchmark.Measurement],
    model: str,
    num_experts: int,
    topk: int,
    per_act_token: bool,
    per_out_ch: bool,
    mkn: tuple[int, int, int],
) -> None
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Default benchmark (all batch sizes)
python benchmarks/kernels/benchmark_cutlass_fp4_moe.py

# Custom model and batch sizes
python benchmarks/kernels/benchmark_cutlass_fp4_moe.py \
    --models nvidia/DeepSeek-R1-FP4 \
    --batch-sizes 4 8 16 32 64 128 \
    --tp-sizes 1

# Limit to specific dimensions
python benchmarks/kernels/benchmark_cutlass_fp4_moe.py \
    --limit-k 2048 \
    --limit-n 7168 \
    --batch-sizes 16 32 64
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| models || list[str] || Model names to benchmark (default: nvidia/DeepSeek-R1-FP4)
|-
| batch_sizes || list[int] || Batch sizes to test (default: 4-2048)
|-
| tp_sizes || list[int] || Tensor parallel sizes (default: [1])
|-
| limit_k || list[int] || Filter specific K dimensions
|-
| limit_n || list[int] || Filter specific N dimensions
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| benchmark_results || Comparison || Formatted table comparing triton_moe, triton_moe_cuda_graphs, cutlass_moe_fp4, cutlass_moe_fp4_cuda_graphs
|-
| execution_time || float || Time per operation in seconds
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Internal benchmark function structure
def run_triton_moe(a, w1, w2, topk_weights, topk_ids,
                   w1_scale, w2_scale, a_fp8_scale, num_repeats):
    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w1_scale, w2_scale=w2_scale, a1_scale=a_fp8_scale
    )
    for _ in range(num_repeats):
        fused_experts(a, w1, w2, topk_weights, topk_ids,
                     quant_config=quant_config)

def run_cutlass_moe_fp4(a, w1_fp4, w2_fp4, w1_blockscale, w2_blockscale,
                        w1_gs, w2_gs, a1_gs, a2_gs, topk_weights, topk_ids,
                        m, n, k, e, device, num_repeats):
    quant_config = nvfp4_moe_quant_config(
        a1_gscale=a1_gs, a2_gscale=a2_gs,
        w1_scale=w1_blockscale, w2_scale=w2_blockscale,
        g1_alphas=w1_gs, g2_alphas=w2_gs
    )
    for _ in range(num_repeats):
        cutlass_moe_fp4(a, w1_fp4, w2_fp4, topk_weights, topk_ids,
                       m, n, k, e, quant_config=quant_config)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[Implementation:benchmark_cutlass_moe_fp8]]
* [[Implementation:benchmark_moe]]
* [[Concept:FP4_Quantization]]
* [[Concept:Mixture_of_Experts]]
