{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::MoE]], [[domain::CUTLASS]], [[domain::GEMM]], [[domain::FP8]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmarks CUTLASS grouped GEMM implementation for FP8-quantized MoE models comparing it against Triton MoE kernels.

=== Description ===
This benchmark evaluates CUTLASS grouped GEMM performance for mixture-of-experts layers with FP8 quantization. It compares the CUTLASS FP8 MoE implementation (using grouped GEMM operations) against the Triton FP8 MoE fused kernel. Both implementations use FP8 quantized weights and handle the complex routing and computation patterns required for MoE architectures.

The benchmark tests performance across various model configurations including Mixtral, DeepSeek-V2, and Granite models, measuring latency for different batch sizes with both standard execution and CUDA graph modes. It captures stride configurations for the grouped GEMM operations and measures performance with 25 runs after warmup, providing comprehensive timing comparisons to identify the fastest implementation for each configuration.

=== Usage ===
Run this benchmark when evaluating CUTLASS grouped GEMM for MoE models, comparing CUTLASS vs Triton performance for FP8 MoE inference, or determining optimal batch sizes for grouped GEMM operations.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_grouped_gemm_cutlass.py benchmarks/kernels/benchmark_grouped_gemm_cutlass.py]

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
# Default benchmark (multiple models)
python benchmarks/kernels/benchmark_grouped_gemm_cutlass.py

# Specific model and batch sizes
python benchmarks/kernels/benchmark_grouped_gemm_cutlass.py \
    --models mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --batch-sizes 1 16 32 64 128 \
    --tp-sizes 1

# Limit to specific dimensions
python benchmarks/kernels/benchmark_grouped_gemm_cutlass.py \
    --models deepseek-ai/DeepSeek-V2-Lite \
    --limit-k 5120 \
    --limit-n 12288
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| models || list[str] || Model names from WEIGHT_SHAPES_MOE
|-
| batch_sizes || list[int] || Batch sizes to test (default: 1-512)
|-
| tp_sizes || list[int] || Tensor parallel sizes
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
| benchmark_results || Comparison || Formatted comparison of triton_moe, triton_moe_cuda_graphs, grouped_gemm_moe, grouped_gemm_moe_cuda_graphs
|-
| execution_time || float || Time per operation in seconds
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Stride tensor setup for CUTLASS grouped GEMM
ab_strides1 = torch.full((num_experts,), k, dtype=torch.int64)
ab_strides2 = torch.full((num_experts,), n, dtype=torch.int64)
c_strides1 = torch.full((num_experts,), 2 * n, dtype=torch.int64)
c_strides2 = torch.full((num_experts,), k, dtype=torch.int64)

# CUTLASS grouped GEMM execution
def run_cutlass_moe(a, a_scale, w1, w2, w1_scale, w2_scale,
                    ab_strides1, ab_strides2, c_strides1, c_strides2,
                    topk_weights, topk_ids, per_act_token, num_repeats):
    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w1_scale, w2_scale=w2_scale,
        per_act_token_quant=per_act_token
    )
    for _ in range(num_repeats):
        cutlass_moe_fp8(
            a, w1, w2, topk_weights, topk_ids,
            ab_strides1, ab_strides2, c_strides1, c_strides2,
            quant_config=quant_config
        )
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[Implementation:benchmark_cutlass_moe_fp8]]
* [[Implementation:benchmark_moe]]
* [[Concept:Grouped_GEMM]]
* [[Concept:Mixture_of_Experts]]
