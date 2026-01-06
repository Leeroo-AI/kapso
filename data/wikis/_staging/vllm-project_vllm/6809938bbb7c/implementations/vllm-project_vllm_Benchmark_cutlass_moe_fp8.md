{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::MoE]], [[domain::CUTLASS]], [[domain::Quantization]], [[domain::FP8]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmarks CUTLASS FP8 MoE kernels against Triton FP8 MoE kernels for mixture-of-experts models with different quantization strategies.

=== Description ===
This benchmark compares the performance of CUTLASS FP8 MoE implementation against Triton FP8 MoE kernels. Both kernels operate on FP8 quantized weights with 16-bit activations, but they use different quantization strategies and execution backends. The benchmark supports both per-tensor and per-channel quantization modes (though per-tensor is currently forced as a workaround for CUTLASS FP8 issues).

The benchmark uses CUDA graphs for accurate performance measurement, capturing 10 invocations per graph to amortize graph launch overhead. It reports results in microseconds for each batch size across multiple model configurations including Mixtral-8x7B, DeepSeek-V2, and custom MoE architectures. Results are displayed in a table format showing timing comparisons between Triton and CUTLASS implementations.

=== Usage ===
Run this benchmark when comparing CUTLASS vs Triton FP8 MoE performance, evaluating different quantization strategies for MoE models, or determining optimal batch sizes for FP8-quantized mixture-of-experts inference.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_cutlass_moe_fp8.py benchmarks/kernels/benchmark_cutlass_moe_fp8.py]

=== Signature ===
<syntaxhighlight lang="python">
def bench_run(
    results: list,
    model: str,
    num_experts: int,
    topk: int,
    per_act_token: bool,
    per_out_ch: bool,
    mkn: tuple[int, int, int],
) -> dict
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Default benchmark (Mixtral-8x7B)
python benchmarks/kernels/benchmark_cutlass_moe_fp8.py

# Specific model with custom settings
python benchmarks/kernels/benchmark_cutlass_moe_fp8.py \
    --models "Llama-4-Maverick-17B-128E-Instruct-FP8" \
    --tp-sizes 8 \
    --batch-sizes 2 4 8 \
    --per-act-token-opts false \
    --per-out-ch-opts false

# DeepSeek-V2 benchmark
python benchmarks/kernels/benchmark_cutlass_moe_fp8.py \
    --models deepseek-v2 \
    --batch-sizes 16 32 64 128
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| models || list[str] || Model names (mixtral-8x7b, deepseek-v2, glm45-fp8, etc.)
|-
| batch_sizes || list[int] || Batch sizes to test (default: 4-2048)
|-
| tp_sizes || list[int] || Tensor parallel sizes
|-
| per_act_token_opts || list[bool] || Per-activation token quantization options
|-
| per_out_ch_opts || list[bool] || Per-output channel quantization options
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
| batch_size || int || Number of tokens processed
|-
| triton_time_us || float || Triton MoE latency in microseconds
|-
| cutlass_time_us || float || CUTLASS MoE latency in microseconds
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Internal benchmark structure
def bench_run(results, model, num_experts, topk, per_act_token,
              per_out_ch, mkn):
    m, k, n = mkn

    # Create FP8 quantized weights
    w1_fp8q = torch.empty(
        (num_experts, 2 * n, k), dtype=torch.float8_e4m3fn
    )
    w2_fp8q = torch.empty(
        (num_experts, k, n), dtype=torch.float8_e4m3fn
    )

    # Benchmark CUTLASS FP8 MoE
    cutlass_graph_time = bench_cuda_graph(cutlass_graph,
                                          num_warmup=5, num_iters=100)

    # Benchmark Triton MoE
    triton_graph_time = bench_cuda_graph(triton_graph,
                                         num_warmup=5, num_iters=100)

    return {
        "batch_size": m,
        "triton_time_us": triton_graph_time * 1000,
        "cutlass_time_us": cutlass_graph_time * 1000
    }
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[Implementation:benchmark_cutlass_fp4_moe]]
* [[Implementation:benchmark_moe]]
* [[Concept:FP8_Quantization]]
* [[Concept:Mixture_of_Experts]]
