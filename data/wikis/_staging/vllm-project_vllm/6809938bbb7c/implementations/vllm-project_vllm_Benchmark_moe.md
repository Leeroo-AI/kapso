{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::MoE]], [[domain::Kernels]], [[domain::Triton]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Comprehensive MoE kernel tuning benchmark for finding optimal Triton kernel configurations across different models, batch sizes, and quantization modes.

=== Description ===
This is the main MoE kernel benchmarking and tuning tool for vLLM. It benchmarks fused_experts (Triton MoE kernel) performance and optionally tunes kernel configurations by sweeping over BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_warps, and num_stages parameters. The benchmark supports FP16/BF16 unquantized, FP8 W8A8 quantized (with optional block quantization), and INT8 W8A16 quantized modes, as well as DeepGEMM fusion.

The tuning mode uses Ray for distributed benchmarking across multiple GPUs, testing extensive configuration spaces (thousands of combinations for compute-bound kernels). For ROCm, it includes specialized pruning logic to reduce the search space based on GEMM dimensions and hardware characteristics. Results are saved as JSON configuration files that vLLM uses to select optimal kernel parameters at runtime for each (num_experts, intermediate_size, dtype, block_shape) combination.

=== Usage ===
Run this benchmark when tuning MoE kernels for new models, evaluating different quantization strategies, finding optimal Triton configurations for specific hardware, or comparing baseline vs tuned performance across batch sizes.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py benchmarks/kernels/benchmark_moe.py]

=== Signature ===
<syntaxhighlight lang="python">
def benchmark_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    num_iters: int = 100,
    block_quant_shape: list[int] = None,
    use_deep_gemm: bool = False,
) -> float
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Benchmark mode (use existing configs)
python benchmarks/kernels/benchmark_moe.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tp-size 2 \
    --batch-size 16 32 64

# Tuning mode (find optimal configs)
python benchmarks/kernels/benchmark_moe.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tp-size 2 \
    --tune \
    --save-dir ./configs

# FP8 quantization with block quant
python benchmarks/kernels/benchmark_moe.py \
    --model deepseek-ai/DeepSeek-V2-Lite \
    --dtype fp8_w8a8 \
    --tp-size 4 \
    --tune

# INT8 W8A16 quantization
python benchmarks/kernels/benchmark_moe.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --dtype int8_w8a16 \
    --batch-size 16 32 64

# Expert parallelism
python benchmarks/kernels/benchmark_moe.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --enable-expert-parallel \
    --tp-size 2

# DeepGEMM fusion (experimental)
python benchmarks/kernels/benchmark_moe.py \
    --model deepseek-ai/DeepSeek-V2-Lite \
    --use-deep-gemm \
    --dtype fp8_w8a8
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || str || Model name (Mixtral, DeepSeek-V2, Qwen2-MoE, etc.)
|-
| tp_size || int || Tensor parallel size (default: 2)
|-
| dtype || str || Quantization mode: auto, fp8_w8a8, int8_w8a16
|-
| batch_size || list[int] || Batch sizes to test (default: 1-4096)
|-
| tune || bool || Enable tuning mode to find optimal configs
|-
| save_dir || str || Directory to save tuned configs (default: ./)
|-
| seed || int || Random seed (default: 0)
|-
| enable_expert_parallel || bool || Enable expert parallelism
|-
| use_deep_gemm || bool || Enable DeepGEMM fusion
|-
| trust_remote_code || bool || Trust remote code for model loading
|-
| model_prefix || str || Config prefix for nested model configs
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| config || BenchmarkConfig || Optimal Triton kernel configuration
|-
| kernel_time || float || Average kernel time in microseconds
|-
| config_json || dict || JSON file with tuned configs per batch size
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Internal benchmark structure
def benchmark_config(config, num_tokens, num_experts,
                    shard_intermediate_size, hidden_size,
                    topk, dtype, use_fp8_w8a8, use_int8_w8a16,
                    num_iters=100, block_quant_shape=None,
                    use_deep_gemm=False):
    # Create input tensors
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    w1 = torch.randn(num_experts, shard_intermediate_size,
                    hidden_size, dtype=dtype)
    w2 = torch.randn(num_experts, hidden_size,
                    shard_intermediate_size // 2, dtype=dtype)

    # Setup quantization config
    quant_config = FusedMoEQuantConfig.make(
        quant_dtype=quant_dtype,
        w1_scale=w1_scale, w2_scale=w2_scale,
        a1_scale=a1_scale, a2_scale=a2_scale,
        block_shape=block_quant_shape
    )

    # Benchmark with CUDA graph
    def run():
        with override_config(config):
            topk_weights, topk_ids, _ = fused_topk(x, input_gating, topk)
            return fused_experts(x, w1, w2, topk_weights, topk_ids,
                               quant_config=quant_config,
                               allow_deep_gemm=use_deep_gemm)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            run()

    # Measure latency
    for i in range(num_iters):
        start_event.record()
        graph.replay()
        end_event.record()

    return avg_latency_us

# Tuning search space
configs = get_configs_compute_bound(use_fp16=True,
                                   block_quant_shape=None)
# Returns configs with BLOCK_SIZE_M/N/K, GROUP_SIZE_M,
# num_warps, num_stages variations
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[Implementation:benchmark_cutlass_moe_fp8]]
* [[Implementation:benchmark_cutlass_fp4_moe]]
* [[Concept:Mixture_of_Experts]]
* [[Concept:Kernel_Tuning]]
