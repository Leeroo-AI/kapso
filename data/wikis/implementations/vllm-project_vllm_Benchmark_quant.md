{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::Quantization]], [[domain::FP8]], [[domain::INT8]], [[domain::Kernels]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmarks scaled quantization operations (FP8 and INT8) measuring kernel performance for dynamic activation quantization.

=== Description ===
This focused benchmark evaluates the performance of scaled_fp8_quant and scaled_int8_quant kernels used for dynamic activation quantization in vLLM. These kernels compute quantization scales and convert activations to lower precision formats (FP8 or INT8) on-the-fly during inference. The benchmark supports both static scale mode (pre-computed scales) and dynamic scale mode (computed per-batch).

The benchmark measures kernel execution time across different tensor shapes (num_tokens x hidden_size) and data types (half, bfloat16, float). It includes optional profiling mode using CUDA profiler for detailed kernel analysis. Results help validate quantization kernel performance and identify any bottlenecks in the dynamic quantization pipeline, which is critical for maintaining inference throughput with quantized models.

=== Usage ===
Run this benchmark when validating quantization kernel performance, measuring dynamic quantization overhead, comparing FP8 vs INT8 quantization speed, or profiling quantization operations for optimization.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_quant.py benchmarks/kernels/benchmark_quant.py]

=== Signature ===
<syntaxhighlight lang="python">
@torch.inference_mode()
def main(
    num_tokens: int,
    hidden_size: int,
    static_scale: bool,
    quant_dtype: torch.dtype,
    dtype: torch.dtype,
    seed: int = 0,
    do_profile: bool = False,
    num_warmup_iters: int = 5,
    num_iters: int = 100,
) -> None
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Default benchmark (INT8, 4096 tokens, 8192 hidden)
python benchmarks/kernels/benchmark_quant.py

# FP8 quantization with static scale
python benchmarks/kernels/benchmark_quant.py \
    --quant-dtype fp8 \
    --static-scale \
    --num-tokens 2048 \
    --hidden-size 4096

# INT8 with bfloat16 input
python benchmarks/kernels/benchmark_quant.py \
    --quant-dtype int8 \
    --dtype bfloat16 \
    --num-tokens 8192 \
    --hidden-size 8192

# Profile mode
python benchmarks/kernels/benchmark_quant.py \
    --quant-dtype fp8 \
    --profile

# Custom iterations
python benchmarks/kernels/benchmark_quant.py \
    --num-warmup-iters 10 \
    --num-iters 200
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| num_tokens || int || Number of tokens (default: 4096)
|-
| hidden_size || int || Hidden dimension size (default: 8192)
|-
| static_scale || bool || Use static scale (vs dynamic)
|-
| quant_dtype || str || Quantization dtype: fp8 or int8
|-
| dtype || str || Input dtype: half, bfloat16, float
|-
| seed || int || Random seed (default: 0)
|-
| profile || bool || Enable CUDA profiling
|-
| num_warmup_iters || int || Warmup iterations (default: 5)
|-
| num_iters || int || Benchmark iterations (default: 100, ignored if profile=True)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| kernel_time_us || float || Average kernel execution time in microseconds
|-
| profiler_data || binary || CUDA profiler output (if profile=True)
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# FP8 quantization benchmark
x = torch.randn(num_tokens, hidden_size, dtype=dtype)
scale = torch.randn(1, 1, dtype=torch.float32) if static_scale else None

def run_cuda_benchmark(num_iters: int, profile: bool = False) -> float:
    torch.cuda.synchronize()
    if profile:
        torch.cuda.cudart().cudaProfilerStart()

    start_time = time.perf_counter()
    for _ in range(num_iters):
        if quant_dtype == torch.int8:
            ops.scaled_int8_quant(x, scale)
        else:
            ops.scaled_fp8_quant(x, scale)
    torch.cuda.synchronize()

    end_time = time.perf_counter()
    if profile:
        torch.cuda.cudart().cudaProfilerStop()

    return (end_time - start_time) / num_iters

# Dynamic scale mode (scale=None)
quant_x, computed_scale = ops.scaled_fp8_quant(x, scale=None)

# Static scale mode
quant_x, _ = ops.scaled_fp8_quant(x, scale=scale)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[Implementation:benchmark_per_token_group_quant]]
* [[Concept:Dynamic_Quantization]]
* [[Concept:FP8_Quantization]]
* [[Concept:INT8_Quantization]]
