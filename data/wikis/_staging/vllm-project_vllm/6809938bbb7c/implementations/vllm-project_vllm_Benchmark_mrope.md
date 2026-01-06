{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::Attention]], [[domain::RoPE]], [[domain::Triton]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmarks multi-dimensional RoPE (mRoPE) kernel performance comparing Triton implementation against native PyTorch for Qwen2-VL and Qwen2.5-VL models.

=== Description ===
This benchmark evaluates the performance of the Triton-optimized mRoPE (multi-dimensional rotary position embedding) kernel against the native PyTorch implementation. The mRoPE mechanism is used in Qwen2-VL and Qwen2.5-VL models to handle multi-dimensional positional encodings for vision-language tasks. Unlike standard RoPE which uses 1D positions, mRoPE uses 2D or 3D positions to represent spatial and temporal information.

The benchmark generates 2D position tensors (3 x num_tokens) for multimodal cases and applies rotary embeddings to query and key tensors. It measures performance across various batch sizes (powers of 2 from 1 to 131072 tokens), different tensor parallel sizes, and multiple Qwen model configurations. Results are saved to CSV with detailed statistics including mean, median, p99, min, max latencies and speedup ratios, helping validate the Triton optimization benefits.

=== Usage ===
Run this benchmark when validating mRoPE optimizations for Qwen2-VL models, comparing Triton vs native PyTorch performance, or evaluating rotary embedding overhead across different batch sizes and tensor parallel configurations.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_mrope.py benchmarks/kernels/benchmark_mrope.py]

=== Signature ===
<syntaxhighlight lang="python">
def benchmark_mrope(
    model_name: str,
    num_tokens: int,
    head_dim: int,
    tp_size: int,
    num_heads: int,
    num_kv_heads: int,
    max_position: int = 8192,
    is_neox_style: bool = True,
    rope_parameters: dict[str, Any] | None = None,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
    warmup_iter: int = 10,
    benchmark_iter: int = 100,
    csv_writer=None,
) -> tuple[dict[str, float], dict[str, float]]
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Single model benchmark
python benchmarks/kernels/benchmark_mrope.py \
    --model-name Qwen/Qwen2-VL-7B-Instruct \
    --tp-size 1 \
    --warmup-iter 10 \
    --benchmark-iter 100 \
    --dtype bfloat16 \
    --num-tokens 1024

# All Qwen2-VL models
python benchmarks/kernels/benchmark_mrope.py \
    --model-name "" \
    --tp-size 1 \
    --num-tokens 1024 4096 16384

# Multiple TP sizes
python benchmarks/kernels/benchmark_mrope.py \
    --model-name Qwen/Qwen2-VL-72B-Instruct \
    --tp-size 2 4 8 \
    --num-tokens 2048

# Custom output file
python benchmarks/kernels/benchmark_mrope.py \
    --model-name Qwen/Qwen2.5-VL-7B-Instruct \
    --output-csv qwen25_mrope_results.csv
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model_name || str || Model name (Qwen2-VL or Qwen2.5-VL variants)
|-
| tp_size || int || Tensor parallel size (default: 1)
|-
| num_tokens || list[int] || Token counts to test (default: 2^0 to 2^17)
|-
| dtype || str || Data type (bfloat16)
|-
| warmup_iter || int || Warmup iterations (default: 10)
|-
| benchmark_iter || int || Benchmark iterations (default: 100)
|-
| seed || int || Random seed (default: 0)
|-
| trust_remote_code || bool || Trust remote code for model loading
|-
| output_csv || str || Output CSV filename (default: mrope_benchmark_results.csv)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| torch_stats || dict || Mean, median, p99, min, max for native PyTorch (seconds)
|-
| triton_stats || dict || Mean, median, p99, min, max for Triton kernel (seconds)
|-
| speedup || float || Triton speedup over PyTorch (torch_mean / triton_mean)
|-
| csv_results || CSV || Detailed results with all statistics per configuration
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Generate test data for mRoPE
def generate_test_data(num_tokens, num_q_heads, num_kv_heads,
                      head_size, max_position_embeddings,
                      dtype, device):
    # Create 2D positions (3, num_tokens) for multimodal case
    positions = torch.randint(
        0, max_position_embeddings // 4, (3, num_tokens), device=device
    )

    query = torch.randn(num_tokens, num_q_heads * head_size,
                       dtype=dtype, device=device)
    key = torch.randn(num_tokens, num_kv_heads * head_size,
                     dtype=dtype, device=device)

    return positions, query, key

# Benchmark mRoPE
mrope_helper = get_rope(
    head_size=head_dim,
    max_position=max_position,
    is_neox_style=is_neox_style,
    rope_parameters=rope_parameters,
    dtype=dtype
)

# Native PyTorch implementation
torch.cuda.synchronize()
start_time = time.time()
mrope_helper.forward_native(positions, query, key)
torch.cuda.synchronize()
torch_time = time.time() - start_time

# Triton kernel implementation
torch.cuda.synchronize()
start_time = time.time()
mrope_helper.forward_cuda(positions, query, key)
torch.cuda.synchronize()
triton_time = time.time() - start_time

speedup = torch_time / triton_time
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[Concept:Rotary_Position_Embedding]]
* [[Concept:Multi_Dimensional_RoPE]]
* [[Concept:Qwen2_VL]]
* [[Concept:Vision_Language_Models]]
