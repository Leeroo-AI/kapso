{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::GEMM]], [[domain::Quantization]], [[domain::BitBLAS]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmark for BitBLAS low-bit quantized matrix multiplication kernels.

=== Description ===
This benchmark evaluates BitBLAS library performance for low-precision quantized matrix multiplications. BitBLAS supports various quantization formats including INT4, INT2, INT1, NF4 (normal float 4-bit), and FP4_E2M1. The benchmark tests both single-token inference (M=1) and batched scenarios (M=8192, M=16384) using weight shapes from large models like BLOOM-176B, OPT-65B, and Llama-70B.

The benchmark supports grouped quantization with configurable group sizes, optional bias addition, scaling factors, and zero-point handling with multiple modes (original, rescale, quantized). It uses BitBLAS's auto-tuning feature to find optimal kernel configurations and reports top-20 tuning results. The tool is particularly useful for evaluating extreme quantization schemes where standard libraries may not provide optimized implementations.

=== Usage ===
Run when evaluating BitBLAS quantization performance for ultra-low precision inference, particularly for INT4/INT2/NF4 formats on large language models.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_bitblas.py benchmarks/kernels/benchmark_bitblas.py]
* '''Lines:''' 1-244

=== Signature ===
<syntaxhighlight lang="python">
# BitBLAS configuration
config = MatmulConfig(
    M, K, N,
    A_dtype, W_dtype, out_dtype, accum_dtype,
    layout, with_bias, group_size,
    with_scaling, with_zeros, zeros_mode
)
matmul = Matmul(config, target=target, enable_tuning=True)
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Basic INT4 benchmark
python benchmarks/kernels/benchmark_bitblas.py \
    --W_dtype int4 \
    --A_dtype float16 \
    --target cuda

# NF4 with grouped quantization
python benchmarks/kernels/benchmark_bitblas.py \
    --W_dtype nf4 \
    --A_dtype float16 \
    --group_size 128 \
    --with_scaling \
    --with_zeros

# FP4 E2M1 format
python benchmarks/kernels/benchmark_bitblas.py \
    --W_dtype fp4_e2m1 \
    --A_dtype float16 \
    --accum_dtype float16 \
    --out_dtype float16

# INT2 extreme quantization
python benchmarks/kernels/benchmark_bitblas.py \
    --W_dtype int2 \
    --A_dtype float16 \
    --group_size 64

# With bias and rescale zeros mode
python benchmarks/kernels/benchmark_bitblas.py \
    --W_dtype int4 \
    --with_bias \
    --with_scaling \
    --with_zeros \
    --zeros_mode rescale

# Auto-detect target GPU
python benchmarks/kernels/benchmark_bitblas.py \
    --target auto \
    --W_dtype int4
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| M || int || Batch/token dimension (from predefined shapes)
|-
| K || int || Input feature dimension
|-
| N || int || Output feature dimension
|-
| A_dtype || str || Activation data type (float16, float32, int8, etc.)
|-
| W_dtype || str || Weight data type (int4, int2, int1, nf4, fp4_e2m1, etc.)
|-
| accum_dtype || str || Accumulation data type (float16, int32)
|-
| out_dtype || str || Output data type
|-
| layout || str || Matrix layout ("nt" or "nn")
|-
| group_size || int || Group size for grouped quantization (optional)
|-
| with_bias || bool || Include bias addition
|-
| with_scaling || bool || Include scaling factors
|-
| with_zeros || bool || Include zero-points
|-
| zeros_mode || str || Zero-point mode (original, rescale, quantized)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| kernel_latency || float || Kernel execution time in milliseconds
|-
| benchmark_results || dict || Detailed results for all tested shapes
|}

== Usage Examples ==

<syntaxhighlight lang="python">
from bitblas import Matmul, MatmulConfig, auto_detect_nvidia_target

# Configure INT4 quantized matmul
M, K, N = 1, 8192, 28672  # Llama-70B shape
A_dtype = "float16"
W_dtype = "int4"
out_dtype = "float16"
accum_dtype = "float16"
layout = "nt"
group_size = 128
with_bias = False
with_scaling = True
with_zeros = True
zeros_mode = "original"

config = MatmulConfig(
    M, K, N,
    A_dtype, W_dtype, out_dtype, accum_dtype,
    layout, with_bias, group_size,
    with_scaling, with_zeros, zeros_mode
)

# Create and tune matmul operator
target = auto_detect_nvidia_target()
matmul = Matmul(config, target=target, enable_tuning=True)

# Profile latency (includes top-20 tuning)
kernel_latency = matmul.profile_latency()
print(f"Kernel latency: {kernel_latency:.3f} ms")

# Test shapes from benchmark
test_shapes = [
    # Single token inference
    (1, 16384, 16384),  # Square test
    (1, 43008, 14336),  # BLOOM-176B
    (1, 8192, 22016),   # Llama-70B
    # Batched inference
    (8192, 16384, 16384),
    (8192, 8192, 28672),
]

for M, K, N in test_shapes:
    config = MatmulConfig(M, K, N, A_dtype, W_dtype, out_dtype,
                          accum_dtype, layout, with_bias, group_size,
                          with_scaling, with_zeros, zeros_mode)
    matmul = Matmul(config, target=target, enable_tuning=True)
    latency = matmul.profile_latency()
    print(f"M={M}, K={K}, N={N}: {latency:.3f} ms")
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[uses::Library:BitBLAS]]
* [[related::Concept:LowBitQuantization]]
* [[related::Concept:GroupedQuantization]]
