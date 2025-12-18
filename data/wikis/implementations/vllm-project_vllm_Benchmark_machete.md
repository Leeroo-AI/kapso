{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::Quantization]], [[domain::GEMM]], [[domain::Machete]], [[domain::Mixed_Precision]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmarks Machete mixed-precision GEMM kernels for quantized weight inference comparing against Marlin and torch.matmul implementations.

=== Description ===
This comprehensive benchmark evaluates Machete's mixed-precision GEMM performance for various quantization configurations. Machete supports activations in bfloat16/float16/int8/fp8 and weights in 4-bit or 8-bit formats, with optional group scales, zero points, channel scales, and token scales. The benchmark compares Machete against Marlin (for compatible configurations), CUTLASS scaled_mm (for int8/fp8), and baseline torch.matmul.

The benchmark supports three modes: square_bench (square matrices), range_bench (sweeping M/K/N dimensions), and model_bench (real model shapes). It includes optional schedule sweeping to find the best kernel configuration for each shape, with results saved to CSV. The benchmark creates multiple weight tensors exceeding L2 cache size (>100MB) to ensure realistic performance measurement without cache effects.

=== Usage ===
Run this benchmark when evaluating Machete for quantized inference, comparing mixed-precision GEMM implementations, tuning kernel schedules for specific model shapes, or determining optimal quantization strategies for memory-bound workloads.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_machete.py benchmarks/kernels/benchmark_machete.py]

=== Signature ===
<syntaxhighlight lang="python">
def bench(
    types: TypeConfig,
    group_size: int,
    m: int,
    k: int,
    n: int,
    label: str,
    sub_label: str,
    sweep_schedules: bool = True,
) -> list[TMeasurement]
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Square GEMM benchmark
python benchmarks/kernels/benchmark_machete.py \
    --act-type float16 \
    --group-scale-type float16 \
    --group-size 128 \
    square_bench --dim-start 128 --dim-end 512 --dim-increment 64

# Range benchmark with specific dimensions
python benchmarks/kernels/benchmark_machete.py \
    --act-type bfloat16 \
    --group-size 128 \
    range_bench --dim-start 128,4096,4096 --dim-end 2048,4096,4096 \
    --dim-increment 128,0,0

# Model benchmark with schedule sweeping
python benchmarks/kernels/benchmark_machete.py \
    --act-type float16 \
    --group-scale-type float16 \
    --group-size 128 \
    --sweep-schedules \
    model_bench --models meta-llama/Llama-2-7b-hf --batch-sizes 16 32 64

# FP8 activations with channel scales
python benchmarks/kernels/benchmark_machete.py \
    --act-type float8_e4m3fn \
    --group-scale-type float16 \
    --channel-scale-type float \
    --group-size 128 \
    model_bench --models meta-llama/Llama-3-8b
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| act_type || torch.dtype || Activation dtype (bfloat16, float16, int8, float8_e4m3fn)
|-
| group_scale_type || torch.dtype || Group scale dtype (optional)
|-
| group_zero_type || torch.dtype || Group zero point dtype (optional)
|-
| channel_scale_type || torch.dtype || Channel scale dtype (optional)
|-
| token_scale_type || torch.dtype || Token scale dtype (optional)
|-
| out_type || torch.dtype || Output dtype (optional)
|-
| group_size || int || Quantization group size (default: 128)
|-
| sweep_schedules || bool || Search over all kernel schedules
|-
| sweep_csv_out || str || CSV file for schedule sweep results
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| benchmark_results || list[TMeasurement] || Timing results for all implementations
|-
| best_schedule || str || Optimal kernel schedule configuration
|-
| schedule_sweep_csv || DataFrame || Detailed schedule performance data
|-
| pickle_file || bytes || Serialized benchmark results
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Machete GEMM with group quantization
def machete_create_bench_fn(bt: BenchmarkTensors, out_type, schedule=None):
    w_q = bt.w_q.t().contiguous().t()  # make col major
    w_q = ops.machete_prepack_B(
        w_q, bt.a.dtype, bt.wtype,
        None if bt.w_g_s is None else bt.w_g_s.dtype
    )

    w_g_zp = None if bt.w_g_zp is None else -1 * bt.w_g_s * bt.w_g_zp

    return lambda: ops.machete_mm(
        a=bt.a,
        b_q=w_q,
        b_type=bt.wtype,
        b_group_scales=bt.w_g_s,
        b_group_zeros=w_g_zp,
        b_group_size=bt.group_size,
        b_channel_scales=bt.w_ch_s,
        a_token_scales=bt.w_tok_s,
        out_type=out_type,
        schedule=schedule
    )

# Type configuration
types = TypeConfig(
    act_type=torch.float16,
    weight_type=scalar_types.uint4b8,
    output_type=torch.float16,
    group_scale_type=torch.float16,
    group_zero_type=None,
    channel_scale_type=None,
    token_scale_type=None
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[Implementation:benchmark_marlin]]
* [[Concept:Machete]]
* [[Concept:Mixed_Precision_GEMM]]
* [[Concept:Weight_Quantization]]
