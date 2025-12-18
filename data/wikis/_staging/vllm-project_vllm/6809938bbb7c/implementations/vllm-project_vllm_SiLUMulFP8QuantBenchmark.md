{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::Kernels]], [[domain::Quantization]], [[domain::Activation]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmark for fused 2D SiLU-mul-FP8-quantization kernel comparing against unfused reference implementation.

=== Description ===
This benchmark evaluates a highly optimized fused kernel that combines three operations: SiLU activation, element-wise multiplication, and per-token group FP8 quantization. The fusion targets the MLP gating pattern common in transformers where inputs are split, one half goes through SiLU, then multiplied with the other half, and finally quantized for subsequent GEMM operations.

The benchmark tests various token counts (128 to 128K) and hidden dimensions (2048 to 8192) using CUDA graph benchmarking with argument pooling for stability. It compares the fused implementation against a reference that runs silu_and_mul followed by separate FP8 quantization. The kernel uses 128-element groups for quantization and generates column-major scale layouts optimized for downstream GEMM kernels.

=== Usage ===
Run when optimizing MLP gate projection fusion in transformer models, particularly for memory-bound activation-quantization sequences.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_2d_silu_mul_fp8_quant.py benchmarks/kernels/benchmark_2d_silu_mul_fp8_quant.py]
* '''Lines:''' 1-244

=== Signature ===
<syntaxhighlight lang="python">
def reference(
    input: torch.Tensor,
    act_out: torch.Tensor,
    quant_out: torch.Tensor,
    use_ue8m0: bool,
) -> tuple[torch.Tensor, torch.Tensor]

def bench_impl(
    bench_tensors: list[BenchmarkTensors],
    impl_type: ImplType
) -> TMeasurement

def run(
    Ts: list[int],
    Ns: list[int],
    arg_pool_size: int
) -> list[TMeasurement]
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Run comprehensive benchmark
python benchmarks/kernels/benchmark_2d_silu_mul_fp8_quant.py

# Sweeps token counts: 128*i (i=1..15) + 2048*i (i=1..64)
# Hidden dimensions: 2048, 4096, 8192
# Uses CUDA graph with arg_pool_size=8
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| input || torch.Tensor || Input tensor (T, N) where N is divisible by 256
|-
| output || torch.Tensor || Fused output (T, N//2) in FP8
|-
| use_ue8m0 || bool || Use DeepGEMM E8M0 scale format
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| quantized_output || torch.Tensor || FP8 quantized result (T, N//2)
|-
| scales || torch.Tensor || Per-group FP8 scales in column-major layout
|-
| measurements || list[TMeasurement] || Timing comparisons for fused vs reference
|}

== Usage Examples ==

<syntaxhighlight lang="python">
import torch
from benchmarks.kernels.benchmark_2d_silu_mul_fp8_quant import (
    BenchmarkTensors,
    ImplType,
    bench_impl
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    silu_mul_per_token_group_quant_fp8_colmajor
)

# Create benchmark tensors
T, N = 1024, 4096
GROUP_SIZE = 128
assert T % GROUP_SIZE == 0
assert N % (GROUP_SIZE * 2) == 0

bench_tensor = BenchmarkTensors.make(T, N)

# Run fused implementation
output, scales = silu_mul_per_token_group_quant_fp8_colmajor(
    input=bench_tensor.input,
    output=bench_tensor.output,
    use_ue8m0=False
)

# Run reference (unfused)
act_out = torch.empty((T, N // 2), dtype=torch.bfloat16, device="cuda")
quant_out = torch.empty((T, N // 2), device="cuda").to(torch.float8_e4m3fn)

# First: SiLU and multiply
torch.ops._C.silu_and_mul(act_out, bench_tensor.input)

# Then: Per-token group quantization
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _per_token_group_quant_fp8_colmajor
)
x_q, x_s = _per_token_group_quant_fp8_colmajor(
    act_out, quant_out, use_ue8m0=False
)

# Compare correctness
torch.testing.assert_close(output.to(torch.float32), x_q.to(torch.float32))
torch.testing.assert_close(scales, x_s)

# Benchmark both implementations
bench_tensors = [BenchmarkTensors.make(T, N) for _ in range(8)]
fused_timer = bench_impl(
    bench_tensors,
    ImplType.SILU_MUL_PER_TOKEN_GROUP_QUANT_FP8_COLMAJOR
)
ref_timer = bench_impl(bench_tensors, ImplType.REFERENCE)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[related::Concept:KernelFusion]]
* [[related::Concept:MLPGating]]
