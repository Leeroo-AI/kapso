{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::Kernels]], [[domain::Normalization]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmark for fused RMSNorm with dynamic per-token quantization kernels.

=== Description ===
This benchmark evaluates the performance of fused RMSNorm layer normalization combined with dynamic quantization to INT8 or FP8. It compares unfused implementations (separate normalization and quantization) against fused kernels that combine both operations into a single pass. The benchmark tests both per-token quantization and per-block (groupwise) quantization with configurable group sizes.

The benchmark sweeps across various configurations including different token counts (1 to 1024), hidden dimensions (1024 to 8192), data types (bfloat16, float32), and residual connection handling. It measures the performance advantage of kernel fusion, which eliminates intermediate memory reads/writes and improves memory bandwidth utilization.

=== Usage ===
Run when optimizing RMSNorm + quantization fusion for transformer models, particularly for evaluating memory-bound kernel performance improvements.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/fused_kernels/layernorm_rms_benchmarks.py benchmarks/fused_kernels/layernorm_rms_benchmarks.py]
* '''Lines:''' 1-310

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class bench_params_t:
    num_tokens: int
    hidden_size: int
    add_residual: bool
    dtype: torch.dtype
    group_size: list[int]

def bench(
    params: bench_params_t,
    label: str,
    sub_label: str
) -> Iterable[TMeasurement]
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Run comprehensive benchmark
python benchmarks/fused_kernels/layernorm_rms_benchmarks.py

# Results saved to rms_norm_dpt_quant-{timestamp}.pkl
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| num_tokens || int || Number of tokens (2^0 to 2^10)
|-
| hidden_size || int || Hidden dimension (1024 to 8129, step 1024)
|-
| add_residual || bool || Whether to include residual connection
|-
| dtype || torch.dtype || Data type (torch.bfloat16 or torch.float)
|-
| group_size || list[int] || Group sizes for blockwise quantization [1, 64] or [1, 128]
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| measurements || list[TMeasurement] || Timing results for unfused vs fused implementations
|-
| pickle_file || .pkl file || Serialized benchmark results
|}

== Usage Examples ==

<syntaxhighlight lang="python">
import torch
from benchmarks.fused_kernels.layernorm_rms_benchmarks import (
    bench_params_t,
    bench,
    fused_impl,
    unfused_int8_impl
)
from vllm.model_executor.layers.layernorm import RMSNorm

# Setup benchmark parameters
params = bench_params_t(
    num_tokens=512,
    hidden_size=4096,
    add_residual=False,
    dtype=torch.bfloat16,
    group_size=[1, 64]  # Per-token with 64-element groups
)

# Create RMSNorm layer
layer = RMSNorm(params.hidden_size, eps=1e-6).to(dtype=params.dtype)
x = torch.randn(params.num_tokens, params.hidden_size,
                dtype=params.dtype, device="cuda")

# Run fused implementation
out_fused, scale_fused = fused_impl(
    layer, x, residual=None,
    quant_dtype=torch.float8_e4m3fn,
    group_size=params.group_size
)

# Run unfused implementation
out_unfused, scale_unfused = unfused_fp8_impl(
    layer, x, residual=None,
    quant_dtype=torch.float8_e4m3fn,
    group_size=params.group_size
)

# Run full benchmark
timers = bench(params, "rms-norm-quant", params.description())
# Results show speedup of fused kernels over unfused
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
