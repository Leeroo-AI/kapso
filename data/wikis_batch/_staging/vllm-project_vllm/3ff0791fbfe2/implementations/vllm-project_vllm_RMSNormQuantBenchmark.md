{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::Quantization]], [[domain::Kernels]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==
Performance benchmark for fused RMSNorm plus dynamic quantization kernels.

=== Description ===
This benchmark evaluates the performance benefits of fusing RMSNorm (Root Mean Square Layer Normalization) with quantization operations. It compares unfused implementations (separate RMSNorm and quantization kernels) against fused implementations for both INT8 and FP8 quantization, including per-token and per-token-group (blockwise) variants. The script tests across a wide parameter space: batch sizes from 1 to 2048 tokens, hidden dimensions from 1024 to 8192, with and without residual addition, using BF16 or FP32 data types, and group sizes of 64 or 128.

=== Usage ===
Use this benchmark to validate that fused RMSNorm+quantization kernels provide significant performance improvements over unfused implementations. This is critical for efficient quantized inference pipelines where every millisecond of latency reduction matters.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/fused_kernels/layernorm_rms_benchmarks.py#L1-L310 benchmarks/fused_kernels/layernorm_rms_benchmarks.py]
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

def bench(params: bench_params_t, label: str, sub_label: str) -> Iterable[TMeasurement]

def unfused_int8_impl(
    rms_norm_layer: RMSNorm,
    x: torch.Tensor,
    residual: torch.Tensor | None,
    quant_dtype: torch.dtype,
    group_size: list[int],
)

def fused_impl(
    rms_norm_layer: RMSNorm,
    x: torch.Tensor,
    residual: torch.Tensor | None,
    quant_dtype: torch.dtype,
    group_size: list[int],
)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# This is a standalone benchmark script
python benchmarks/fused_kernels/layernorm_rms_benchmarks.py
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| NUM_TOKENS || list[int] || Internal || Batch sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
|-
| HIDDEN_SIZES || list[int] || Internal || Hidden dimensions: [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
|-
| ADD_RESIDUAL || list[bool] || Internal || Test with/without residual: [True, False]
|-
| DTYPES || list[torch.dtype] || Internal || Data types: [torch.bfloat16, torch.float]
|-
| GROUP_SIZES || list[list[int]] || Internal || Groupwise quantization: [[1, 64], [1, 128]]
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| measurements || list[TMeasurement] || PyTorch benchmark measurements for all configurations
|-
| pkl_file || file || Pickled results: rms_norm_dpt_quant-{timestamp}.pkl
|-
| stdout || text || Formatted comparison tables for each configuration
|}

== Usage Examples ==
<syntaxhighlight lang="python">
# Example 1: Run full benchmark suite
# Tests all combinations of parameters (can take hours)
python benchmarks/fused_kernels/layernorm_rms_benchmarks.py

# Example 2: Analyze results from a previous run
import pickle
import torch.utils.benchmark as TBenchmark

with open('rms_norm_dpt_quant-1234567890.pkl', 'rb') as f:
    measurements = pickle.load(f)

# Filter to specific configuration
bf16_2048_no_residual = [
    m for m in measurements
    if 'N 2048' in m.sub_label
    and 'D 4096' in m.sub_label
    and 'R False' in m.sub_label
    and 'bf16' in m.sub_label
]

compare = TBenchmark.Compare(bf16_2048_no_residual)
compare.print()

# Example 3: Custom benchmark with specific parameters
from benchmarks.fused_kernels.layernorm_rms_benchmarks import (
    bench_params_t, bench
)

# Test only the configurations you care about
params = bench_params_t(
    num_tokens=1024,
    hidden_size=4096,
    add_residual=True,
    dtype=torch.bfloat16,
    group_size=[1, 128],
)

results = bench(
    params,
    "rms-norm-dynamic-per-token-quant",
    params.description()
)

# Example 4: Compare fused vs unfused for specific case
import torch
from vllm.model_executor.layers.layernorm import RMSNorm

hidden_size = 4096
num_tokens = 512
dtype = torch.bfloat16

# Setup
layer = RMSNorm(hidden_size, 1e-6).to(dtype=dtype)
layer.weight.data.normal_(mean=1.0, std=0.1)

x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda") * (1/hidden_size)
residual = torch.randn_like(x) * (1/hidden_size)

# Benchmark unfused INT8
from benchmarks.fused_kernels.layernorm_rms_benchmarks import (
    unfused_int8_impl, fused_impl
)
import torch.utils.benchmark as TBenchmark

unfused_timer = TBenchmark.Timer(
    stmt="unfused_int8_impl(layer, x, residual, torch.int8, [1, 128])",
    globals={'unfused_int8_impl': unfused_int8_impl, 'layer': layer,
             'x': x, 'residual': residual, 'torch': torch}
)

fused_timer = TBenchmark.Timer(
    stmt="fused_impl(layer, x, residual, torch.int8, [1, 128])",
    globals={'fused_impl': fused_impl, 'layer': layer,
             'x': x, 'residual': residual, 'torch': torch}
)

print("Unfused:", unfused_timer.blocked_autorange(min_run_time=1))
print("Fused:", fused_timer.blocked_autorange(min_run_time=1))

# Expected output format:
# ================================================================================
# rms-norm-dynamic-per-token-quant | N 512 x D 4096 x R True x DT torch.bfloat16 x GS [1, 128]
# ================================================================================
# Description                          |   Time (mean +/- std)     | Speedup
# --------------------------------------------------------------------------------
# unfused_int8_impl                    |   X.XXXe-XX +/- X.XXXe-XX |   1.00x
# unfused_fp8_impl                     |   X.XXXe-XX +/- X.XXXe-XX |   X.XXx
# fused_int8_impl                      |   X.XXXe-XX +/- X.XXXe-XX |   X.XXx
# fused_fp8_impl                       |   X.XXXe-XX +/- X.XXXe-XX |   X.XXx
# unfused_groupwise_fp8_impl           |   X.XXXe-XX +/- X.XXXe-XX |   X.XXx
# fused_groupwise_fp8_impl             |   X.XXXe-XX +/- X.XXXe-XX |   X.XXx

# Typical speedup: 1.3-1.8x for fused vs unfused
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_CUDA_Environment]]
