{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::Quantization]], [[domain::FP8]], [[domain::Kernels]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmark comparing Torch compiled, CUDA, and Triton implementations of per-token FP8 quantization.

=== Description ===
This benchmark evaluates dynamic FP8 quantization kernels that compute per-token or per-group scales for input activations. It compares three implementations: Torch compiled (PyTorch 2.0+ with torch.compile and Inductor), custom CUDA kernels, and Triton kernels. The benchmark tests various scaling granularities including per-tensor, per-token, and per-group with configurable group sizes (64, 128).

The benchmark also supports column-major scale layout testing, which is important for certain GEMM kernel implementations. It includes correctness checking against reference implementations and computes geometric mean speedups across different configurations. The QuantFP8 module supports both dynamic quantization (computing scales on-the-fly) and using pre-computed scales.

=== Usage ===
Run when evaluating per-token quantization kernel performance, particularly for comparing compilation strategies and scale layout formats.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/bench_per_token_quant_fp8.py benchmarks/kernels/bench_per_token_quant_fp8.py]
* '''Lines:''' 1-270

=== Signature ===
<syntaxhighlight lang="python">
def benchmark_quantization(
    batch_size: int,
    hidden_size: int,
    provider: str,
    group_shape: GroupShape,
    col_major: bool,
    dtype: torch.dtype,
)

def calculate_diff(
    batch_size: int,
    hidden_size: int,
    group_shape: GroupShape,
    dtype: torch.dtype,
)
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Run full benchmark suite
python benchmarks/kernels/bench_per_token_quant_fp8.py

# Check correctness only
python benchmarks/kernels/bench_per_token_quant_fp8.py --check

# Custom configuration
python benchmarks/kernels/bench_per_token_quant_fp8.py \
    --dtype bfloat16 \
    --hidden-sizes 896 1024 2048 4096 7168 \
    --batch-sizes 1 16 128 512 1024 \
    --group-sizes 0 -1 64 128

# Disable column-major testing
python benchmarks/kernels/bench_per_token_quant_fp8.py --no-column-major

# Group sizes: 0=PER_TENSOR, -1=PER_TOKEN, >0=GroupShape(1,N)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| batch_size || int || Number of tokens
|-
| hidden_size || int || Feature dimension
|-
| provider || str || Implementation ("torch", "cuda", "triton")
|-
| group_shape || GroupShape || Quantization granularity (PER_TENSOR, PER_TOKEN, or GroupShape(1,N))
|-
| col_major || bool || Use column-major scale layout
|-
| dtype || torch.dtype || Input data type
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| latency_us || float || Latency in microseconds (median)
|-
| latency_min_us || float || Minimum latency (20th percentile)
|-
| latency_max_us || float || Maximum latency (80th percentile)
|-
| speedup_table || DataFrame || Geometric mean speedups by configuration
|}

== Usage Examples ==

<syntaxhighlight lang="python">
import torch
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

device = torch.device("cuda")
batch_size, hidden_size = 512, 4096

# Create input
x = torch.randn((batch_size, hidden_size), dtype=torch.bfloat16, device=device)

# Per-token quantization
quant_per_token = QuantFP8(
    dynamic_only=False,
    group_shape=GroupShape.PER_TOKEN,
    column_major_scales=False
)
x_q_token, scale_token = quant_per_token.forward_cuda(x)

# Per-group quantization (group size 64)
quant_per_group = QuantFP8(
    dynamic_only=False,
    group_shape=GroupShape(1, 64),
    column_major_scales=True  # Column-major layout
)
x_q_group, scale_group = quant_per_group.forward_cuda(x)

# Torch compiled version (with dynamic shape marking)
def with_dyn_compile(fn):
    compiled = torch.compile(fn, fullgraph=True, dynamic=False)
    def wrapped(x):
        torch._dynamo.mark_dynamic(x, 0)  # First dim is dynamic
        return compiled(x)
    return wrapped

compiled_quant = with_dyn_compile(quant_per_token.forward_native)
x_q_compiled, scale_compiled = compiled_quant(x)

# Triton implementation (forced via platform patching)
from unittest.mock import patch
with patch("vllm.platforms.current_platform.is_cuda", return_value=False):
    x_q_triton, scale_triton = quant_per_token.forward_cuda(x)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
