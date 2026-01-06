{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::Activation]], [[domain::Kernels]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmark comparing custom activation kernels against torch.compile implementations.

=== Description ===
This benchmark evaluates various custom activation function implementations in vLLM's CustomOp registry against PyTorch 2.0+ compiled versions. It tests fused activation patterns commonly used in transformer MLP layers including silu_and_mul, gelu_and_mul, fatrelu_and_mul, and single activations like gelu_new, gelu_fast, and quick_gelu. The benchmark measures performance across different batch sizes, sequence lengths, and intermediate dimensions typical of transformer architectures.

The CustomOp implementations use optimized CUDA kernels for fused operations, while the compiled baseline uses torch.compile with fullgraph=True. Results help determine when custom kernels provide advantages over compiler-generated code for activation functions.

=== Usage ===
Run when evaluating activation function performance in transformer models, particularly for choosing between custom ops and compiled PyTorch code.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_activation.py benchmarks/kernels/benchmark_activation.py]
* '''Lines:''' 1-105

=== Signature ===
<syntaxhighlight lang="python">
def benchmark_activation(
    batch_size: int,
    seq_len: int,
    intermediate_size: int,
    provider: str,
    func_name: str,
    dtype: torch.dtype,
)
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Benchmark SiLU and mul (default)
python benchmarks/kernels/benchmark_activation.py

# Benchmark specific activation
python benchmarks/kernels/benchmark_activation.py \
    --func-name gelu_and_mul \
    --dtype bfloat16

# Test GELU variants
python benchmarks/kernels/benchmark_activation.py --func-name gelu_new
python benchmarks/kernels/benchmark_activation.py --func-name gelu_fast
python benchmarks/kernels/benchmark_activation.py --func-name quick_gelu

# Test tanh approximation GELU
python benchmarks/kernels/benchmark_activation.py --func-name gelu_and_mul_tanh

# Test FATReLU
python benchmarks/kernels/benchmark_activation.py --func-name fatrelu_and_mul

# Available functions:
# - mul_and_silu, silu_and_mul
# - gelu_and_mul, gelu_and_mul_tanh
# - fatrelu_and_mul, swigluoai_and_mul
# - gelu_new, gelu_fast, quick_gelu
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| batch_size || int || Batch size (1, 16, or 128)
|-
| seq_len || int || Sequence length (1, 16, 64, 1024, or 4096)
|-
| intermediate_size || int || Hidden dimension (3072, 9728, or 12288)
|-
| provider || str || Implementation ("custom" or "compiled")
|-
| func_name || str || Activation function name
|-
| dtype || torch.dtype || Data type (half, bfloat16, or float)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| latency_ms || float || Latency in milliseconds (median)
|-
| latency_min_ms || float || Minimum latency (20th percentile)
|-
| latency_max_ms || float || Maximum latency (80th percentile)
|}

== Usage Examples ==

<syntaxhighlight lang="python">
import torch
from vllm.model_executor.custom_op import CustomOp

device = "cuda"
batch_size, seq_len = 16, 1024
intermediate_size = 4096
num_tokens = batch_size * seq_len
dtype = torch.bfloat16

# Create input
x = torch.randn(num_tokens, intermediate_size, dtype=dtype, device=device)

# SiLU and mul (gated MLP pattern)
silu_mul_layer = CustomOp.op_registry["silu_and_mul"]()
output_silu = silu_mul_layer(x)  # Output: (num_tokens, intermediate_size // 2)

# GELU and mul with exact GELU
gelu_mul_layer = CustomOp.op_registry["gelu_and_mul"](approximate="none")
output_gelu = gelu_mul_layer(x)

# GELU and mul with tanh approximation
gelu_tanh_layer = CustomOp.op_registry["gelu_and_mul"](approximate="tanh")
output_gelu_tanh = gelu_tanh_layer(x)

# FATReLU and mul
threshold = 0.5
fatrelu_layer = CustomOp.op_registry["fatrelu_and_mul"](threshold)
output_fatrelu = fatrelu_layer(x)

# Single activations (no mul)
gelu_new_layer = CustomOp.op_registry["gelu_new"]()
output_gelu_new = gelu_new_layer(x)

# Compiled baseline
compiled_silu = torch.compile(silu_mul_layer.forward_native)
output_compiled = compiled_silu(x)

# Compare performance using triton.testing.do_bench_cudagraph
from vllm.triton_utils import triton
ms_custom, _, _ = triton.testing.do_bench_cudagraph(
    lambda: silu_mul_layer(x), quantiles=[0.5, 0.2, 0.8]
)
ms_compiled, _, _ = triton.testing.do_bench_cudagraph(
    lambda: compiled_silu(x), quantiles=[0.5, 0.2, 0.8]
)
print(f"Custom: {ms_custom:.3f}ms, Compiled: {ms_compiled:.3f}ms")
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
