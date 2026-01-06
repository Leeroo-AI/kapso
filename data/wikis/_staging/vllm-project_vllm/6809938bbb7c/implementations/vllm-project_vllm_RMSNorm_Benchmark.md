{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Kernel Benchmarking]], [[domain::Normalization]], [[domain::Performance Testing]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
A benchmark comparing three RMSNorm implementations (HuggingFace, FlashInfer, vLLM) across different batch sizes and sequence lengths.

=== Description ===
This benchmark script evaluates RMSNorm (Root Mean Square Layer Normalization) kernel performance across three implementations: a naive HuggingFace reference, FlashInfer's optimized version, and vLLM's custom implementation. It tests both with and without residual connections, measuring execution time in microseconds and generating performance comparison plots using Triton's benchmarking framework. The script validates correctness by comparing outputs across implementations before running performance tests.

=== Usage ===
Use this benchmark to compare RMSNorm kernel performance when optimizing model inference, selecting the best implementation for specific hardware, or validating custom RMSNorm kernels against reference implementations.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_rmsnorm.py benchmarks/kernels/benchmark_rmsnorm.py]

=== Signature ===
<syntaxhighlight lang="python">
def rmsnorm_naive(x: torch.Tensor, weight: torch.Tensor,
                  residual: torch.Tensor | None = None, eps: float = 1e-6)

def rmsnorm_flashinfer(x: torch.Tensor, weight: torch.Tensor,
                       residual: torch.Tensor | None = None, eps: float = 1e-6)

def rmsnorm_vllm(x: torch.Tensor, weight: torch.Tensor,
                 residual: torch.Tensor | None = None, eps: float = 1e-6)

def get_benchmark(use_residual: bool) -> Callable
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Run as script
python benchmarks/kernels/benchmark_rmsnorm.py \
    --batch-size 4 --seq-len 128 --hidden-size 4096 \
    --use-residual --save-path ./configs/rmsnorm/
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| x || torch.Tensor || Input tensor of shape (batch_size, seq_len, hidden_size)
|-
| weight || torch.Tensor || Normalization weights of shape (hidden_size,)
|-
| residual || torch.Tensor | None || Optional residual tensor for fused add+norm
|-
| eps || float || Small value for numerical stability (default 1e-6)
|-
| --batch-size || int || Batch size for benchmarking (default 4)
|-
| --seq-len || int || Sequence length (default 128)
|-
| --hidden-size || int || Hidden dimension size (default 4096)
|-
| --use-residual || bool || Whether to test with residual connections
|-
| --save-path || str || Directory to save benchmark results
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| normalized_output || torch.Tensor || RMSNorm output tensor
|-
| residual_output || torch.Tensor (optional) || Updated residual if provided
|-
| benchmark_plots || File || Performance comparison plots saved to save-path
|-
| performance_data || CSV || Execution time metrics in microseconds
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Run benchmark with residual connections
python benchmarks/kernels/benchmark_rmsnorm.py \
    --batch-size 4 \
    --seq-len 128 \
    --hidden-size 4096 \
    --use-residual \
    --save-path ./rmsnorm_results/

# Test correctness for specific configuration
from benchmark_rmsnorm import calculate_diff
calculate_diff(batch_size=2, seq_len=64, hidden_size=2048, use_residual=True)

# Use individual implementations
import torch
from benchmark_rmsnorm import rmsnorm_vllm

x = torch.randn(4, 128, 4096, dtype=torch.bfloat16, device="cuda")
weight = torch.ones(4096, dtype=torch.bfloat16, device="cuda")
residual = torch.randn_like(x)

output, updated_residual = rmsnorm_vllm(x, weight, residual, eps=1e-6)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[Implementation:vllm_custom_ops]]
* [[Concept:RMSNorm]]
* [[Benchmark:Kernel_Performance_Testing]]
