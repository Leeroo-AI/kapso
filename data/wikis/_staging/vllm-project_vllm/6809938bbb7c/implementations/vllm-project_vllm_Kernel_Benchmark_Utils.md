{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Kernel Benchmarking]], [[domain::Utilities]], [[domain::CUDA Graphs]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Shared utility classes for kernel benchmarking with support for CUDA graphs, argument pooling, and automated timing.

=== Description ===
This module provides the Bench class and supporting utilities for consistent kernel benchmarking across vLLM. It wraps PyTorch's benchmark utilities with enhanced features including CUDA graph support for reduced kernel launch overhead, argument pooling to test kernels with varying inputs without cache pollution, and automated timing with confidence intervals. The ArgPool class enables testing kernels with different input tensors on each iteration to simulate real-world cache behavior. The Bench class automatically handles warmup, timing, and result validation.

=== Usage ===
Use these utilities when creating new kernel benchmarks to ensure consistent methodology, when testing with CUDA graphs to measure true kernel performance, or when benchmarking with varying inputs to avoid unrealistic cache hits.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/utils.py benchmarks/kernels/utils.py]

=== Signature ===
<syntaxhighlight lang="python">
@dataclasses.dataclass
class CudaGraphBenchParams:
    num_ops_in_cuda_graph: int

@dataclasses.dataclass
class ArgPool:
    values: Iterable[Any]

class Bench:
    def __init__(self, cuda_graph_params: CudaGraphBenchParams | None,
                 label: str, sub_label: str, description: str,
                 fn: Callable, *args, **kwargs)

    def run(self) -> TMeasurement
    def run_eager(self) -> TMeasurement
    def run_cudagrah(self) -> TMeasurement
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from benchmarks.kernels.utils import Bench, ArgPool, CudaGraphBenchParams
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| fn || Callable || Kernel function to benchmark
|-
| args || tuple || Positional arguments for fn
|-
| kwargs || dict || Keyword arguments for fn
|-
| label || str || Benchmark category label
|-
| sub_label || str || Specific configuration label
|-
| description || str || Detailed description
|-
| cuda_graph_params || CudaGraphBenchParams | None || Enable CUDA graph benchmarking
|-
| min_run_time || float || Minimum benchmark duration in seconds
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| measurement || TMeasurement || PyTorch benchmark measurement with timing statistics
|-
| mean_time || float || Average execution time
|-
| median_time || float || Median execution time
|-
| iqr || float || Interquartile range
|-
| meets_confidence || bool || Whether results meet statistical confidence
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Basic benchmark
from benchmarks.kernels.utils import Bench
import torch

def my_kernel(x, y):
    return x @ y

with Bench(
    cuda_graph_params=None,
    label="matmul",
    sub_label="shape_1024x1024",
    description="Matrix multiplication benchmark",
    fn=my_kernel,
    x=torch.randn(1024, 1024, device="cuda"),
    y=torch.randn(1024, 1024, device="cuda")
) as bench:
    result = bench.run()
    print(f"Mean time: {result.mean:.3f}ms")

# With argument pooling (different inputs each iteration)
from benchmarks.kernels.utils import ArgPool

x_pool = ArgPool([
    torch.randn(1024, 1024, device="cuda") for _ in range(10)
])
y_pool = ArgPool([
    torch.randn(1024, 1024, device="cuda") for _ in range(10)
])

with Bench(
    cuda_graph_params=None,
    label="matmul_pooled",
    sub_label="shape_1024x1024",
    description="Matmul with varying inputs",
    fn=my_kernel,
    x=x_pool,
    y=y_pool
) as bench:
    result = bench.run()

# With CUDA graphs (reduced launch overhead)
from benchmarks.kernels.utils import CudaGraphBenchParams

with Bench(
    cuda_graph_params=CudaGraphBenchParams(num_ops_in_cuda_graph=10),
    label="matmul_cudagraph",
    sub_label="shape_1024x1024",
    description="Matmul with CUDA graphs",
    fn=my_kernel,
    x=torch.randn(1024, 1024, device="cuda"),
    y=torch.randn(1024, 1024, device="cuda")
) as bench:
    result = bench.run()  # Automatically uses CUDA graph
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[Concept:CUDA_Graphs]]
* [[Pattern:Kernel_Benchmarking]]
* [[Tool:PyTorch_Benchmark]]
