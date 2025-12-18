{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::Distributed]], [[domain::Communication]], [[domain::NCCL]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmarks inter-GPU communication performance for different device communicator implementations including CustomAllreduce, PyNcclCommunicator, and SymmMemCommunicator.

=== Description ===
This benchmark script compares the performance of multiple device communicator implementations for all-reduce operations in distributed LLM inference. It tests CustomAllreduce (oneshot and twoshot variants), PyNcclCommunicator, PyNccl with symmetric memory, SymmMemCommunicator (multimem and two-shot modes). All benchmarks use CUDA graphs to capture 10 cycles of the operation for accurate performance measurement.

The benchmark runs with torchrun for multi-GPU setups and measures latency across different tensor sizes (sequence lengths). For NCCL symmetric memory benchmarks, environment variables NCCL_NVLS_ENABLE=1, NCCL_CUMEM_ENABLE=1, and VLLM_USE_NCCL_SYMM_MEM=1 must be set to enable fast NVLS implementation. Results show timing comparisons and speedup relative to PyNccl baseline, helping identify the fastest communicator for each tensor size and world size configuration.

=== Usage ===
Run this benchmark when evaluating distributed inference performance, comparing communication backends for tensor parallelism, or tuning all-reduce operations for different model sizes and GPU configurations.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_device_communicators.py benchmarks/kernels/benchmark_device_communicators.py]

=== Signature ===
<syntaxhighlight lang="python">
def benchmark_allreduce(
    self,
    sequence_length: int,
    num_warmup: int,
    num_trials: int
) -> dict[str, float]
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Run with 2 GPUs, default sequence lengths
torchrun --nproc_per_node=2 benchmarks/kernels/benchmark_device_communicators.py

# Custom sequence lengths and iterations
torchrun --nproc_per_node=4 benchmarks/kernels/benchmark_device_communicators.py \
    --sequence-lengths 512 1024 2048 4096 \
    --num-warmup 10 \
    --num-trials 100

# Save results to JSON
torchrun --nproc_per_node=8 benchmarks/kernels/benchmark_device_communicators.py \
    --output-json results.json
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| sequence_lengths || list[int] || Sequence lengths to benchmark (default: 128-8192)
|-
| num_warmup || int || Number of warmup iterations (default: 5)
|-
| num_trials || int || Number of benchmark trials (default: 50)
|-
| output_json || str || Optional JSON output file path
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| results || dict[int, dict[str, float]] || Latency in ms per communicator per sequence length
|-
| speedup_info || str || Fastest communicator and speedup vs PyNccl
|-
| formatted_table || str || Console table with all results
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Internal benchmark structure
class CommunicatorBenchmark:
    def benchmark_allreduce_single(self, sequence_length, allreduce_fn,
                                   should_use_fn, context,
                                   num_warmup, num_trials):
        tensor = torch.randn(sequence_length, HIDDEN_SIZE,
                           dtype=BENCHMARK_DTYPE, device=self.device)

        # Capture CUDA graph
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(CUDA_GRAPH_CAPTURE_CYCLES):
                allreduce_fn(tensor)

        # Benchmark graph replay
        for _ in range(num_trials):
            graph.replay()

        return latency_ms

# Usage
benchmark = CommunicatorBenchmark(rank, world_size, device,
                                 cpu_group, sequence_lengths)
results = benchmark.benchmark_allreduce(1024, num_warmup=5, num_trials=50)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[Implementation:benchmark_fused_collective]]
* [[Concept:Tensor_Parallelism]]
* [[Concept:NCCL]]
* [[Concept:All_Reduce]]
