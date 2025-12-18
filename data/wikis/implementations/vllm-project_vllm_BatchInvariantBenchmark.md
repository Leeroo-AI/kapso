{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::Attention]], [[domain::Performance]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmark measuring the performance overhead of VLLM_BATCH_INVARIANT mode on Hopper GPUs.

=== Description ===
This benchmark evaluates the computational overhead introduced by vLLM's batch invariant mode, which ensures deterministic outputs regardless of batch composition. The benchmark runs identical workloads twice - once with batch invariance disabled (baseline) and once enabled - then compares initialization time, trial execution time, and token throughput. It uses a needle-in-haystack approach where one consistent prompt is embedded in randomly generated prompts to validate output consistency.

The benchmark requires CUDA compute capability 9.0 (Hopper) or higher and supports configurable parameters via environment variables including model selection, tensor parallelism size, batch size, number of trials, and attention backend. Results include detailed metrics on overhead percentage, throughput changes, and timing comparisons.

=== Usage ===
Run when testing batch invariance overhead on Hopper GPUs, particularly for models like Qwen3 or DeepSeek-V3 with tensor parallelism configurations.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_batch_invariance.py benchmarks/benchmark_batch_invariance.py]
* '''Lines:''' 1-380

=== Signature ===
<syntaxhighlight lang="python">
def run_benchmark_with_batch_invariant(
    model: str,
    tp_size: int,
    max_batch_size: int,
    num_trials: int,
    min_prompt: int,
    max_prompt: int,
    max_tokens: int,
    temperature: float,
    gpu_mem_util: float,
    max_model_len: int,
    backend: str,
    batch_invariant: bool,
    seed: int = 12345,
) -> dict
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Benchmark Qwen3 (default)
python benchmarks/benchmark_batch_invariance.py

# Benchmark DeepSeek-V3 with 8 GPUs
VLLM_BENCH_MODEL="deepseek-ai/DeepSeek-V3" VLLM_BENCH_TP_SIZE=8 \
    python benchmarks/benchmark_batch_invariance.py

# Quick test with fewer trials
VLLM_BENCH_NUM_TRIALS=2 VLLM_BENCH_BATCH_SIZE=32 \
    python benchmarks/benchmark_batch_invariance.py

# Custom configuration
VLLM_BENCH_MODEL="Qwen/Qwen3-1.7B" \
VLLM_BENCH_BATCH_SIZE=128 \
VLLM_BENCH_NUM_TRIALS=5 \
VLLM_BENCH_MAX_TOKENS=128 \
VLLM_BENCH_BACKEND=FLASH_ATTN \
    python benchmarks/benchmark_batch_invariance.py
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| VLLM_BENCH_MODEL || str || Model to benchmark (default: "Qwen/Qwen3-1.7B")
|-
| VLLM_BENCH_TP_SIZE || int || Tensor parallel size (default: 1, use 8 for DeepSeek)
|-
| VLLM_BENCH_BATCH_SIZE || int || Max batch size (default: 128)
|-
| VLLM_BENCH_NUM_TRIALS || int || Number of trials (default: 5)
|-
| VLLM_BENCH_MIN_PROMPT || int || Min prompt length in words (default: 1024)
|-
| VLLM_BENCH_MAX_PROMPT || int || Max prompt length in words (default: 2048)
|-
| VLLM_BENCH_MAX_TOKENS || int || Max tokens to generate (default: 128)
|-
| VLLM_BENCH_BACKEND || str || Attention backend (default: FLASH_ATTN)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| metrics || dict || Contains init_time, avg_time, throughput, overhead percentages, and trial_times
|}

== Usage Examples ==

<syntaxhighlight lang="python">
from benchmarks.benchmark_batch_invariance import run_benchmark_with_batch_invariant

# Run baseline benchmark (batch invariant off)
baseline_results = run_benchmark_with_batch_invariant(
    model="Qwen/Qwen3-1.7B",
    tp_size=1,
    max_batch_size=128,
    num_trials=5,
    min_prompt=1024,
    max_prompt=2048,
    max_tokens=128,
    temperature=0.0,
    gpu_mem_util=0.4,
    max_model_len=5120,
    backend="FLASH_ATTN",
    batch_invariant=False
)

# Run with batch invariance enabled
batch_inv_results = run_benchmark_with_batch_invariant(
    model="Qwen/Qwen3-1.7B",
    tp_size=1,
    max_batch_size=128,
    num_trials=5,
    min_prompt=1024,
    max_prompt=2048,
    max_tokens=128,
    temperature=0.0,
    gpu_mem_util=0.4,
    max_model_len=5120,
    backend="FLASH_ATTN",
    batch_invariant=True
)

# Calculate overhead
overhead_pct = (
    (batch_inv_results["avg_time"] - baseline_results["avg_time"])
    / baseline_results["avg_time"] * 100
)
print(f"Batch invariance overhead: {overhead_pct:.1f}%")
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[requires::Concept:VLLM_BatchInvariance]]
