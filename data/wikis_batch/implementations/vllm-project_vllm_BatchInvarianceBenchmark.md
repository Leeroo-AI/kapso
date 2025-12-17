{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::Performance]], [[domain::Testing]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==
Benchmark script to measure the performance overhead of VLLM_BATCH_INVARIANT mode.

=== Description ===
This executable script measures the performance impact of enabling batch-invariant execution mode in vLLM. It runs identical workloads twice - once with VLLM_BATCH_INVARIANT=0 (baseline) and once with VLLM_BATCH_INVARIANT=1 (batch invariant mode) - and compares initialization time, trial time, and throughput metrics. The script uses a "needle prompt" approach to verify output consistency: a known prompt is injected at a random position in each batch to ensure deterministic behavior. Requires CUDA with SM90+ (Hopper architecture).

=== Usage ===
Use this benchmark when validating the performance cost of enabling batch-invariant mode, which ensures deterministic outputs regardless of batch composition. This is critical for applications requiring reproducible results across different batch configurations.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_batch_invariance.py#L1-L380 benchmarks/benchmark_batch_invariance.py]
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

=== Import ===
<syntaxhighlight lang="python">
# This is a standalone benchmark script
python benchmarks/benchmark_batch_invariance.py
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| VLLM_BENCH_MODEL || str (env) || No || Model to benchmark (default: "Qwen/Qwen3-1.7B")
|-
| VLLM_BENCH_TP_SIZE || int (env) || No || Tensor parallel size (default: 1, use 8 for deepseek)
|-
| VLLM_BENCH_BATCH_SIZE || int (env) || No || Max batch size (default: 128)
|-
| VLLM_BENCH_NUM_TRIALS || int (env) || No || Number of trials to run (default: 5)
|-
| VLLM_BENCH_MIN_PROMPT || int (env) || No || Min prompt length in words (default: 1024)
|-
| VLLM_BENCH_MAX_PROMPT || int (env) || No || Max prompt length in words (default: 2048)
|-
| VLLM_BENCH_MAX_TOKENS || int (env) || No || Max tokens to generate (default: 128)
|-
| VLLM_BENCH_TEMPERATURE || float (env) || No || Temperature for sampling (default: 0.0)
|-
| VLLM_BENCH_GPU_MEMORY_UTILIZATION || float (env) || No || GPU memory utilization (default: 0.4)
|-
| VLLM_BENCH_MAX_MODEL_LEN || int (env) || No || Max model length (default: 5120)
|-
| VLLM_BENCH_BACKEND || str (env) || No || Attention backend (default: FLASH_ATTN)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| init_time || float || Engine initialization time in seconds
|-
| avg_time || float || Average time per trial in seconds
|-
| min_time || float || Minimum trial time in seconds
|-
| max_time || float || Maximum trial time in seconds
|-
| total_tokens || int || Total tokens generated across all trials
|-
| total_prompts || int || Total prompts processed across all trials
|-
| throughput || float || Tokens per second throughput
|-
| prompts_per_sec || float || Prompts per second throughput
|-
| comparison_metrics || dict || Overhead percentages and throughput changes
|}

== Usage Examples ==
<syntaxhighlight lang="python">
# Example 1: Default benchmark (Qwen3-1.7B)
# Run from repository root:
python benchmarks/benchmark_batch_invariance.py

# Example 2: Benchmark DeepSeek-V3 with 8 GPUs
# Requires 8 GPUs with Hopper+ architecture
VLLM_BENCH_MODEL="deepseek-ai/DeepSeek-V3" \
VLLM_BENCH_TP_SIZE=8 \
python benchmarks/benchmark_batch_invariance.py

# Example 3: Quick test with fewer trials and smaller batch
VLLM_BENCH_NUM_TRIALS=2 \
VLLM_BENCH_BATCH_SIZE=32 \
VLLM_BENCH_MAX_TOKENS=64 \
python benchmarks/benchmark_batch_invariance.py

# Example 4: Custom model with specific memory settings
VLLM_BENCH_MODEL="meta-llama/Llama-2-7b-hf" \
VLLM_BENCH_GPU_MEMORY_UTILIZATION=0.8 \
VLLM_BENCH_MAX_MODEL_LEN=4096 \
VLLM_BENCH_MIN_PROMPT=512 \
VLLM_BENCH_MAX_PROMPT=1024 \
python benchmarks/benchmark_batch_invariance.py

# Example 5: Test with different attention backend
VLLM_BENCH_BACKEND="XFORMERS" \
VLLM_BENCH_BATCH_SIZE=64 \
python benchmarks/benchmark_batch_invariance.py

# Expected output structure:
# ===========================================================================
# VLLM BATCH INVARIANCE BENCHMARK
# ===========================================================================
# Configuration: [displays all settings]
#
# ===========================================================================
# PHASE 1: Running WITHOUT batch invariance (baseline)
# ===========================================================================
# Engine initialization time: XX.XXs
# Trial 1/5: batch_size=XX, time=X.XXs
# ...
# Average time per trial: X.XXs
# Throughput: XXX.XX tokens/s
#
# ===========================================================================
# PHASE 2: Running WITH batch invariance
# ===========================================================================
# [Similar output]
#
# ===========================================================================
# COMPARISON: Batch Invariance vs Baseline
# ===========================================================================
# Initialization Time:
#   Baseline:         X.XXs
#   Batch Invariant:  X.XXs
#   Overhead:         +X.XX%
#
# Average Trial Time:
#   Baseline:         X.XXs
#   Batch Invariant:  X.XXs
#   Overhead:         +X.XX%
#
# Throughput (tokens/s):
#   Baseline:         XXX.XX
#   Batch Invariant:  XXX.XX
#   Change:           +/-X.XX%
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_CUDA_Environment]]
