{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Performance_Testing]], [[domain::Benchmarking]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

A multi-commit benchmark orchestrator that runs optimum-benchmark tests across different git commits and aggregates performance metrics for comparison.

=== Description ===

The `benchmark.py` module provides a sophisticated wrapper around the `optimum-benchmark` library to enable performance testing across multiple git commits. It allows comparing metrics like latency, throughput, and token generation speed between different versions of code.

The module's core functionality includes:
* **Git commit checkout management** via context manager that safely switches commits and returns to original state
* **Multi-commit benchmarking** with support for comparing current HEAD vs main branch (via "diff" mode)
* **Multi-model benchmarking** to test multiple model configurations in a single run
* **Metric extraction and summarization** from optimum-benchmark's JSON reports
* **Result aggregation** that combines metrics across commits and configurations
* **Hub integration** for uploading benchmark results to HuggingFace datasets

Key components:
* `checkout_commit()`: Context manager for safe git operations
* `summarize()`: Extracts selected metrics from benchmark output directories
* `combine_summaries()`: Aggregates results across all benchmark runs into a unified structure

The orchestrator integrates with Hydra for configuration management and supports parallel execution across different model/commit/config combinations.

=== Usage ===

Use this module when you need to benchmark Transformers models across multiple commits to detect performance regressions, evaluate optimization improvements, or generate comprehensive performance reports. It's particularly useful for CI/CD pipelines that need to track performance metrics over time.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers huggingface_transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/benchmark/benchmark.py benchmark/benchmark.py]
* '''Lines:''' 1-324

=== Signature ===
<syntaxhighlight lang="python">
@contextmanager
def checkout_commit(repo: Repo, commit_id: str):
    """Context manager for safe git checkout operations"""
    pass


def summarize(
    run_dir: str,
    metrics: list[str],
    expand_metrics: bool = False
) -> list[dict]:
    """Extract metrics from benchmark output directories"""
    pass


def combine_summaries(summaries: list[dict]) -> dict:
    """Aggregate summaries into hierarchical structure"""
    pass
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from benchmark import checkout_commit, summarize, combine_summaries
from git import Repo
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| --config-dir || str || Yes || Path to the config directory for optimum-benchmark
|-
| --config-name || str || Yes || Name of the config file (without .yaml extension)
|-
| --commit || str (comma-separated) || No || Commit SHAs or "diff" to compare HEAD vs main
|-
| --metrics || str (comma-separated) || No || Metrics to extract (default: prefill/decode latency and throughput)
|-
| --ensure_empty || bool || No || Create temporary directory for results (default: True)
|-
| --repo_id || str || No || HuggingFace dataset repo ID for uploading results
|-
| --path_in_repo || str || No || Path in the HuggingFace repo for uploaded results
|-
| --token || str || No || HuggingFace API token for uploads
|-
| backend.model || str (in args) || No || Model(s) to benchmark (comma-separated)
|-
| hydra.sweep.dir || str (in args) || No || Directory for multi-run outputs
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| summary.json || JSON file || Per-commit summary of metrics for each run
|-
| summaries.json || JSON file || List of all summaries across all runs
|-
| combined summary.json || JSON file || Hierarchical aggregation by model/config/commit
|-
| benchmark_report.json || JSON file || Full optimum-benchmark report (per run)
|-
| benchmark.json || JSON file || Benchmark configuration (per run)
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Example 1: Basic benchmark on current commit
# Run from transformers repository root
import subprocess

cmd = [
    "python", "benchmark/benchmark.py",
    "--config-dir", "benchmark/config",
    "--config-name", "generation",
    "backend.model=google/gemma-2b"
]
subprocess.run(cmd)


# Example 2: Compare two commits
cmd = [
    "python", "benchmark/benchmark.py",
    "--config-dir", "benchmark/config",
    "--config-name", "generation",
    "--commit", "9b9c7f03da625b13643e99205c691fe046461724,c97ee28b117c0abe8e08891f402065e4df6d72aa",
    "--metrics", "decode.latency.mean,per_token.throughput.value",
    "backend.model=google/gemma-2b"
]
subprocess.run(cmd)


# Example 3: Compare current HEAD vs main branch
cmd = [
    "python", "benchmark/benchmark.py",
    "--config-dir", "benchmark/config",
    "--config-name", "generation",
    "--commit", "diff",  # Special keyword to compare HEAD vs main
    "backend.model=google/gemma-2b"
]
subprocess.run(cmd)


# Example 4: Multi-model, multi-config benchmark with Hydra multirun
cmd = [
    "python", "benchmark/benchmark.py",
    "--config-dir", "benchmark/config",
    "--config-name", "generation",
    "--commit", "9b9c7f03da625b13643e99205c691fe046461724",
    "--metrics", "decode.latency.mean,per_token.latency.mean,per_token.throughput.value",
    "backend.model=google/gemma-2b,gpt2",
    "benchmark.input_shapes.sequence_length=5,7",
    "benchmark.input_shapes.batch_size=1,2",
    "--multirun"  # Hydra multirun mode
]
subprocess.run(cmd)


# Example 5: Programmatic usage with git checkout
from git import Repo
from benchmark import checkout_commit, summarize, combine_summaries
from pathlib import Path

repo = Repo(Path(__file__).parent.parent)

commits = ["main", "dev"]
all_summaries = []

for commit in commits:
    with checkout_commit(repo, commit):
        # Run your benchmark here
        # optimum_benchmark will generate outputs
        pass

    # Summarize results after checkout context exits
    metrics = ["decode.latency.mean", "per_token.throughput.value"]
    summaries = summarize(f"_benchmark/commit={commit}", metrics)
    all_summaries.extend(summaries)

# Combine all results
combined = combine_summaries(all_summaries)
print(combined)
# {
#   "google/gemma-2b": {
#     "benchmark.input_shapes.batch_size=1": {
#       "abc123...": {"metrics": {...}},
#       "def456...": {"metrics": {...}}
#     }
#   }
# }


# Example 6: Upload results to HuggingFace Hub
cmd = [
    "python", "benchmark/benchmark.py",
    "--config-dir", "benchmark/config",
    "--config-name", "generation",
    "--commit", "diff",
    "backend.model=google/gemma-2b",
    "--repo_id", "my-org/benchmark-results",
    "--path_in_repo", "results/2025-12-18",
    "--token", "hf_..."
]
subprocess.run(cmd)
</syntaxhighlight>

== Related Pages ==

* (Leave empty for orphan pages)
