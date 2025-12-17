# Implementation: huggingface_transformers_benchmarks_entrypoint

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::CI_CD]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Automated metrics collection system serving as the entry point for CI benchmark workflows, coordinating model benchmarking and result reporting.

=== Description ===

The `benchmark/benchmarks_entrypoint.py` module (502 lines) is the main entry point for CI benchmark jobs. It orchestrates:
- Model selection based on PR changes
- Benchmark execution via optimum-benchmark
- Result aggregation and formatting
- Integration with CI reporting systems

=== Usage ===

Called by CI workflows to run benchmarks on affected models. Outputs structured results for dashboard consumption.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/benchmark/benchmarks_entrypoint.py benchmark/benchmarks_entrypoint.py]
* '''Lines:''' 1-502

=== Signature ===
<syntaxhighlight lang="python">
def get_models_to_benchmark(modified_files: list[str]) -> list[str]:
    """Determine which models need benchmarking based on changes."""

def run_benchmarks(models: list[str], config: dict) -> dict:
    """Execute benchmarks for specified models."""

def format_results(results: dict) -> str:
    """Format benchmark results for CI output."""

def main():
    """CI entry point for benchmark execution."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="bash">
python benchmark/benchmarks_entrypoint.py
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| Modified files || env/git || Yes || Files changed in PR
|-
| Benchmark config || dict || No || Override default settings
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| benchmark_results.json || File || Structured benchmark data
|-
| CI annotations || stdout || Performance warnings/info
|}

== Usage Examples ==

=== CI Workflow Integration ===
<syntaxhighlight lang="yaml">
# .github/workflows/benchmark.yml
- name: Run Benchmarks
  run: python benchmark/benchmarks_entrypoint.py
  env:
    BENCHMARK_MODELS: auto  # Detect from PR changes
</syntaxhighlight>

== Related Pages ==
