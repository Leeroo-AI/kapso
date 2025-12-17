# Implementation: huggingface_transformers_get_ci_error_statistics

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::CI_CD]], [[domain::Analytics]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

CI failure analytics tool that aggregates error patterns across test runs to identify flaky tests and recurring issues.

=== Description ===

The `utils/get_ci_error_statistics.py` module (305 lines) analyzes CI failure patterns. It:
- Parses JUnit test result files
- Groups failures by error type/message
- Identifies flaky tests (intermittent failures)
- Generates statistics on failure frequency

=== Usage ===

Run to analyze CI health and identify tests needing attention. Useful for prioritizing test reliability work.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/utils/get_ci_error_statistics.py utils/get_ci_error_statistics.py]
* '''Lines:''' 1-305

=== Signature ===
<syntaxhighlight lang="python">
def parse_junit_results(results_dir: str) -> list[dict]:
    """Parse test failures from JUnit XML."""

def group_by_error(failures: list[dict]) -> dict:
    """Group failures by error message."""

def identify_flaky_tests(history: list[dict]) -> list[str]:
    """Find intermittently failing tests."""

def main():
    """Generate CI error statistics report."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="bash">
python utils/get_ci_error_statistics.py --results_dir ./test-results --days 30
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| --results_dir || str || No || Test results directory
|-
| --days || int || No || Analysis period
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Statistics report || stdout || Failure patterns and trends
|}

== Usage Examples ==

=== Analyze CI Failures ===
<syntaxhighlight lang="bash">
python utils/get_ci_error_statistics.py --days 30

# Output:
# Top failures (last 30 days):
#   1. test_bert_cuda: 15 failures (OOM)
#   2. test_gpt2_generation: 8 failures (flaky)
</syntaxhighlight>

== Related Pages ==
