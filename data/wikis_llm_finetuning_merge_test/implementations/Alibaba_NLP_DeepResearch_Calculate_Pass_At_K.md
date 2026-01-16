# Implementation: Calculate_Pass_At_K

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::File|evaluate_deepsearch_official.py|evaluation/evaluate_deepsearch_official.py]]
|-
! Domains
| [[domain::Evaluation]], [[domain::Metrics]], [[domain::Agent_Systems]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

Concrete tool for computing Pass@k metrics from aggregated multi-round evaluation results.

=== Description ===

The `calculate_pass_at_k()` function computes the Pass@k metric, which measures the percentage of questions answered correctly in at least one of k attempts. This is a key benchmark metric for evaluating research agents.

The function:
1. Iterates through all questions in the aggregated results
2. Checks the first k rounds for each question
3. Counts questions with at least one "Correct" judgement
4. Returns the success rate as a percentage

With the default k=3 (matching the 3 evaluation rounds), this computes the headline Pass@3 metric used in DeepResearch benchmarks.

=== Usage ===

Use `calculate_pass_at_k()` when:
- Computing Pass@3 benchmark metrics
- Evaluating agent capability (best-case performance)
- Comparing different agent configurations
- Reporting evaluation results

This is the primary metric function in the evaluation pipeline.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba-NLP/DeepResearch]
* '''File:''' evaluation/evaluate_deepsearch_official.py
* '''Lines:''' 405-415

=== Signature ===
<syntaxhighlight lang="python">
def calculate_pass_at_k(query_results: Dict, k: int = 10) -> float:
    """
    Calculate Pass@k metric.

    Args:
        query_results: Dict - Aggregated results from aggregate_results()
            {question: {"round1": ..., "round2": ..., "round3": ..., "answer": ...}}
        k: int - Number of attempts to consider (default 10, typically use 3)

    Returns:
        float - Percentage of questions with at least one correct answer
                in the first k attempts (0-100 scale)
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from evaluate_deepsearch_official import calculate_pass_at_k
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| query_results || Dict || Yes || Aggregated results from aggregate_results()
|-
| query_results[q]["round1"] || str || Yes || Round 1 judgement
|-
| query_results[q]["round2"] || str || Yes || Round 2 judgement
|-
| query_results[q]["round3"] || str || Yes || Round 3 judgement
|-
| k || int || No || Number of attempts (default: 10, capped at 3 available rounds)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| pass_at_k || float || Percentage of questions correct in at least one of k attempts (0-100)
|}

== Usage Examples ==

=== Basic Pass@3 Calculation ===
<syntaxhighlight lang="python">
from evaluate_deepsearch_official import calculate_pass_at_k, aggregate_results

# Aggregate results from three rounds
query_results = aggregate_results(round1_results, round2_results, round3_results)

# Calculate Pass@3
pass_at_3 = calculate_pass_at_k(query_results, k=3)
print(f"Pass@3: {pass_at_3}%")  # e.g., "Pass@3: 75.5%"
</syntaxhighlight>

=== Comparing Different k Values ===
<syntaxhighlight lang="python">
query_results = aggregate_results(round1_results, round2_results, round3_results)

# Compute metrics for different k values
for k in [1, 2, 3]:
    pass_at_k = calculate_pass_at_k(query_results, k=k)
    print(f"Pass@{k}: {pass_at_k}%")

# Typical output:
# Pass@1: 62.0%  (best single round)
# Pass@2: 70.5%  (at least one of first two rounds)
# Pass@3: 75.5%  (at least one of all three rounds)
</syntaxhighlight>

=== Full Metrics Report ===
<syntaxhighlight lang="python">
from evaluate_deepsearch_official import (
    calculate_pass_at_k,
    calculate_best_pass_at_1,
    calculate_avg_pass_at_3
)

query_results = aggregate_results(round1_results, round2_results, round3_results)

# Compute all standard metrics
pass_at_3 = calculate_pass_at_k(query_results, k=3)
best_pass_at_1 = calculate_best_pass_at_1(query_results)
avg_pass_at_3 = calculate_avg_pass_at_3(query_results)

print(f"=== Evaluation Results ===")
print(f"Pass@3: {pass_at_3}%")
print(f"Best Pass@1: {best_pass_at_1}%")
print(f"Avg Pass@3: {avg_pass_at_3}%")
</syntaxhighlight>

=== Manual Verification ===
<syntaxhighlight lang="python">
# Understanding the calculation
query_results = {
    "Q1": {"round1": "Correct", "round2": "Incorrect", "round3": "Correct", "answer": "A1"},
    "Q2": {"round1": "Incorrect", "round2": "Incorrect", "round3": "Incorrect", "answer": "A2"},
    "Q3": {"round1": "Incorrect", "round2": "Correct", "round3": "Incorrect", "answer": "A3"},
    "Q4": {"round1": "Correct", "round2": "Correct", "round3": "Correct", "answer": "A4"},
}

# Q1: Has "Correct" -> counts
# Q2: No "Correct" -> doesn't count
# Q3: Has "Correct" -> counts
# Q4: Has "Correct" -> counts

# Pass@3 = 3/4 * 100 = 75.0%
pass_at_3 = calculate_pass_at_k(query_results, k=3)
assert pass_at_3 == 75.0
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Alibaba_NLP_DeepResearch_Pass_At_K_Metrics]]
