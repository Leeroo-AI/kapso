# Principle: Result_Aggregation

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

Multi-rollout result aggregation for computing consensus metrics across independent agent runs.

=== Description ===

Result Aggregation combines judgement results from multiple independent agent rollouts into a unified structure that enables Pass@k metric computation. In the DeepResearch evaluation framework, agents are run 3 times independently on each question, producing 3 separate judgements per query.

The aggregation process:
1. Collects judgements from all rounds (round1, round2, round3)
2. Groups results by question to create a per-query view
3. Preserves the original ground truth answer for reference
4. Normalizes judgement values to "Correct" or other statuses

This aggregated structure enables computing various evaluation metrics:
- **Pass@1**: Success rate of individual rounds
- **Pass@3**: At least one correct answer across 3 attempts
- **Avg Pass@3**: Average accuracy across all rounds
- **Best Pass@1**: Highest single-round accuracy

=== Usage ===

Use Result Aggregation when:
- Computing Pass@k metrics from multi-rollout evaluations
- Analyzing consistency of agent responses across runs
- Identifying questions with variable success rates
- Preparing data for benchmark reporting

This principle bridges per-round judgements to aggregate performance metrics.

== Theoretical Basis ==

Result Aggregation creates a query-centric view of multi-round evaluation results.

'''Aggregation Schema:'''
<syntaxhighlight lang="python">
{
    "question_1": {
        "round1": "Correct" | "Incorrect" | "Error",
        "round2": "Correct" | "Incorrect" | "Error",
        "round3": "Correct" | "Incorrect" | "Error",
        "answer": "ground_truth_answer"
    },
    "question_2": { ... },
    ...
}
</syntaxhighlight>

'''Aggregation Process:'''
<syntaxhighlight lang="text">
Input:
  - round1_results: List of {question, answer, judgement} from round 1
  - round2_results: List of {question, answer, judgement} from round 2
  - round3_results: List of {question, answer, judgement} from round 3

For each round and result:
  1. Use question as key
  2. Initialize query entry if new
  3. Store normalized judgement for round
  4. Preserve ground truth answer

Output:
  - query_results: Dict mapping questions to per-round correctness
</syntaxhighlight>

'''Correctness Normalization:'''
{| class="wikitable"
|-
! Raw Judgement !! Normalized Value
|-
| "correct" (case-insensitive) || "Correct"
|-
| Starts with "a" or "A" || "Correct"
|-
| Other values || Capitalized original value
|}

The aggregated structure supports efficient Pass@k calculation by providing O(1) lookup of per-question, per-round results.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_Aggregate_Results]]
