# Principle: Pass_At_K_Metrics

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::File|evaluate_deepsearch_official.py|evaluation/evaluate_deepsearch_official.py]]
* [[source::Paper|Evaluating Large Language Models Trained on Code|https://arxiv.org/abs/2107.03374]]
|-
! Domains
| [[domain::Evaluation]], [[domain::Metrics]], [[domain::Agent_Systems]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

Pass@k evaluation metrics that measure success rate when k attempts are allowed per question.

=== Description ===

Pass@k is an evaluation metric originally popularized in code generation benchmarks (e.g., HumanEval) that measures the probability of obtaining at least one correct solution when k independent attempts are made. In the DeepResearch context, it quantifies agent reliability across multiple rollouts.

The metric captures two key aspects:
1. **Capability**: Can the agent solve the problem at all?
2. **Consistency**: How reliably does it produce correct answers?

The DeepResearch framework computes several Pass@k variants:

{| class="wikitable"
|-
! Metric !! Definition !! Interpretation
|-
| Pass@1 || Accuracy of a single round || Base success rate
|-
| Pass@3 || At least 1 correct in 3 attempts || Agent capability upper bound
|-
| Avg Pass@3 || Mean accuracy across 3 rounds || Expected single-attempt accuracy
|-
| Best Pass@1 || Highest round accuracy || Peak performance
|}

=== Usage ===

Use Pass@k Metrics when:
- Evaluating agents that exhibit non-deterministic behavior
- Comparing systems where multiple attempts are practical
- Understanding the gap between average and best-case performance
- Benchmarking research agent systems on complex tasks

Pass@3 is the primary metric reported in DeepResearch benchmark results.

== Theoretical Basis ==

Pass@k measures the probability that at least one of k attempts succeeds.

'''Mathematical Definition:'''

For a question with n total samples and c correct samples:

<math>
\text{Pass@k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}
</math>

'''Simplified Calculation (DeepResearch):'''

When exactly k samples are taken (k=3 in DeepResearch):

<syntaxhighlight lang="text">
Pass@k for single question = 1 if any attempt is correct, else 0

Overall Pass@k = (# questions with >= 1 correct in k attempts) / (total questions)
</syntaxhighlight>

'''Implementation Logic:'''
<syntaxhighlight lang="python">
def calculate_pass_at_k(query_results, k):
    total_correct = 0
    for query, results in query_results.items():
        rounds = [results["round1"], results["round2"], results["round3"]][:k]
        if "Correct" in rounds:
            total_correct += 1
    return total_correct / len(query_results) * 100
</syntaxhighlight>

'''Metric Relationships:'''
<syntaxhighlight lang="text">
Avg Pass@1 <= Best Pass@1 <= Pass@3

Example:
- Round 1: 60% correct
- Round 2: 65% correct
- Round 3: 62% correct
- Avg Pass@3: 62.3%
- Best Pass@1: 65%
- Pass@3: ~75% (some questions solved in different rounds)
</syntaxhighlight>

The gap between Pass@3 and Avg Pass@3 indicates how much agent variability helps coverage - larger gaps suggest the agent can solve different questions in different runs.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_Calculate_Pass_At_K]]
