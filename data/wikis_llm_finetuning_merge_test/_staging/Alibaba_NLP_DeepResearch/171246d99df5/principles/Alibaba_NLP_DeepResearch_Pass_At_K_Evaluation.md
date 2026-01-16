# Principle: Pass_At_K_Evaluation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Evaluating Large Language Models on Code|https://arxiv.org/abs/2107.03374]]
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Evaluation]], [[domain::Metrics]], [[domain::Multi_Rollout]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:30 GMT]]
|}

== Overview ==

Statistical evaluation metric that measures the probability of finding at least one correct answer among K independent attempts, accounting for sampling variance in agent responses.

=== Description ===

Pass@K Evaluation is an evaluation methodology borrowed from code generation that applies naturally to multi-rollout agent systems. Instead of measuring accuracy on single attempts, it asks: "If we run the agent K times, what's the probability at least one run produces a correct answer?"

The metric is calculated as:

1. **Run N rollouts** per question (N >= K)
2. **Count correct answers** c out of N
3. **Calculate Pass@K** using the unbiased estimator

This approach is valuable because:
- Agent outputs are stochastic (temperature, sampling)
- Multiple attempts reveal true capability better than single shots
- Matches real-world usage where users can re-run queries

=== Usage ===

Use Pass@K Evaluation when:
- Evaluating agents with stochastic outputs
- Comparing multi-rollout systems
- You need confidence intervals on performance
- Single-shot accuracy is too noisy

== Theoretical Basis ==

The unbiased Pass@K estimator:

<math>
\text{Pass@K} = 1 - \frac{\binom{N-c}{K}}{\binom{N}{K}}
</math>

Where N is total samples, c is correct samples, and K is the number of attempts.

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
import numpy as np
from math import comb

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate Pass@K metric.

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: Number of attempts (k <= n)

    Returns:
        Probability of at least one correct in k attempts
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

# Example: 5 rollouts, 2 correct, calculate Pass@1, Pass@3
n, c = 5, 2
print(f"Pass@1: {pass_at_k(n, c, 1):.3f}")  # 0.400
print(f"Pass@3: {pass_at_k(n, c, 3):.3f}")  # 0.800
</syntaxhighlight>

Key properties:
- **Unbiased**: Correctly estimates true Pass@K probability
- **Monotonic**: Pass@K >= Pass@(K-1) always
- **Bounded**: Always in [0, 1]

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_WebResummer_Evaluate]]

=== Related Principles ===
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Pass_At_K_Metrics]]
* [[related_to::Principle:Alibaba_NLP_DeepResearch_LLM_Judge_Scoring]]
