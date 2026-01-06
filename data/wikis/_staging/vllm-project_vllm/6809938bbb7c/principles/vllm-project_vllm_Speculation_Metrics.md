{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Speculative Decoding|https://arxiv.org/abs/2211.17192]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Monitoring]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

The practice of collecting, analyzing, and utilizing metrics from speculative decoding to optimize performance.

=== Description ===

Speculation Metrics provide insight into how well speculative decoding is performing:

1. **Acceptance Rate:** What fraction of speculated tokens are accepted
2. **Draft Efficiency:** How often draft proposals match target
3. **Position Analysis:** Acceptance rate decay by position
4. **Throughput Impact:** Actual speedup achieved
5. **Overhead Measurement:** Cost of speculation vs. benefit

=== Usage ===

Analyze speculation metrics when:
- Tuning num_speculative_tokens parameter
- Comparing speculation methods
- Diagnosing poor performance
- Making method selection decisions

== Theoretical Basis ==

'''Key Metrics:'''

<syntaxhighlight lang="python">
# Speculation metrics
metrics = {
    "num_drafts": int,           # Total speculation attempts
    "num_accepted": int,         # Accepted tokens
    "acceptance_rate": float,    # num_accepted / (num_drafts * K)
    "per_position_rate": list,   # Acceptance by position [p1, p2, ...]
}
</syntaxhighlight>

'''Acceptance Rate Analysis:'''

<math>
Acceptance\_Rate = \frac{Accepted\_Tokens}{Drafted\_Tokens} = \frac{Accepted}{Drafts \times K}
</math>

Typical values:
- Good: 70-90%
- Moderate: 50-70%
- Poor: <50%

'''Per-Position Decay:'''

Acceptance typically decreases with position:

<syntaxhighlight lang="text">
Position:    1     2     3     4     5
Acceptance: 85%   75%   65%   55%   45%

This decay is normal - errors compound over positions
</syntaxhighlight>

'''Speedup Calculation:'''

<syntaxhighlight lang="python">
def calculate_expected_speedup(acceptance_rate, K, overhead_ratio):
    """
    Calculate expected speedup from speculation.

    Args:
        acceptance_rate: Fraction of accepted tokens (0-1)
        K: Number of speculative tokens
        overhead_ratio: draft_time / target_time

    Returns:
        Expected speedup multiplier
    """
    effective_tokens = K * acceptance_rate
    cost = 1 + overhead_ratio  # One target pass + draft overhead
    return effective_tokens / cost

# Example: 70% acceptance, K=5, 10% overhead
speedup = calculate_expected_speedup(0.70, 5, 0.10)
# speedup ≈ 3.18x
</syntaxhighlight>

'''Tuning Guidelines:'''

<syntaxhighlight lang="text">
If acceptance rate < 50%:
  - Reduce num_speculative_tokens (K)
  - Try different method (e.g., eagle instead of ngram)
  - Check if workload suits speculation

If acceptance rate > 80%:
  - Can try increasing K
  - May see more speedup

If per-position drops sharply:
  - Current K may be too high
  - Reduce K to match where rate drops below 50%
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_get_metrics_spec]]
