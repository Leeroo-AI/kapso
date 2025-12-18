{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Speculative Decoding|https://arxiv.org/abs/2211.17192]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Inference]], [[domain::Configuration]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

The process of specifying parameters for speculative decoding including method, speculation depth, and draft model settings.

=== Description ===

Speculative Configuration defines how speculation is performed during generation:

1. **Method Selection:** Which algorithm to use (ngram, eagle, etc.)
2. **Speculation Depth:** How many tokens to propose per step
3. **Draft Model:** Which model generates proposals (if applicable)
4. **N-gram Settings:** Window sizes for pattern matching
5. **Acceptance Threshold:** Criteria for token acceptance

=== Usage ===

Configure speculation parameters when:
- Tuning speculation depth for target latency
- Setting up draft models
- Optimizing acceptance rates
- Balancing throughput and latency

== Theoretical Basis ==

'''Key Parameters:'''

<syntaxhighlight lang="python">
# Parameter relationships
speculation_overhead = time_to_draft(K) + time_to_verify(K)
acceptance_benefit = K * acceptance_rate * time_saved_per_token

# Optimal K depends on:
# - Acceptance rate (higher = more aggressive speculation)
# - Draft overhead (lower = more speculation viable)
# - Batch size (larger batches = less speculation benefit)
</syntaxhighlight>

'''N-gram Configuration:'''

<syntaxhighlight lang="python">
# N-gram speculation looks for patterns in prompt/history
# prompt_lookup_max: longest pattern to match (e.g., 4 = "word word word word")
# prompt_lookup_min: shortest pattern to match (e.g., 2 = "word word")

# Larger max = more potential matches but slower lookup
# Smaller min = more matches but lower quality proposals
</syntaxhighlight>

'''Speculation Depth Trade-offs:'''

<syntaxhighlight lang="text">
num_speculative_tokens:
┌─────────────────────────────────────────────────────────────────┐
│ K=1: Minimal speculation, small benefit                         │
│ K=3: Conservative, good acceptance rate                        │
│ K=5: Balanced (recommended default)                            │
│ K=7: Aggressive, may have lower acceptance                     │
│ K=10+: Very aggressive, often diminishing returns              │
└─────────────────────────────────────────────────────────────────┘

Recommendation: Start with K=5, tune based on acceptance metrics
</syntaxhighlight>

'''Draft Model Sizing:'''

<syntaxhighlight lang="python">
# Rule of thumb for draft model selection:
# - Draft should be ~10-20x smaller than target
# - Same architecture family preferred
# - Similar training distribution

# Examples:
# Target: Llama-3.1-70B → Draft: Llama-3.2-1B or 3B
# Target: Llama-3.1-8B → Draft: Llama-3.2-1B
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_SpeculativeConfig_init]]
