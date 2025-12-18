{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Speculative Decoding|https://arxiv.org/abs/2211.17192]]
* [[source::Paper|EAGLE: Speculative Sampling|https://arxiv.org/abs/2401.15077]]
|-
! Domains
| [[domain::NLP]], [[domain::Inference]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

The decision process for choosing an appropriate speculative decoding strategy based on model architecture, resource constraints, and performance requirements.

=== Description ===

Speculative Method Selection determines which speculation technique to use for accelerating autoregressive generation. Key considerations:

1. **Model Support:** Not all methods work with all models
2. **Memory Budget:** Some methods require additional models/heads
3. **Target Speedup:** Different methods offer different acceleration
4. **Setup Complexity:** Trade-off between effort and benefit
5. **Acceptance Rate:** How often speculated tokens are accepted

=== Usage ===

Select speculative methods when:
- Optimizing latency-sensitive applications
- Balancing memory vs. speed trade-offs
- Working with supported model architectures
- Designing inference pipelines

== Theoretical Basis ==

'''Speculative Decoding Overview:'''

Standard autoregressive generation is memory-bound (reading model weights). Speculative decoding:
1. Draft K tokens quickly (speculation)
2. Verify all K tokens in parallel (single forward pass)
3. Accept correct tokens, resample from divergence point

<math>
Speedup \approx \frac{K \cdot acceptance\_rate}{1 + overhead}
</math>

'''Method Comparison:'''

<syntaxhighlight lang="text">
┌─────────────────────────────────────────────────────────────────┐
│                    Speculative Methods                          │
├─────────────────────────────────────────────────────────────────┤
│ N-gram Based:                                                   │
│   - Uses patterns in prompt/history for speculation             │
│   - No additional model needed                                  │
│   - Lower acceptance rate (~50-70%)                            │
│   - Best for: repetitive text, code completion                 │
├─────────────────────────────────────────────────────────────────┤
│ Draft Model:                                                    │
│   - Separate smaller model (e.g., 7B drafts for 70B)           │
│   - Higher acceptance rate (~70-85%)                           │
│   - Additional memory for draft model                          │
│   - Best for: general text, flexible setup                     │
├─────────────────────────────────────────────────────────────────┤
│ EAGLE/EAGLE3:                                                   │
│   - Trained draft heads on target model                        │
│   - High acceptance rate (~80-90%)                             │
│   - Minimal additional memory                                  │
│   - Best for: supported models, maximum speedup                │
├─────────────────────────────────────────────────────────────────┤
│ Medusa:                                                         │
│   - Multiple parallel draft heads                              │
│   - Can propose multiple candidates per position               │
│   - Best for: models with Medusa training                      │
└─────────────────────────────────────────────────────────────────┘
</syntaxhighlight>

'''Selection Algorithm:'''

<syntaxhighlight lang="python">
def select_speculative_method(model, constraints):
    """
    Select best speculative method for given model and constraints.

    Args:
        model: Target model name
        constraints: Dict with memory_budget, target_speedup, etc.

    Returns:
        Recommended method and configuration
    """
    # Check for native support
    if has_eagle_support(model):
        if constraints.get("target_speedup", 0) > 2.5:
            return "eagle", {"num_speculative_tokens": 5}

    # Memory-constrained: use ngram
    if constraints.get("low_memory", False):
        return "ngram", {
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 2,
        }

    # High speedup with any model: draft model
    if constraints.get("target_speedup", 0) > 2.0:
        draft = find_compatible_draft(model)
        if draft:
            return "draft_model", {
                "model": draft,
                "num_speculative_tokens": 5,
            }

    # Default: ngram (universal, simple)
    return "ngram", {"prompt_lookup_max": 4}
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_SpeculativeMethod_choice]]
