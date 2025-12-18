{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Speculative Decoding|https://arxiv.org/abs/2211.17192]]
* [[source::Paper|Fast Inference from Transformers via Speculative Decoding|https://arxiv.org/abs/2211.17192]]
|-
! Domains
| [[domain::NLP]], [[domain::Inference]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

The process of generating text using speculative decoding where draft tokens are proposed and verified in parallel for acceleration.

=== Description ===

Speculative Generation accelerates autoregressive text generation by:

1. **Speculation Phase:** Draft K tokens using fast mechanism
2. **Verification Phase:** Verify all K tokens in single forward pass
3. **Acceptance:** Keep correctly speculated tokens
4. **Rejection Handling:** Resample from correct distribution at divergence

This maintains the exact output distribution while reducing wall-clock time.

=== Usage ===

Use speculative generation when:
- Latency is critical (interactive apps)
- Serving large models
- Memory-bound (not compute-bound) workloads
- Single-request scenarios (less benefit for large batches)

== Theoretical Basis ==

'''Speculative Decoding Algorithm:'''

<syntaxhighlight lang="python">
def speculative_generate(target, draft, prompt, K):
    """
    Generate with speculative decoding.

    Args:
        target: Target (large) model
        draft: Draft (fast) model/mechanism
        prompt: Input prompt
        K: Number of tokens to speculate

    Returns:
        Generated tokens
    """
    tokens = tokenize(prompt)

    while not is_complete(tokens):
        # 1. Draft K tokens
        draft_tokens = []
        draft_probs = []
        for i in range(K):
            p = draft.forward(tokens + draft_tokens)
            token = sample(p)
            draft_tokens.append(token)
            draft_probs.append(p[token])

        # 2. Verify all K in parallel
        target_probs = target.forward_parallel(tokens, draft_tokens)

        # 3. Accept/reject each token
        for i, (draft_tok, draft_p, target_p) in enumerate(
            zip(draft_tokens, draft_probs, target_probs)
        ):
            # Rejection sampling criterion
            if random() < min(1, target_p[draft_tok] / draft_p):
                tokens.append(draft_tok)  # Accept
            else:
                # Reject and resample
                adjusted_p = max(0, target_p - draft_p)
                tokens.append(sample(adjusted_p))
                break  # Stop accepting this batch

    return tokens
</syntaxhighlight>

'''Speedup Analysis:'''

<math>
Speedup = \frac{K \cdot \alpha}{1 + \frac{T_{draft}}{T_{target}}}
</math>

Where:
- <math>K</math> = speculation depth
- <math>\alpha</math> = acceptance rate
- <math>T_{draft}</math> = draft time
- <math>T_{target}</math> = target forward time

'''Correctness Guarantee:'''

Speculative decoding produces samples from the exact target distribution:
- Accepted tokens match what target would have produced
- Rejected tokens are resampled correctly
- No approximation or distortion

'''Batch Size Effect:'''

<syntaxhighlight lang="text">
Batch Size vs. Speculation Benefit:
┌─────────────────────────────────────────────────────────────────┐
│ Batch=1:  Maximum speedup (memory-bound, speculation helps)     │
│ Batch=8:  Good speedup (still memory-bound)                    │
│ Batch=32: Moderate speedup (becoming compute-bound)            │
│ Batch=128: Minimal speedup (compute-bound, batching is better) │
└─────────────────────────────────────────────────────────────────┘

Recommendation: Best for small batches, interactive use
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_LLM_generate_spec]]
