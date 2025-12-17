= Speculative Generation Principle =

== Metadata ==
{| class="wikitable"
|-
! Field !! Value
|-
| Knowledge Sources || [https://arxiv.org/abs/2302.01318 Speculative Sampling], [https://arxiv.org/abs/2401.15077 EAGLE], [https://arxiv.org/abs/2211.17192 Rejection Sampling], vLLM Design Documents
|-
| Domains || Inference Optimization, Token Generation, Rejection Sampling
|-
| Last Updated || 2025-12-17
|}

== Overview ==

The Speculative Generation principle defines the core execution pattern for accelerated text generation using speculative decoding. It encompasses draft token proposal, parallel verification, acceptance/rejection decisions, and the orchestration of multi-step speculation to achieve reduced inter-token latency while maintaining output quality.

== Description ==

Speculative generation is a multi-phase process that generates candidate tokens speculatively and verifies them in parallel using the target model. This principle establishes the theoretical foundation and practical implementation patterns for all speculative decoding methods in vLLM.

=== Core Algorithm ===

The fundamental speculative decoding loop consists of:

1. '''Draft Phase''': Generate k candidate tokens using a fast method (draft model, ngram, etc.)
2. '''Verification Phase''': Compute target model probabilities for all candidates in parallel
3. '''Acceptance Phase''': Use rejection sampling to accept/reject candidates
4. '''Recovery Phase''': Sample from target model if candidates rejected
5. '''Iteration''': Repeat until generation complete or max tokens reached

=== Theoretical Foundation ===

Speculative decoding maintains '''losslessness''' through rejection sampling:

* Target distribution p(x) is preserved exactly (up to floating-point precision)
* Draft distribution q(x) proposes candidates
* Acceptance probability: min(1, p(x)/q(x))
* Rejected tokens replaced with corrected sample from (p(x) - q(x))/Z

This ensures the output distribution matches non-speculative generation while achieving speedup through parallelization.

=== Method-Specific Strategies ===

==== N-gram Proposal ====
* '''Draft''': Pattern match last n tokens in prompt/history
* '''Speedup''': Zero-cost proposal, effective for repetitive content
* '''Acceptance''': Depends on pattern repetition frequency
* '''Fallback''': Standard sampling when no matches found

==== EAGLE Proposal ====
* '''Draft''': Feature-level autoregression using draft heads
* '''Speedup''': High acceptance rates (2-2.8x typical)
* '''Acceptance''': Best for natural language, lower for code
* '''Overhead''': Draft model forward pass per speculation step

==== MLP Speculator ====
* '''Draft''': Lightweight MLP networks condition on context + previous tokens
* '''Speedup''': Balanced performance (1.5-2x)
* '''Acceptance''': Consistent across tasks
* '''Overhead''': Minimal due to small speculator size

==== Suffix Decoding ====
* '''Draft''': Dynamic pattern matching with frequency-based ranking
* '''Speedup''': Adaptive speculation length per request
* '''Acceptance''': Excellent for repetitive tasks (agentic loops, RL)
* '''Overhead''': Tree maintenance and prefix matching

=== Acceptance Rate Optimization ===

Key factors affecting acceptance rate:

* '''Draft Quality''': Better draft models → higher acceptance
* '''Temperature''': Lower temperature (greedy) → higher acceptance
* '''Task Type''': Repetitive tasks → higher acceptance
* '''Prompt Context''': Longer, relevant context → better proposals

=== Batch Speculation ===

Speculative generation must handle batching efficiently:

* '''Heterogeneous Lengths''': Different requests may have different acceptance rates
* '''Tree Attention''': EAGLE uses tree attention to handle multiple candidates
* '''Memory Management''': KV cache allocation for speculative tokens
* '''Scheduling''': Lookahead scheduling coordinates multiple speculation depths

== Usage Context ==

This principle applies during:

* Every <code>generate()</code> call with speculative_config enabled
* Streaming generation with speculation
* Batch inference with mixed speculation depths
* Online serving with dynamic speculation control

The principle governs:
* Token generation speed and efficiency
* Memory usage patterns during speculation
* Quality maintenance through rejection sampling
* Adaptive behavior based on acceptance rates

== Design Considerations ==

=== Trade-offs ===

* '''Speculation Depth vs. Overhead''': More speculative tokens increase potential speedup but add overhead
* '''Draft Quality vs. Cost''': Better draft models cost more memory/compute but improve acceptance
* '''Batch Size vs. Speculation''': Larger batches may reduce speculation effectiveness
* '''Latency vs. Throughput''': Speculation optimizes latency, may not always improve throughput

=== Performance Factors ===

==== Favorable Conditions ====
* Memory-bound inference (large models)
* Low to medium batch sizes (1-32)
* Longer generation lengths (>50 tokens)
* Repetitive or structured outputs
* Greedy or low-temperature sampling

==== Unfavorable Conditions ====
* Very high batch sizes (>64)
* Very short generations (<10 tokens)
* Highly creative/random outputs
* High temperature sampling
* Compute-bound inference

=== Implementation Strategies ===

==== Rejection Sampling ====
Implemented using modified rejection sampling algorithm:
<syntaxhighlight lang="text">
for each candidate token x:
    r = uniform(0, 1)
    if r < p(x) / q(x):
        accept x
    else:
        reject x and sample from adjusted distribution
        break (no more candidates accepted)
</syntaxhighlight>

==== Tree Attention (EAGLE) ====
* Organize candidates in tree structure
* Compute attention over tree nodes
* Batch multiple speculation paths
* Efficient verification of multiple candidates

==== Adaptive Speculation ====
* Monitor acceptance rates per request
* Dynamically adjust speculation depth
* Disable speculation for low-acceptance requests
* Re-enable when conditions improve

=== Losslessness Guarantees ===

vLLM's speculative decoding aims for losslessness with these considerations:

* '''Theoretical''': Algorithmically lossless up to floating-point precision
* '''Practical''': May have minor numerical differences due to:
  - Batching effects on softmax computation
  - Different execution paths (CUDA graphs, attention backends)
  - Hardware floating-point variations

* '''Validation''': Greedy sampling (temperature=0) should match non-speculative exactly

== Optimization Strategies ==

=== Memory Optimization ===
* Reuse KV cache slots for rejected tokens
* Efficient block management for speculative tokens
* Prune speculation trees when acceptance rate drops

=== Compute Optimization ===
* CUDA graphs for speculation when possible
* Fused kernels for rejection sampling
* Efficient tree attention implementations
* Batch speculation across multiple requests

=== Scheduling Optimization ===
* Lookahead scheduling for speculative tokens
* Dynamic batch expansion/contraction
* Priority scheduling based on acceptance rates
* Coordinated speculation across distributed workers

== Related Principles ==

* [[implements::vllm-project_vllm_LLM_generate_spec]] - Generation implementation
* Rejection Sampling Principles - Statistical correctness
* KV Cache Management - Memory efficiency
* Scheduler Design - Batch coordination

== See Also ==

* [[implemented_by::Implementation:vllm-project_vllm_LLM_generate_spec]]
* [[implements::vllm-project_vllm_get_metrics]]
* vLLM Scheduler Documentation
* Speculative Decoding FAQ
