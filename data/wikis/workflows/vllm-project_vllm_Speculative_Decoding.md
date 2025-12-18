{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|Speculative Decoding|https://docs.vllm.ai/en/latest/serving/spec_decode.html]]
|-
! Domains
| [[domain::LLM_Inference]], [[domain::Speculative_Decoding]], [[domain::Performance_Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
End-to-end process for accelerating inference using speculative decoding techniques including EAGLE, n-gram, and MTP methods.

=== Description ===
This workflow demonstrates vLLM's speculative decoding capabilities, which accelerate generation by predicting multiple tokens ahead and verifying them in parallel. Speculative decoding uses a faster draft mechanism (small model, n-gram lookup, or EAGLE predictor) to propose candidate tokens, then the target model verifies them in a single forward pass. This reduces latency by accepting multiple tokens per iteration while maintaining output quality identical to standard decoding.

=== Usage ===
Execute this workflow when you need to reduce generation latency for single-request scenarios or interactive applications. Speculative decoding provides significant speedups (2-3x) for models where autoregressive decoding is the bottleneck. Best suited for high-quality, latency-sensitive deployments where throughput is less critical than time-to-first-token and tokens-per-second per request.

== Execution Steps ==

=== Step 1: Speculative Method Selection ===
[[step::Principle:vllm-project_vllm_Speculative_Method_Selection]]

Choose the speculative decoding method based on model architecture and deployment constraints. Options include EAGLE (learned drafter), n-gram (prompt-based lookup), and MTP (multi-token prediction). Each method has different accuracy-speed tradeoffs and resource requirements.

'''Available methods:'''
* `eagle` / `eagle3` - Learned draft model for high acceptance rates
* `ngram` - N-gram lookup from prompt for zero-overhead drafting
* `mtp` - Multi-token prediction head (model must support this)
* Draft model methods require additional GPU memory
* N-gram has no memory overhead but lower acceptance rates

=== Step 2: Speculative Configuration ===
[[step::Principle:vllm-project_vllm_Speculative_Configuration]]

Configure speculative decoding parameters through the `speculative_config` dictionary. Key parameters include the draft method, number of speculative tokens, and method-specific options like draft model path or n-gram settings.

'''Configuration parameters:'''
* `method` - Speculative method ("eagle", "ngram", "mtp")
* `model` - Path to draft model (for EAGLE/draft model methods)
* `num_speculative_tokens` - Number of tokens to predict ahead
* `prompt_lookup_min/max` - N-gram window sizes (for ngram method)
* Balance speculation depth vs verification overhead

=== Step 3: Engine Initialization ===
[[step::Principle:vllm-project_vllm_Speculative_Engine_Init]]

Initialize the vLLM engine with speculative decoding enabled. The engine loads both the target model and draft mechanism, allocating memory appropriately. For EAGLE, this involves loading a separate draft model; for n-gram, only additional buffers are needed.

'''Initialization considerations:'''
* Draft models require additional GPU memory
* KV cache allocated for both target and draft models
* CUDA graphs may be compiled for speculation paths
* `enforce_eager` can be useful for debugging speculation

=== Step 4: Speculative Generation ===
[[step::Principle:vllm-project_vllm_Speculative_Generation]]

Execute text generation with speculative decoding active. The engine alternates between draft token generation and target model verification. Multiple tokens may be accepted per iteration, reducing total forward passes needed. Temperature and other sampling parameters work with speculation.

'''Generation flow:'''
* Draft mechanism proposes `num_speculative_tokens` candidates
* Target model verifies all candidates in single forward pass
* Accepted tokens are committed; rejected tokens trigger rollback
* Process repeats until max tokens or stop condition

=== Step 5: Acceptance Metrics Analysis ===
[[step::Principle:vllm-project_vllm_Speculation_Metrics]]

Analyze speculative decoding performance through acceptance metrics. Key metrics include acceptance rate per position, mean acceptance length, and overall speedup. These metrics help tune speculation parameters and compare different methods.

'''Performance metrics:'''
* `num_drafts` - Total speculation rounds
* `num_draft_tokens` - Total tokens proposed
* `num_accepted_tokens` - Tokens accepted without recomputation
* `acceptance_per_pos` - Per-position acceptance rates
* Mean acceptance length indicates effective speedup

== Execution Diagram ==
{{#mermaid:graph TD
    A[Speculative Method Selection] --> B[Speculative Configuration]
    B --> C[Engine Initialization]
    C --> D[Speculative Generation]
    D --> E[Acceptance Metrics Analysis]
}}

== Related Pages ==
* [[step::Principle:vllm-project_vllm_Speculative_Method_Selection]]
* [[step::Principle:vllm-project_vllm_Speculative_Configuration]]
* [[step::Principle:vllm-project_vllm_Speculative_Engine_Init]]
* [[step::Principle:vllm-project_vllm_Speculative_Generation]]
* [[step::Principle:vllm-project_vllm_Speculation_Metrics]]
