{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::LLMs]], [[domain::Inference]], [[domain::Optimization]], [[domain::Speculative_Decoding]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

End-to-end process for accelerating LLM inference using speculative decoding techniques that predict multiple tokens in parallel and verify them with the target model.

=== Description ===

This workflow demonstrates vLLM's speculative decoding capabilities, which can significantly reduce inference latency by speculatively generating multiple tokens in parallel. vLLM supports several speculative decoding methods including EAGLE, n-gram, and MLP speculator approaches.

Key capabilities:
* **EAGLE**: Draft model trained to predict multi-token sequences
* **EAGLE3**: Enhanced version with improved acceptance rates
* **N-gram**: Prompt-based lookup for repetitive patterns
* **MLP speculator**: Lightweight neural draft predictor
* **MTP (Multi-Token Prediction)**: Model-native multi-token heads

=== Usage ===

Execute this workflow when you need to:
* Reduce generation latency for interactive applications
* Accelerate inference without sacrificing output quality
* Utilize spare GPU compute for speculative drafting
* Handle prompts with repetitive patterns (n-gram excels here)

Best suited for scenarios where latency is critical and the workload has predictable patterns or compatible draft models are available.

== Execution Steps ==

=== Step 1: Select Speculative Method ===
[[step::Principle:vllm-project_vllm_spec_method_selection]]

Choose the appropriate speculative decoding method based on your use case. Each method has different trade-offs in terms of setup complexity, acceptance rate, and computational overhead.

'''Method comparison:'''
* **EAGLE/EAGLE3**: Best acceptance rate, requires trained draft model
* **N-gram**: No extra model needed, works for repetitive content
* **MLP speculator**: Lightweight, moderate acceptance rate
* **MTP**: Model must have multi-token prediction heads

=== Step 2: Configure Speculative Parameters ===
[[step::Principle:vllm-project_vllm_speculative_engine_init]]

Set up the speculative configuration dictionary with method-specific parameters. This includes the number of speculative tokens, draft model paths, and method-specific settings.

'''Configuration options:'''
* `method`: Speculative method ("eagle", "eagle3", "ngram", "mtp")
* `num_speculative_tokens`: Tokens to speculate per step (2-5 typical)
* `model`: Path to draft model (for EAGLE methods)
* `prompt_lookup_max/min`: N-gram window sizes

=== Step 3: Initialize Speculative Engine ===
[[step::Principle:vllm-project_vllm_speculative_prompt_prep]]

Create the LLM instance with speculative decoding enabled. The engine loads both the target model and draft mechanism, setting up the verification pipeline.

'''Initialization process:'''
1. Load target model weights
2. Load draft model/mechanism based on method
3. Set up speculative verification pipeline
4. Allocate buffers for draft token management

=== Step 4: Prepare Prompts for Speculation ===
[[step::Principle:vllm-project_vllm_speculative_prompt_prep]]

Format input prompts for speculative decoding. Token-level inputs work best as they avoid tokenization overhead during the tight speculation loop.

'''Input preparation:'''
* Pre-tokenize prompts when possible
* Use `TokensPrompt` for direct token input
* Chat template application handled normally

=== Step 5: Execute Speculative Generation ===
[[step::Principle:vllm-project_vllm_speculative_generation]]

Run generation with speculative decoding active. The engine drafts multiple tokens, verifies them against the target model, and accepts matching tokens in parallel.

'''Speculation loop:'''
1. Draft mechanism proposes `num_speculative_tokens`
2. Target model verifies draft in single forward pass
3. Accepted tokens added to sequence
4. Rejected position starts new speculation
5. Metrics track acceptance rate per position

=== Step 6: Analyze Speculation Metrics ===
[[step::Principle:vllm-project_vllm_speculative_metrics]]

Collect and analyze speculation performance metrics to understand efficiency gains. Key metrics include acceptance rate, mean acceptance length, and per-position acceptance.

'''Metrics available:'''
* `spec_decode_num_drafts`: Total speculation attempts
* `spec_decode_num_draft_tokens`: Tokens proposed
* `spec_decode_num_accepted_tokens`: Tokens accepted
* `spec_decode_num_accepted_tokens_per_pos`: Per-position breakdown
* Mean acceptance length indicates speedup potential

== Execution Diagram ==
{{#mermaid:graph TD
    A[Select Speculative Method] --> B[Configure Speculative Parameters]
    B --> C[Initialize Speculative Engine]
    C --> D[Prepare Prompts for Speculation]
    D --> E[Execute Speculative Generation]
    E --> F[Analyze Speculation Metrics]
}}

== Related Pages ==
* [[step::Principle:vllm-project_vllm_spec_method_selection]]
* [[step::Principle:vllm-project_vllm_speculative_engine_init]]
* [[step::Principle:vllm-project_vllm_speculative_prompt_prep]]
* [[step::Principle:vllm-project_vllm_speculative_prompt_prep]]
* [[step::Principle:vllm-project_vllm_speculative_generation]]
* [[step::Principle:vllm-project_vllm_speculative_metrics]]
