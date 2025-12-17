= Speculative Method Selection Principle =

== Metadata ==
{| class="wikitable"
|-
! Field !! Value
|-
| Knowledge Sources || [https://arxiv.org/abs/2401.15077 EAGLE: Speculative Sampling via Feature-Level Autoregression], [https://arxiv.org/abs/2404.19124 MLP Speculative Decoding], [https://arxiv.org/abs/2302.01318 Speculative Sampling], [https://arxiv.org/abs/2411.04975 Suffix Decoding], vLLM Documentation
|-
| Domains || Speculative Decoding, Model Optimization, LLM Inference
|-
| Last Updated || 2025-12-17
|}

== Overview ==

The Speculative Method Selection principle defines the strategy for choosing appropriate speculative decoding methods to accelerate LLM inference while maintaining output quality. This principle guides the selection among various methods including EAGLE (feature-level autoregression), N-gram prompt lookup, MLP speculator, draft model speculation, and suffix decoding based on workload characteristics and resource constraints.

== Description ==

Speculative decoding is a technique that improves inter-token latency in memory-bound LLM inference by generating multiple candidate tokens speculatively and verifying them in parallel. The method selection principle encompasses:

=== Core Selection Criteria ===

* '''Model Architecture Compatibility''': Different speculative methods require specific model architectures (e.g., EAGLE requires trained draft heads, MLP requires speculator weights)
* '''Resource Availability''': Memory and compute constraints determine feasibility (e.g., draft models require additional GPU memory)
* '''Workload Characteristics''': Repetitive vs. creative tasks benefit from different methods
* '''Acceptance Rate Requirements''': Target speedup goals influence method selection

=== Available Methods ===

* '''ngram''': Pattern-matching based on prompt n-grams, no additional model needed
* '''eagle/eagle3''': Feature-level autoregressive drafting using trained EAGLE heads
* '''mlp_speculator''': Lightweight MLP networks for token prediction
* '''draft_model''': Separate smaller draft model (not yet implemented in v1)
* '''suffix''': Dynamic pattern-matching against prompt and generation history
* '''mtp''': Multi-token prediction for models with built-in MTP layers (DeepSeek-V3, etc.)

=== Selection Guidelines ===

* Use '''ngram''' for: Zero-cost speculation, repetitive patterns, code generation
* Use '''EAGLE''' for: Best acceptance rates, when EAGLE weights are available
* Use '''MLP speculator''' for: Balanced performance with minimal overhead
* Use '''suffix''' for: Agentic loops, self-reflection, high repetition tasks
* Use '''mtp''' for: Models with native multi-token prediction support

== Usage Context ==

This principle applies when:

* Configuring vLLM LLM instances for production deployment
* Optimizing inference latency for specific workloads
* Balancing throughput vs. latency requirements
* Resource-constrained environments requiring efficiency

The selection is typically made at LLM initialization time through the <code>speculative_config</code> parameter and cannot be changed during inference.

== Design Considerations ==

=== Trade-offs ===

* '''Accuracy vs. Speed''': All methods are theoretically lossless but may have numerical differences
* '''Memory vs. Performance''': Draft models and EAGLE heads require additional memory
* '''Flexibility vs. Complexity''': Simple methods (ngram) are easier to deploy but less effective
* '''Acceptance Rate vs. Overhead''': More sophisticated methods have higher per-token cost but better acceptance

=== Constraints ===

* EAGLE and MLP speculators require pre-trained weights matching the target model
* Draft model method is not yet implemented in vLLM v1
* Suffix decoding requires Arctic Inference library
* MTP method requires models with native multi-token prediction layers
* Tensor parallelism support varies by method

== Related Principles ==

* [[implements::vllm-project_vllm_SpeculativeConfig]] - Configuration implementation
* Prompt Engineering Principles - Workload characteristics influence method selection
* Model Selection Principles - Base model choice affects speculative method compatibility
* Resource Allocation Principles - Memory and compute constraints

== See Also ==

* [[implemented_by::Implementation:vllm-project_vllm_SpeculativeConfig]]
* vLLM Speculative Decoding Documentation
* EAGLE Model Repository (yuhuili on HuggingFace)
* MLP Speculator Models (ibm-ai-platform on HuggingFace)
