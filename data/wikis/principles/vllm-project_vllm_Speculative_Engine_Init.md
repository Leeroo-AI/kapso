{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Speculative Decoding|https://arxiv.org/abs/2211.17192]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Inference]], [[domain::Systems]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

The process of initializing an inference engine with speculative decoding capabilities, including loading target and draft models.

=== Description ===

Speculative Engine Initialization sets up the infrastructure for accelerated inference:

1. **Target Model Loading:** Load the main model for generation
2. **Draft Setup:** Initialize speculation mechanism (draft model or heads)
3. **Verification Pipeline:** Configure parallel verification
4. **Memory Allocation:** Plan memory for both models/heads
5. **Scheduler Configuration:** Set up speculative batch scheduling

=== Usage ===

Initialize speculative engines when:
- Deploying latency-sensitive inference
- Serving large models with interactive requirements
- Building real-time applications

== Theoretical Basis ==

'''Initialization Flow:'''

<syntaxhighlight lang="python">
# Conceptual initialization
def init_speculative_engine(model, speculative_config):
    # 1. Load target model
    target = load_model(model)

    # 2. Initialize speculation based on method
    method = speculative_config["method"]

    if method == "ngram":
        # No additional model needed
        spec = NGramSpeculator(
            max_n=speculative_config["prompt_lookup_max"],
            min_n=speculative_config["prompt_lookup_min"],
        )
    elif method in ["eagle", "eagle3"]:
        # Load draft heads
        draft_path = speculative_config["model"]
        spec = EAGLESpeculator(draft_path)
    elif method == "draft_model":
        # Load separate draft model
        draft = load_model(speculative_config["model"])
        spec = DraftModelSpeculator(draft)

    # 3. Create verification pipeline
    verifier = SpeculativeVerifier(
        target=target,
        num_tokens=speculative_config["num_speculative_tokens"],
    )

    # 4. Combine into engine
    return SpeculativeEngine(target, spec, verifier)
</syntaxhighlight>

'''Memory Layout:'''

<syntaxhighlight lang="text">
GPU Memory with Speculation:
┌─────────────────────────────────────────────────────────────────┐
│ Target Model Weights                                            │
├─────────────────────────────────────────────────────────────────┤
│ Target KV Cache                                                 │
├─────────────────────────────────────────────────────────────────┤
│ Draft Model/Heads (if applicable)                               │
├─────────────────────────────────────────────────────────────────┤
│ Draft KV Cache (if applicable)                                  │
├─────────────────────────────────────────────────────────────────┤
│ Verification Buffers                                            │
└─────────────────────────────────────────────────────────────────┘
</syntaxhighlight>

'''Engine Startup Sequence:'''

1. Parse speculative configuration
2. Load target model weights
3. Initialize draft mechanism
4. Allocate verification buffers
5. Warm up speculation pipeline
6. Mark engine ready

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_LLM_speculative]]
