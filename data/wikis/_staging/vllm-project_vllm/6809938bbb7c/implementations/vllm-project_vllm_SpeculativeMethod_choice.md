{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Paper|Speculative Decoding|https://arxiv.org/abs/2211.17192]]
|-
! Domains
| [[domain::NLP]], [[domain::Inference]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Pattern documentation for selecting the appropriate speculative decoding method based on use case and model architecture.

=== Description ===

vLLM supports multiple speculative decoding strategies, each with different trade-offs:
- **ngram:** Uses prompt patterns for speculation (no extra model)
- **eagle/eagle3:** Trained draft heads for high acceptance
- **draft_model:** Separate smaller model for speculation
- **mtp:** Multi-token prediction heads
- **medusa:** Multiple draft heads for parallel speculation
- **mlp_speculator:** Lightweight MLP-based speculation

=== Usage ===

Select speculative method when:
- Optimizing inference latency
- Trading off complexity vs. speedup
- Working with specific model architectures
- Balancing memory usage with performance

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' vllm/config/speculative.py
* '''Lines:''' L42-49

=== Pattern Specification ===
<syntaxhighlight lang="python">
# Available speculative methods
SpeculativeMethod = Literal[
    "ngram",         # N-gram based (no extra model)
    "eagle",         # EAGLE draft heads
    "eagle3",        # EAGLE-3 improved
    "mtp",           # Multi-token prediction
    "medusa",        # Medusa multi-head
    "mlp_speculator", # MLP-based speculation
    "draft_model",   # Separate draft model
    "suffix",        # Suffix-based
]
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| method || str || Yes || Speculative method name from supported list
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| method || str || Selected method for speculative_config
|}

== Usage Examples ==

=== Method Selection Guide ===
<syntaxhighlight lang="python">
# Method selection based on requirements

# 1. ngram: Simplest, no extra memory, moderate speedup
# Best for: Quick setup, memory-constrained, repetitive text
method = "ngram"

# 2. eagle/eagle3: Trained draft heads, high acceptance
# Best for: Maximum speedup, supported models
method = "eagle"

# 3. draft_model: Flexible, any compatible model
# Best for: Custom draft models, model-agnostic approach
method = "draft_model"

# 4. mtp: Multi-token prediction
# Best for: Models with MTP heads trained
method = "mtp"

# 5. medusa: Multiple speculative heads
# Best for: Medusa-trained models
method = "medusa"
</syntaxhighlight>

=== Method Comparison ===
<syntaxhighlight lang="python">
"""
Method Comparison Table:

| Method      | Extra Model? | Memory | Speedup | Setup Complexity |
|-------------|--------------|--------|---------|------------------|
| ngram       | No           | Low    | 1.5-2x  | Easy             |
| eagle/eagle3| Draft heads  | Medium | 2-3x    | Medium           |
| draft_model | Yes (small)  | High   | 2-3x    | Medium           |
| mtp         | MTP heads    | Medium | 2-3x    | Requires training|
| medusa      | Multi-heads  | Medium | 2-3x    | Requires training|
| mlp_speculator| MLP        | Low    | 1.5-2x  | Medium           |

Recommendations by use case:
- Quick start: ngram
- Maximum speedup: eagle (if supported) or draft_model
- Memory constrained: ngram or mlp_speculator
- Custom models: draft_model
"""
</syntaxhighlight>

=== Checking Model Support ===
<syntaxhighlight lang="python">
# Some models have specific speculative support

# Models with EAGLE support (check vLLM docs for current list):
# - Llama family
# - Vicuna
# - Some Mistral variants

# Models with MTP support:
# - Models trained with multi-token prediction heads

# Universal methods (work with any model):
# - ngram
# - draft_model (with compatible draft)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Speculative_Method_Selection]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
