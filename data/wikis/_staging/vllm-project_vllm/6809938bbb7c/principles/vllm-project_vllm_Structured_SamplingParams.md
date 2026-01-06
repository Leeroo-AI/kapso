{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Structured_Output]], [[domain::Sampling]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

The practice of integrating structured output constraints into sampling parameters for constrained generation.

=== Description ===

Structured SamplingParams combines constraint configuration with generation parameters:

1. **Constraint Integration:** Attach StructuredOutputsParams to SamplingParams
2. **Token Limits:** Set appropriate max_tokens for constrained outputs
3. **Temperature Tuning:** Balance creativity vs. constraint compliance
4. **Stop Sequences:** Configure appropriate stop conditions
5. **Logit Processing:** Enable constraint-aware logit masking

=== Usage ===

Configure structured SamplingParams when:
- Running constrained generation requests
- Building extraction pipelines
- Implementing structured APIs
- Creating classification endpoints

== Theoretical Basis ==

'''Integration Pattern:'''

<syntaxhighlight lang="python">
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# Full configuration
sampling_params = SamplingParams(
    # Constraint configuration
    structured_outputs=StructuredOutputsParams(
        json={"type": "object", "properties": {...}},
    ),

    # Generation parameters
    max_tokens=200,
    temperature=0.7,
    top_p=0.95,

    # Stop conditions
    stop=["\n\n"],
)
</syntaxhighlight>

'''Parameter Interactions:'''

<syntaxhighlight lang="text">
Parameter           Effect on Constrained Output
─────────────────   ─────────────────────────────
temperature=0       Deterministic, may miss valid paths
temperature=0.7     Good balance for structured output
temperature=1.0+    May increase constraint violations

max_tokens (low)    Risk of truncated JSON/structures
max_tokens (high)   Safe for complex schemas
</syntaxhighlight>

'''Logit Processing Flow:'''

<syntaxhighlight lang="text">
Generation Step:
┌─────────────────────────────────────────────────────┐
│ 1. Model produces logits for all tokens            │
│ 2. Constraint processor masks invalid tokens       │
│ 3. Sampling applies temperature/top_p to valid     │
│ 4. Selected token advances constraint state        │
│ 5. Repeat until stop condition or max_tokens       │
└─────────────────────────────────────────────────────┘
</syntaxhighlight>

'''Recommended Settings by Constraint Type:'''

<syntaxhighlight lang="python">
# JSON schema - allow enough tokens for full structure
json_params = SamplingParams(
    structured_outputs=StructuredOutputsParams(json=schema),
    max_tokens=500,      # Generous for nested objects
    temperature=0.3,     # Lower for consistency
)

# Choice - minimal tokens needed
choice_params = SamplingParams(
    structured_outputs=StructuredOutputsParams(choice=["a", "b", "c"]),
    max_tokens=5,        # Choices are short
    temperature=0,       # Deterministic selection
)

# Regex - depends on pattern
regex_params = SamplingParams(
    structured_outputs=StructuredOutputsParams(regex=r"\d{3}-\d{4}"),
    max_tokens=20,       # Match expected length
    temperature=0,       # Deterministic
)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_SamplingParams_structured]]
