{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Structured_Output]], [[domain::Configuration]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

The process of configuring structured output parameters that enforce generation constraints during inference.

=== Description ===

StructuredOutputsParams Configuration sets up the constraint enforcement mechanism:

1. **Constraint Selection:** Choose one constraint type (mutually exclusive)
2. **Schema Validation:** Validate JSON schema or regex at configuration time
3. **Backend Selection:** Configure constraint enforcement backend
4. **Fallback Behavior:** Control behavior when constraints can't be satisfied
5. **Integration:** Prepare for SamplingParams integration

=== Usage ===

Configure StructuredOutputsParams when:
- Building data extraction pipelines
- Implementing tool calling systems
- Creating classification endpoints
- Enforcing output format compliance

== Theoretical Basis ==

'''Parameter Structure:'''

<syntaxhighlight lang="python">
class StructuredOutputsParams:
    json: str | dict | None = None        # JSON schema
    regex: str | None = None              # Regex pattern
    choice: list[str] | None = None       # Allowed values
    grammar: str | None = None            # EBNF grammar
    json_object: bool | None = None       # Simple JSON mode
    disable_fallback: bool = False        # Strict enforcement
</syntaxhighlight>

'''Mutual Exclusivity:'''

<syntaxhighlight lang="text">
Only ONE constraint type can be active:
┌─────────────────────────────────────────┐
│ ❌ json + regex          # Invalid      │
│ ❌ choice + grammar      # Invalid      │
│ ✅ json only             # Valid        │
│ ✅ choice only           # Valid        │
└─────────────────────────────────────────┘
</syntaxhighlight>

'''Constraint Backends:'''

<syntaxhighlight lang="python">
# vLLM uses different backends for different constraint types:
backends = {
    "json": "outlines or lm-format-enforcer",
    "regex": "outlines",
    "grammar": "outlines",
    "choice": "logit bias masking",
}
</syntaxhighlight>

'''Fallback Behavior:'''

<syntaxhighlight lang="python">
# With disable_fallback=False (default):
# - Falls back to unconstrained generation if constraint fails
# - May produce outputs that don't match constraints

# With disable_fallback=True:
# - Strictly enforces constraints
# - May produce incomplete outputs if constraints are too restrictive
</syntaxhighlight>

'''Configuration Validation:'''

<syntaxhighlight lang="python">
def validate_config(params: StructuredOutputsParams):
    """Validate structured output configuration."""
    constraints = [
        params.json is not None,
        params.regex is not None,
        params.choice is not None,
        params.grammar is not None,
    ]

    if sum(constraints) > 1:
        raise ValueError("Only one constraint type allowed")

    if params.json and isinstance(params.json, str):
        # JSON string will be parsed
        import json
        json.loads(params.json)

    if params.regex:
        # Validate regex syntax
        import re
        re.compile(params.regex)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_StructuredOutputsParams_init]]
