{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Structured_Output]], [[domain::Output_Processing]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

The practice of parsing and validating constrained generation outputs into structured data types.

=== Description ===

Structured Output Parsing converts generated text into usable data:

1. **JSON Parsing:** Convert JSON string to dict/object
2. **Pydantic Validation:** Validate against original schema
3. **Direct Use:** Use choice/regex outputs directly
4. **Error Handling:** Handle malformed outputs gracefully
5. **Type Conversion:** Convert to appropriate Python types

=== Usage ===

Parse structured outputs when:
- Processing JSON schema generation results
- Validating extraction outputs
- Converting to typed objects
- Building data pipelines

== Theoretical Basis ==

'''Parsing by Constraint Type:'''

<syntaxhighlight lang="python">
def parse_output(text: str, constraint_type: str):
    """Parse output based on constraint type."""
    if constraint_type == "json":
        # Parse JSON and optionally validate
        import json
        return json.loads(text)

    elif constraint_type == "choice":
        # Direct string use
        return text.strip()

    elif constraint_type == "regex":
        # Direct string use (already matches pattern)
        return text.strip()

    elif constraint_type == "grammar":
        # May need grammar-specific parser
        return text.strip()
</syntaxhighlight>

'''Pydantic Validation:'''

<syntaxhighlight lang="python">
from pydantic import BaseModel, ValidationError
import json

class Person(BaseModel):
    name: str
    age: int

def parse_and_validate(text: str, model: type[BaseModel]):
    """Parse JSON and validate against Pydantic model."""
    try:
        data = json.loads(text)
        return model.model_validate(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    except ValidationError as e:
        raise ValueError(f"Validation failed: {e}")
</syntaxhighlight>

'''Error Handling Strategies:'''

<syntaxhighlight lang="text">
Error Type          Strategy
─────────────────   ─────────────────────────────────
JSON decode error   Retry with stricter constraints
Validation error    Log and use default values
Truncated output    Increase max_tokens
Unexpected format   Check constraint configuration
</syntaxhighlight>

'''Output Quality Checks:'''

<syntaxhighlight lang="python">
def check_output_quality(output, constraint_type):
    """Verify output quality."""
    quality = {
        "complete": output.outputs[0].finish_reason == "stop",
        "has_content": len(output.outputs[0].text.strip()) > 0,
        "constraint_type": constraint_type,
    }

    if constraint_type == "json":
        try:
            import json
            json.loads(output.outputs[0].text)
            quality["valid_json"] = True
        except:
            quality["valid_json"] = False

    return quality
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_structured_output_parse]]
