{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Structured_Output]], [[domain::Constraints]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Concrete patterns for defining output constraints using JSON schemas, regex patterns, choices, or grammars in vLLM.

=== Description ===

vLLM supports four mutually exclusive constraint types:
- **json:** JSON schema from Pydantic model or raw dict
- **regex:** Regular expression pattern for output format
- **choice:** List of allowed output strings
- **grammar:** EBNF grammar specification

These patterns define how to construct constraints for `StructuredOutputsParams`.

=== Usage ===

Use these patterns when:
- Building structured data extraction pipelines
- Creating classification systems
- Enforcing specific output formats
- Implementing function calling interfaces

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' vllm/sampling_params.py
* '''Lines:''' L32-99

=== Constraint Types ===
<syntaxhighlight lang="python">
# JSON schema constraint (most common)
json: str | dict | None = None

# Regex pattern constraint
regex: str | None = None

# Choice list constraint
choice: list[str] | None = None

# EBNF grammar constraint
grammar: str | None = None

# Simple JSON object mode
json_object: bool | None = None
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from pydantic import BaseModel
# For JSON schema generation
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| json || str | dict || No || JSON schema definition
|-
| regex || str || No || Regular expression pattern
|-
| choice || list[str] || No || Allowed output values
|-
| grammar || str || No || EBNF grammar string
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| constraint || dict || Constraint specification for StructuredOutputsParams
|}

== Usage Examples ==

=== JSON Schema from Pydantic ===
<syntaxhighlight lang="python">
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# Define schema as Pydantic model
class ExtractedInfo(BaseModel):
    name: str
    email: str
    phone: str

# Convert to JSON schema
json_schema = ExtractedInfo.model_json_schema()

# Create constraint
structured = StructuredOutputsParams(json=json_schema)

sampling_params = SamplingParams(
    max_tokens=200,
    structured_outputs=structured,
)
</syntaxhighlight>

=== Regex Pattern ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# Define regex for email extraction
email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

# Create constraint
structured = StructuredOutputsParams(regex=email_pattern)

sampling_params = SamplingParams(
    max_tokens=50,
    structured_outputs=structured,
)
</syntaxhighlight>

=== Choice Constraint ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# Define allowed choices for sentiment classification
sentiments = ["positive", "negative", "neutral"]

# Create constraint
structured = StructuredOutputsParams(choice=sentiments)

sampling_params = SamplingParams(
    max_tokens=10,
    structured_outputs=structured,
)
</syntaxhighlight>

=== Grammar Constraint ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# Define EBNF grammar for arithmetic expressions
arithmetic_grammar = """
?start: expr
expr: term (("+"|"-") term)*
term: factor (("*"|"/") factor)*
factor: NUMBER | "(" expr ")"
NUMBER: /[0-9]+/
"""

# Create constraint
structured = StructuredOutputsParams(grammar=arithmetic_grammar)

sampling_params = SamplingParams(
    max_tokens=50,
    structured_outputs=structured,
)
</syntaxhighlight>

=== Nested JSON Schema ===
<syntaxhighlight lang="python">
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

class Address(BaseModel):
    street: str
    city: str
    country: str

class Company(BaseModel):
    name: str
    industry: str
    headquarters: Address
    employee_count: int

# Generate nested schema
schema = Company.model_json_schema()

structured = StructuredOutputsParams(json=schema)

sampling_params = SamplingParams(
    max_tokens=300,
    structured_outputs=structured,
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Constraint_Definition]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
