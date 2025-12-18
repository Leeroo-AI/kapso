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

Concrete tool for initializing StructuredOutputsParams to configure output constraints for structured generation.

=== Description ===

`StructuredOutputsParams` initialization creates a constraint configuration that can be passed to `SamplingParams`. It supports:
- **json:** JSON schema from Pydantic or raw dict
- **regex:** Regular expression pattern
- **choice:** List of allowed output strings
- **grammar:** EBNF grammar specification
- **json_object:** Simple JSON object mode

=== Usage ===

Initialize StructuredOutputsParams when:
- Setting up constrained generation
- Building extraction pipelines
- Implementing classification
- Enforcing specific output formats

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' vllm/sampling_params.py
* '''Lines:''' L32-99

=== Signature ===
<syntaxhighlight lang="python">
class StructuredOutputsParams(msgspec.Struct):
    json: str | dict | type[BaseModel] | None = None
    regex: str | None = None
    choice: list[str] | None = None
    grammar: str | None = None
    json_object: bool | None = None
    disable_fallback: bool = False
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm.sampling_params import StructuredOutputsParams
# Or via SamplingParams
from vllm import SamplingParams
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| json || str | dict | BaseModel || No || JSON schema constraint
|-
| regex || str || No || Regular expression pattern
|-
| choice || list[str] || No || Allowed output values
|-
| grammar || str || No || EBNF grammar string
|-
| json_object || bool || No || Enable simple JSON mode
|-
| disable_fallback || bool || No || Strict constraint enforcement
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| params || StructuredOutputsParams || Configured constraint parameters
|}

== Usage Examples ==

=== JSON Schema Constraint ===
<syntaxhighlight lang="python">
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

class PersonInfo(BaseModel):
    name: str
    age: int
    occupation: str

# Create structured outputs params
structured = StructuredOutputsParams(
    json=PersonInfo.model_json_schema(),
)

# Integrate with sampling params
sampling_params = SamplingParams(
    max_tokens=150,
    temperature=0.7,
    structured_outputs=structured,
)

# Use in generation
llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")
outputs = llm.generate(
    ["Extract person info: John is a 30-year-old engineer."],
    sampling_params,
)

print(outputs[0].outputs[0].text)
# {"name": "John", "age": 30, "occupation": "engineer"}
</syntaxhighlight>

=== Regex Constraint ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# Date format: YYYY-MM-DD
structured = StructuredOutputsParams(
    regex=r"\d{4}-\d{2}-\d{2}",
)

sampling_params = SamplingParams(
    max_tokens=20,
    structured_outputs=structured,
)

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")
outputs = llm.generate(
    ["What is today's date? Respond in YYYY-MM-DD format:"],
    sampling_params,
)
# Output will match: 2024-01-15
</syntaxhighlight>

=== Choice Constraint ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# Sentiment classification
structured = StructuredOutputsParams(
    choice=["positive", "negative", "neutral"],
)

sampling_params = SamplingParams(
    max_tokens=10,
    structured_outputs=structured,
)

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")
outputs = llm.generate(
    ["Classify sentiment: 'I love this product!' Answer:"],
    sampling_params,
)
# Output: "positive"
</syntaxhighlight>

=== Grammar Constraint ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# SQL SELECT statement grammar
sql_grammar = """
?start: select_stmt
select_stmt: "SELECT" columns "FROM" table
columns: column ("," column)*
column: WORD
table: WORD
WORD: /[a-zA-Z_][a-zA-Z0-9_]*/
"""

structured = StructuredOutputsParams(grammar=sql_grammar)

sampling_params = SamplingParams(
    max_tokens=50,
    structured_outputs=structured,
)
</syntaxhighlight>

=== Strict Mode ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# Strict enforcement - no fallback to unconstrained
structured = StructuredOutputsParams(
    choice=["yes", "no"],
    disable_fallback=True,  # Strict mode
)

sampling_params = SamplingParams(
    max_tokens=5,
    structured_outputs=structured,
)
# Output guaranteed to be "yes" or "no"
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_StructuredOutputsParams_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
