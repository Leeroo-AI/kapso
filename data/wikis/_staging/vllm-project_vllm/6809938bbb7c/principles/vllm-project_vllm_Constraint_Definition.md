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

The practice of defining output constraints using JSON schemas, regex patterns, choices, or grammars for structured generation.

=== Description ===

Constraint Definition involves specifying the format and structure that generated outputs must adhere to:

1. **JSON Schema:** Define structured objects using Pydantic models or raw JSON schema
2. **Regex Patterns:** Match outputs against regular expressions
3. **Choice Lists:** Restrict outputs to specific allowed values
4. **EBNF Grammar:** Define formal grammar rules for complex structures
5. **JSON Object:** Simple JSON formatting without strict schema

=== Usage ===

Define constraints when:
- Extracting structured data from unstructured text
- Building function calling or tool use systems
- Enforcing specific output formats for downstream processing
- Creating classification or categorization systems

== Theoretical Basis ==

'''Constraint Types:'''

<syntaxhighlight lang="python">
# Four main constraint types (mutually exclusive)
constraints = {
    "json": dict | str,      # JSON schema or Pydantic model
    "regex": str,            # Regular expression pattern
    "choice": list[str],     # Allowed output values
    "grammar": str,          # EBNF grammar definition
}
</syntaxhighlight>

'''JSON Schema via Pydantic:'''

<syntaxhighlight lang="python">
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

# Convert to JSON schema
schema = Person.model_json_schema()
# {
#   "properties": {
#     "name": {"type": "string"},
#     "age": {"type": "integer"},
#     "occupation": {"type": "string"}
#   },
#   "required": ["name", "age", "occupation"],
#   "type": "object"
# }
</syntaxhighlight>

'''Regex Constraints:'''

<syntaxhighlight lang="python">
# Match email pattern
email_regex = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

# Match phone number
phone_regex = r"\d{3}-\d{3}-\d{4}"

# Match ISO date
date_regex = r"\d{4}-\d{2}-\d{2}"
</syntaxhighlight>

'''Choice Lists:'''

<syntaxhighlight lang="python">
# Classification choices
sentiment_choices = ["positive", "negative", "neutral"]

# Category choices
category_choices = ["sports", "politics", "technology", "entertainment"]

# Boolean-like choices
yes_no_choices = ["yes", "no"]
</syntaxhighlight>

'''EBNF Grammar:'''

<syntaxhighlight lang="python">
# Arithmetic expression grammar
arithmetic_grammar = """
?start: expr
expr: term (("+"|"-") term)*
term: factor (("*"|"/") factor)*
factor: NUMBER | "(" expr ")"
NUMBER: /\d+(\.\d+)?/
"""
</syntaxhighlight>

'''Choosing the Right Constraint:'''

<syntaxhighlight lang="text">
Use Case                    Recommended Constraint
─────────────────────────   ─────────────────────
Structured data extraction  JSON schema
Pattern matching            Regex
Classification              Choice
Complex formats             Grammar
Simple JSON output          json_object=True
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_StructuredOutputsParams_types]]
