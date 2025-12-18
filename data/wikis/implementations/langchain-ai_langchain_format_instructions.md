{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Output Parsing]], [[domain::Prompt Engineering]], [[domain::Format Templates]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Module `format_instructions` providing template strings for guiding LLM output formatting.

=== Description ===

This module contains predefined format instruction templates used by various output parsers to tell LLMs how to structure their responses. These templates include examples and schemas for JSON, YAML, Pydantic models, and Pandas DataFrame operations.

=== Usage ===

These constants are used internally by output parsers like `StructuredOutputParser`, `YamlOutputParser`, and `PandasDataFrameOutputParser`. They can also be imported directly for custom parsing scenarios.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/output_parsers/format_instructions.py libs/langchain/langchain_classic/output_parsers/format_instructions.py]
* '''Lines:''' 1-80

=== Constants ===
<syntaxhighlight lang="python">
# For StructuredOutputParser - full instructions with schema
STRUCTURED_FORMAT_INSTRUCTIONS = """The output should be a markdown code snippet..."""

# For StructuredOutputParser - minimal schema only
STRUCTURED_FORMAT_SIMPLE_INSTRUCTIONS = """
```json
{{
{format}
}}
```"""

# For Pydantic-based parsers with JSON schema
PYDANTIC_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance..."""

# For YamlOutputParser with examples
YAML_FORMAT_INSTRUCTIONS = """The output should be formatted as a YAML instance..."""

# For PandasDataFrameOutputParser
PANDAS_DATAFRAME_FORMAT_INSTRUCTIONS = """The output should be formatted as a string..."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers.format_instructions import (
    STRUCTURED_FORMAT_INSTRUCTIONS,
    PYDANTIC_FORMAT_INSTRUCTIONS,
    YAML_FORMAT_INSTRUCTIONS,
    PANDAS_DATAFRAME_FORMAT_INSTRUCTIONS,
)
</syntaxhighlight>

== Format Templates ==

=== STRUCTURED_FORMAT_INSTRUCTIONS ===
{| class="wikitable"
|-
! Placeholder !! Description
|-
| {format} || Schema fields with descriptions
|}

Used by `StructuredOutputParser` to produce JSON in markdown code blocks.

=== PYDANTIC_FORMAT_INSTRUCTIONS ===
{| class="wikitable"
|-
! Placeholder !! Description
|-
| {schema} || Full JSON schema from Pydantic model
|}

Includes examples of well-formatted vs poorly-formatted JSON instances.

=== YAML_FORMAT_INSTRUCTIONS ===
{| class="wikitable"
|-
! Placeholder !! Description
|-
| {schema} || JSON schema defining expected structure
|}

Includes multiple examples showing array and object formatting.

=== PANDAS_DATAFRAME_FORMAT_INSTRUCTIONS ===
{| class="wikitable"
|-
! Placeholder !! Description
|-
| {columns} || Available column names from DataFrame
|}

Documents the query syntax: operation:column[array] format.

== Usage Examples ==

=== Custom Parser with Format Instructions ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers.format_instructions import (
    STRUCTURED_FORMAT_INSTRUCTIONS,
)

# Create custom format with specific fields
schema = '''    "name": string  // The person's name
    "age": integer  // The person's age'''

instructions = STRUCTURED_FORMAT_INSTRUCTIONS.format(format=schema)
print(instructions)
</syntaxhighlight>

=== Using with PromptTemplate ===
<syntaxhighlight lang="python">
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers.format_instructions import (
    YAML_FORMAT_INSTRUCTIONS,
)
import json

schema = {"name": {"type": "string"}, "skills": {"type": "array"}}
instructions = YAML_FORMAT_INSTRUCTIONS.format(schema=json.dumps(schema))

prompt = PromptTemplate.from_template(
    "Extract information from this bio:\n{bio}\n\n{format_instructions}"
)

formatted = prompt.format(
    bio="John is a software engineer who knows Python and Go.",
    format_instructions=instructions
)
</syntaxhighlight>

=== DataFrame Query Instructions ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers.format_instructions import (
    PANDAS_DATAFRAME_FORMAT_INSTRUCTIONS,
)

columns = "name, age, salary, department"
instructions = PANDAS_DATAFRAME_FORMAT_INSTRUCTIONS.format(columns=columns)

# Supported query formats documented:
# - column:name -> get column
# - row:1 -> get row
# - column:salary[1,2] -> get salary for rows 1,2
# - mean:salary[1..5] -> mean of salary rows 1-5
</syntaxhighlight>

== Related Pages ==
* [[used_by::Implementation:langchain-ai_langchain_StructuredOutputParser]]
* [[used_by::Implementation:langchain-ai_langchain_YamlOutputParser]]
* [[used_by::Implementation:langchain-ai_langchain_PandasDataFrameOutputParser]]

