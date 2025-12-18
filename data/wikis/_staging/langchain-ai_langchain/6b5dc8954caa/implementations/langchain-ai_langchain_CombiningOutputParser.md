{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Output Parsing]], [[domain::Parser Composition]], [[domain::Multi-Output]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Output parser `CombiningOutputParser` that runs multiple parsers on different sections of LLM output.

=== Description ===

The `CombiningOutputParser` class combines multiple output parsers to extract different types of information from a single LLM response. It expects the LLM to produce outputs separated by double newlines, with each section parsed by the corresponding parser. Results are merged into a single dictionary.

=== Usage ===

Use this parser when you need to extract multiple structured pieces of information from a single LLM call. For example, extracting both a summary and a list of keywords from a document analysis response.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/output_parsers/combining.py libs/langchain/langchain_classic/output_parsers/combining.py]
* '''Lines:''' 1-59

=== Signature ===
<syntaxhighlight lang="python">
class CombiningOutputParser(BaseOutputParser[dict[str, Any]]):
    """Combine multiple output parsers into one."""

    parsers: list[BaseOutputParser]

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""

    def parse(self, text: str) -> dict[str, Any]:
        """Parse the output of an LLM call."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers import CombiningOutputParser
</syntaxhighlight>

== I/O Contract ==

=== Configuration ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| parsers || list[BaseOutputParser] || Yes || At least 2 parsers to combine
|}

=== Constraints ===
* Requires at least 2 parsers
* Cannot nest CombiningOutputParser instances
* Cannot include list parsers

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| text || str || Yes || LLM output with sections separated by double newlines
|}

=== Outputs ===
{| class="wikitable"
|-
! Type !! Description
|-
| dict[str, Any] || Merged results from all parsers
|}

== Usage Examples ==

=== Combine Multiple Parsers ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers import (
    CombiningOutputParser,
    RegexParser,
)
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema

# Create individual parsers
regex_parser = RegexParser(
    regex=r"Summary: (.*)",
    output_keys=["summary"],
)

structured_parser = StructuredOutputParser.from_response_schemas([
    ResponseSchema(name="sentiment", description="Overall sentiment"),
    ResponseSchema(name="confidence", description="Confidence score"),
])

# Combine parsers
combined = CombiningOutputParser(
    parsers=[regex_parser, structured_parser]
)

# Get format instructions for the prompt
print(combined.get_format_instructions())
</syntaxhighlight>

=== Parse Multi-Section Output ===
<syntaxhighlight lang="python">
# LLM output with double-newline separated sections
llm_output = """Summary: This document discusses AI safety.

```json
{"sentiment": "positive", "confidence": "0.85"}
```"""

result = combined.parse(llm_output)
print(result)
# {'summary': 'This document discusses AI safety.',
#  'sentiment': 'positive', 'confidence': '0.85'}
</syntaxhighlight>

=== Integration with LLM Chain ===
<syntaxhighlight lang="python">
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

prompt = PromptTemplate.from_template(
    "Analyze this text:\n{text}\n\n"
    "{format_instructions}"
)

llm = ChatOpenAI(model="gpt-4")
chain = prompt | llm | combined

result = chain.invoke({
    "text": "Great product, highly recommend!",
    "format_instructions": combined.get_format_instructions()
})
</syntaxhighlight>

== Related Pages ==
* [[related_to::Implementation:langchain-ai_langchain_StructuredOutputParser]]
* [[related_to::Implementation:langchain-ai_langchain_RegexParser]]
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]

