{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Output Parsing]], [[domain::Boolean Extraction]], [[domain::Classification]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Output parser `BooleanOutputParser` that extracts boolean values from LLM text responses.

=== Description ===

The `BooleanOutputParser` class parses LLM output to extract a boolean value. It searches for configurable true/false tokens (default "YES"/"NO") in the response text using case-insensitive regex matching. The parser raises an error if both tokens are found (ambiguous) or neither is found.

=== Usage ===

Use this parser for binary classification tasks, yes/no questions, approval workflows, or any scenario where you need to extract a clear true/false decision from LLM output.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/output_parsers/boolean.py libs/langchain/langchain_classic/output_parsers/boolean.py]
* '''Lines:''' 1-55

=== Signature ===
<syntaxhighlight lang="python">
class BooleanOutputParser(BaseOutputParser[bool]):
    """Parse the output of an LLM call to a boolean."""

    true_val: str = "YES"
    """The string value that should be parsed as True."""
    false_val: str = "NO"
    """The string value that should be parsed as False."""

    def parse(self, text: str) -> bool:
        """Parse the output of an LLM call to a boolean."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers import BooleanOutputParser
</syntaxhighlight>

== I/O Contract ==

=== Configuration ===
{| class="wikitable"
|-
! Name !! Type !! Default !! Description
|-
| true_val || str || "YES" || Token representing True
|-
| false_val || str || "NO" || Token representing False
|}

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| text || str || Yes || LLM output text to parse
|}

=== Outputs ===
{| class="wikitable"
|-
! Type !! Description
|-
| bool || True if true_val found, False if false_val found
|}

=== Exceptions ===
{| class="wikitable"
|-
! Exception !! Condition
|-
| ValueError || Both true_val and false_val found (ambiguous)
|-
| ValueError || Neither true_val nor false_val found
|}

== Usage Examples ==

=== Basic Yes/No Parsing ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers import BooleanOutputParser

parser = BooleanOutputParser()

# Parse affirmative response
result = parser.parse("Yes, I think that is correct.")
print(result)  # True

# Parse negative response
result = parser.parse("No, that is not accurate.")
print(result)  # False
</syntaxhighlight>

=== Custom True/False Tokens ===
<syntaxhighlight lang="python">
parser = BooleanOutputParser(
    true_val="APPROVED",
    false_val="REJECTED"
)

result = parser.parse("After review, the proposal is APPROVED.")
print(result)  # True
</syntaxhighlight>

=== With LLM Chain ===
<syntaxhighlight lang="python">
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

prompt = PromptTemplate.from_template(
    "Is the following statement factually correct? "
    "Answer YES or NO only.\n\nStatement: {statement}"
)

llm = ChatOpenAI(model="gpt-4")
parser = BooleanOutputParser()

chain = prompt | llm | parser
result = chain.invoke({"statement": "The Earth orbits the Sun."})
print(result)  # True
</syntaxhighlight>

== Related Pages ==
* [[related_to::Implementation:langchain-ai_langchain_EnumOutputParser]]
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]

