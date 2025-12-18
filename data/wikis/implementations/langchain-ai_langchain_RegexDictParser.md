{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Output Parsing]], [[domain::Pattern Matching]], [[domain::Key-Value Extraction]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Output parser `RegexDictParser` that extracts labeled key-value pairs from LLM output using regex.

=== Description ===

The `RegexDictParser` class extracts multiple named values from LLM output using a configurable regex pattern. Unlike `RegexParser`, it uses labeled patterns where each output key has an associated format string (label) that identifies its value in the text. It can optionally skip values matching a "no update" sentinel.

=== Usage ===

Use this parser when LLM output contains labeled fields like "Name: value" or "Category: value" and you want to extract specific labeled items into a dictionary.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/output_parsers/regex_dict.py libs/langchain/langchain_classic/output_parsers/regex_dict.py]
* '''Lines:''' 1-43

=== Signature ===
<syntaxhighlight lang="python">
class RegexDictParser(BaseOutputParser[dict[str, str]]):
    """Parse the output of an LLM call into a Dictionary using a regex."""

    regex_pattern: str = r"{}:\s?([^.'\n']*)\.?"
    """The regex pattern with {} placeholder for the label."""
    output_key_to_format: dict[str, str]
    """Mapping from output keys to their text labels."""
    no_update_value: str | None = None
    """Value to skip (treat as 'no update')."""

    def parse(self, text: str) -> dict[str, str]:
        """Parse the output of an LLM call."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers import RegexDictParser
</syntaxhighlight>

== I/O Contract ==

=== Configuration ===
{| class="wikitable"
|-
! Name !! Type !! Default !! Description
|-
| regex_pattern || str || r"{}:\s?([^.'\n']*)\.?" || Pattern with {} for label
|-
| output_key_to_format || dict[str, str] || Required || Key → label mapping
|-
| no_update_value || str | None || None || Value to skip in output
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
| dict[str, str] || Dictionary with output keys and extracted values
|}

=== Exceptions ===
{| class="wikitable"
|-
! Exception !! Condition
|-
| ValueError || No match found for an output key
|-
| ValueError || Multiple matches found for an output key
|}

== Usage Examples ==

=== Basic Labeled Extraction ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers import RegexDictParser

# Extract labeled fields
parser = RegexDictParser(
    output_key_to_format={
        "product_name": "Product",
        "price": "Price",
        "category": "Category",
    }
)

text = """Product: Wireless Mouse
Price: $29.99
Category: Electronics"""

result = parser.parse(text)
print(result)
# {'product_name': 'Wireless Mouse', 'price': '$29.99', 'category': 'Electronics'}
</syntaxhighlight>

=== With No-Update Sentinel ===
<syntaxhighlight lang="python">
# Skip fields with "N/A" value
parser = RegexDictParser(
    output_key_to_format={
        "name": "Name",
        "phone": "Phone",
        "email": "Email",
    },
    no_update_value="N/A",
)

text = """Name: John Doe
Phone: N/A
Email: john@example.com"""

result = parser.parse(text)
print(result)
# {'name': 'John Doe', 'email': 'john@example.com'}
# Note: 'phone' is excluded because value was "N/A"
</syntaxhighlight>

=== Custom Regex Pattern ===
<syntaxhighlight lang="python">
# Custom pattern for different format
parser = RegexDictParser(
    regex_pattern=r"\[{}\]: (.+?)(?:\n|$)",  # [Label]: value format
    output_key_to_format={
        "status": "Status",
        "priority": "Priority",
    },
)

text = """[Status]: Completed
[Priority]: High"""

result = parser.parse(text)
print(result)  # {'status': 'Completed', 'priority': 'High'}
</syntaxhighlight>

=== With LLM Chain ===
<syntaxhighlight lang="python">
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

parser = RegexDictParser(
    output_key_to_format={
        "decision": "Decision",
        "reasoning": "Reasoning",
        "confidence": "Confidence",
    }
)

prompt = PromptTemplate.from_template(
    "Review this proposal and respond with:\n"
    "Decision: <approve/reject>\n"
    "Reasoning: <brief explanation>\n"
    "Confidence: <high/medium/low>\n\n"
    "Proposal: {proposal}"
)

llm = ChatOpenAI(model="gpt-4")
chain = prompt | llm | parser

result = chain.invoke({
    "proposal": "Increase marketing budget by 20% for Q4"
})
print(result)
# {'decision': 'approve', 'reasoning': '...', 'confidence': 'medium'}
</syntaxhighlight>

== Related Pages ==
* [[related_to::Implementation:langchain-ai_langchain_RegexParser]]
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]

