{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Output Parsing]], [[domain::Pattern Matching]], [[domain::Text Extraction]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Output parser `RegexParser` that extracts structured data from LLM output using regular expressions.

=== Description ===

The `RegexParser` class uses a configurable regex pattern with capture groups to extract multiple fields from LLM output. Each capture group maps to an output key, producing a dictionary of extracted values. An optional default output key handles cases where the pattern doesn't match.

=== Usage ===

Use this parser when LLM output follows a predictable text pattern that can be captured with regex. It's useful for extracting structured information from semi-structured text responses.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/output_parsers/regex.py libs/langchain/langchain_classic/output_parsers/regex.py]
* '''Lines:''' 1-41

=== Signature ===
<syntaxhighlight lang="python">
class RegexParser(BaseOutputParser[dict[str, str]]):
    """Parse the output of an LLM call using a regex."""

    regex: str
    """The regex pattern with capture groups."""
    output_keys: list[str]
    """The keys for each capture group."""
    default_output_key: str | None = None
    """Default key for unmatched text (optional)."""

    def parse(self, text: str) -> dict[str, str]:
        """Parse the output of an LLM call."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers import RegexParser
</syntaxhighlight>

== I/O Contract ==

=== Configuration ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| regex || str || Yes || Regex pattern with capture groups
|-
| output_keys || list[str] || Yes || Key names for each capture group
|-
| default_output_key || str | None || No || Key to use when pattern doesn't match
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
| dict[str, str] || Dictionary mapping output keys to captured values
|}

=== Exceptions ===
{| class="wikitable"
|-
! Exception !! Condition
|-
| ValueError || Pattern doesn't match and no default_output_key set
|}

== Usage Examples ==

=== Basic Pattern Extraction ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers import RegexParser

# Extract name and age from structured text
parser = RegexParser(
    regex=r"Name: (\w+)\nAge: (\d+)",
    output_keys=["name", "age"],
)

text = "Name: Alice\nAge: 30"
result = parser.parse(text)
print(result)  # {'name': 'Alice', 'age': '30'}
</syntaxhighlight>

=== With Default Fallback ===
<syntaxhighlight lang="python">
parser = RegexParser(
    regex=r"Answer: (.+)",
    output_keys=["answer"],
    default_output_key="answer",
)

# Pattern matches
result = parser.parse("Answer: Paris is the capital of France.")
print(result)  # {'answer': 'Paris is the capital of France.'}

# Pattern doesn't match - uses entire text as default
result = parser.parse("I think Paris is the capital.")
print(result)  # {'answer': "I think Paris is the capital."}
</syntaxhighlight>

=== Multi-Field Extraction ===
<syntaxhighlight lang="python">
# Extract structured review data
parser = RegexParser(
    regex=r"Product: (.+)\nRating: (\d)/5\nComment: (.+)",
    output_keys=["product", "rating", "comment"],
)

text = """Product: Wireless Headphones
Rating: 4/5
Comment: Great sound quality, comfortable fit."""

result = parser.parse(text)
print(result)
# {'product': 'Wireless Headphones', 'rating': '4', 'comment': 'Great sound quality, comfortable fit.'}
</syntaxhighlight>

=== With LLM Chain ===
<syntaxhighlight lang="python">
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

parser = RegexParser(
    regex=r"Summary: (.+)\nSentiment: (\w+)",
    output_keys=["summary", "sentiment"],
)

prompt = PromptTemplate.from_template(
    "Analyze this review and respond in the format:\n"
    "Summary: <one sentence summary>\n"
    "Sentiment: <positive/negative/neutral>\n\n"
    "Review: {review}"
)

llm = ChatOpenAI(model="gpt-4")
chain = prompt | llm | parser

result = chain.invoke({"review": "This product exceeded my expectations!"})
print(result)  # {'summary': '...', 'sentiment': 'positive'}
</syntaxhighlight>

== Related Pages ==
* [[related_to::Implementation:langchain-ai_langchain_RegexDictParser]]
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]

