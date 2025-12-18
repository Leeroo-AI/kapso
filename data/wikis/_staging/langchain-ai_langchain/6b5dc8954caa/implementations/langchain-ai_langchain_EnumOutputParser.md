{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Output Parsing]], [[domain::Classification]], [[domain::Constrained Output]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Output parser `EnumOutputParser` that validates LLM output against a predefined set of enum values.

=== Description ===

The `EnumOutputParser` class parses LLM output and validates that it matches one of the values in a given Python Enum. This ensures type-safe classification where the LLM must choose from a predefined set of options. All enum values must be strings.

=== Usage ===

Use this parser for classification tasks with a fixed set of categories, sentiment analysis with predefined labels, routing decisions, or any scenario requiring constrained choice validation.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/output_parsers/enum.py libs/langchain/langchain_classic/output_parsers/enum.py]
* '''Lines:''' 1-46

=== Signature ===
<syntaxhighlight lang="python">
class EnumOutputParser(BaseOutputParser[Enum]):
    """Parse an output that is one of a set of values."""

    enum: type[Enum]
    """The enum to parse. Its values must be strings."""

    def parse(self, response: str) -> Enum:
        """Parse response to enum value."""

    def get_format_instructions(self) -> str:
        """Return format instructions listing valid options."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers import EnumOutputParser
</syntaxhighlight>

== I/O Contract ==

=== Configuration ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| enum || type[Enum] || Yes || Enum class with string values
|}

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| response || str || Yes || LLM output text to validate
|}

=== Outputs ===
{| class="wikitable"
|-
! Type !! Description
|-
| Enum || The matching enum member
|}

=== Exceptions ===
{| class="wikitable"
|-
! Exception !! Condition
|-
| OutputParserException || Response doesn't match any enum value
|-
| ValueError || Enum values are not all strings
|}

== Usage Examples ==

=== Basic Sentiment Classification ===
<syntaxhighlight lang="python">
from enum import Enum
from langchain_classic.output_parsers import EnumOutputParser

class Sentiment(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

parser = EnumOutputParser(enum=Sentiment)

# Parse classification result
result = parser.parse("positive")
print(result)  # Sentiment.POSITIVE
print(result.value)  # "positive"

# Get format instructions
print(parser.get_format_instructions())
# "Select one of the following options: positive, negative, neutral"
</syntaxhighlight>

=== Intent Classification ===
<syntaxhighlight lang="python">
class Intent(Enum):
    QUESTION = "question"
    COMMAND = "command"
    GREETING = "greeting"
    COMPLAINT = "complaint"

parser = EnumOutputParser(enum=Intent)

result = parser.parse("greeting")
print(result)  # Intent.GREETING
</syntaxhighlight>

=== With LLM Chain ===
<syntaxhighlight lang="python">
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

class Priority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

parser = EnumOutputParser(enum=Priority)

prompt = PromptTemplate.from_template(
    "Classify the priority of this task:\n\n"
    "Task: {task}\n\n"
    "{format_instructions}"
)

llm = ChatOpenAI(model="gpt-4")
chain = prompt | llm | parser

result = chain.invoke({
    "task": "Server is down, customers cannot access the app",
    "format_instructions": parser.get_format_instructions()
})
print(result)  # Priority.HIGH
</syntaxhighlight>

=== Routing Decisions ===
<syntaxhighlight lang="python">
class Department(Enum):
    SALES = "sales"
    SUPPORT = "support"
    BILLING = "billing"
    TECHNICAL = "technical"

parser = EnumOutputParser(enum=Department)

# Route customer inquiry to correct department
prompt = PromptTemplate.from_template(
    "Route this customer inquiry to the correct department:\n\n"
    "Inquiry: {inquiry}\n\n"
    "{format_instructions}"
)

chain = prompt | llm | parser
department = chain.invoke({
    "inquiry": "I can't log into my account",
    "format_instructions": parser.get_format_instructions()
})
# Route to department.value handler
</syntaxhighlight>

== Related Pages ==
* [[related_to::Implementation:langchain-ai_langchain_BooleanOutputParser]]
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]

