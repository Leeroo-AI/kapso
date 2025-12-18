{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Output Parsing]], [[domain::Temporal Data]], [[domain::Date Extraction]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Output parser `DatetimeOutputParser` that extracts datetime values from LLM text responses.

=== Description ===

The `DatetimeOutputParser` class parses LLM output to extract a Python `datetime` object. It uses a configurable format string (default ISO 8601 with microseconds) and provides format instructions with example values to guide the LLM's output format.

=== Usage ===

Use this parser when you need to extract temporal information from LLM responses, such as scheduling tasks, parsing event dates, or extracting timestamps from text.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/output_parsers/datetime.py libs/langchain/langchain_classic/output_parsers/datetime.py]
* '''Lines:''' 1-59

=== Signature ===
<syntaxhighlight lang="python">
class DatetimeOutputParser(BaseOutputParser[datetime]):
    """Parse the output of an LLM call to a datetime."""

    format: str = "%Y-%m-%dT%H:%M:%S.%fZ"
    """The datetime format string."""

    def get_format_instructions(self) -> str:
        """Returns the format instructions for the given format."""

    def parse(self, response: str) -> datetime:
        """Parse a string into a datetime object."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers import DatetimeOutputParser
</syntaxhighlight>

== I/O Contract ==

=== Configuration ===
{| class="wikitable"
|-
! Name !! Type !! Default !! Description
|-
| format || str || "%Y-%m-%dT%H:%M:%S.%fZ" || Python strftime format string
|}

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| response || str || Yes || LLM output text containing datetime string
|}

=== Outputs ===
{| class="wikitable"
|-
! Type !! Description
|-
| datetime || Parsed Python datetime object
|}

=== Exceptions ===
{| class="wikitable"
|-
! Exception !! Condition
|-
| OutputParserException || Response doesn't match expected format
|}

== Usage Examples ==

=== Basic Datetime Parsing ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers import DatetimeOutputParser

parser = DatetimeOutputParser()

# Parse ISO format datetime
result = parser.parse("2024-07-15T14:30:00.000000Z")
print(result)  # datetime(2024, 7, 15, 14, 30, 0)

# Get format instructions for prompts
print(parser.get_format_instructions())
# "Write a datetime string that matches the pattern: '%Y-%m-%dT%H:%M:%S.%fZ'.
#  Examples: 2023-07-04T14:30:00.000000Z, 1999-12-31T23:59:59.999999Z, ...
#  Return ONLY this string, no other words!"
</syntaxhighlight>

=== Custom Date Format ===
<syntaxhighlight lang="python">
# Use simple date format
parser = DatetimeOutputParser(format="%Y-%m-%d")

result = parser.parse("2024-07-15")
print(result)  # datetime(2024, 7, 15, 0, 0, 0)
</syntaxhighlight>

=== With LLM Chain ===
<syntaxhighlight lang="python">
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

parser = DatetimeOutputParser()

prompt = PromptTemplate.from_template(
    "When did the following event happen?\n\n"
    "Event: {event}\n\n"
    "{format_instructions}"
)

llm = ChatOpenAI(model="gpt-4")
chain = prompt | llm | parser

result = chain.invoke({
    "event": "The first moon landing",
    "format_instructions": parser.get_format_instructions()
})
print(result)  # datetime(1969, 7, 20, ...)
</syntaxhighlight>

=== Scheduling Application ===
<syntaxhighlight lang="python">
from datetime import datetime, timezone

parser = DatetimeOutputParser(format="%Y-%m-%d %H:%M")

prompt = PromptTemplate.from_template(
    "Schedule a meeting for next Tuesday at 2pm. "
    "Today is {today}.\n\n{format_instructions}"
)

chain = prompt | llm | parser
meeting_time = chain.invoke({
    "today": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
    "format_instructions": parser.get_format_instructions()
})
</syntaxhighlight>

== Related Pages ==
* [[related_to::Implementation:langchain-ai_langchain_RegexParser]]
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]

