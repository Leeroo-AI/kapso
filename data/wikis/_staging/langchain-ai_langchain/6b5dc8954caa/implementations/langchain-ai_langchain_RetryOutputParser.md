{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Docs|https://docs.langchain.com]]
|-
! Domains
| [[domain::Output_Parsing]], [[domain::Error_Handling]], [[domain::LLM_Orchestration]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Output parser wrappers `RetryOutputParser` and `RetryWithErrorOutputParser` that use an LLM to fix parsing errors by re-prompting with the original context.

=== Description ===

The retry output parsers wrap another parser and attempt to fix parsing failures by invoking an LLM with the original prompt and failed completion. `RetryOutputParser` tells the LLM the output "did not satisfy constraints" while `RetryWithErrorOutputParser` additionally provides the specific error message, giving the LLM more context to correct its output. Both support configurable retry counts.

=== Usage ===

Use these parsers when working with structured outputs where LLM responses may occasionally fail validation. They're particularly useful for JSON/Pydantic parsers where the LLM might produce malformed output that can be corrected with guidance.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/output_parsers/retry.py libs/langchain/langchain_classic/output_parsers/retry.py]
* '''Lines:''' 1-315

=== Signature ===
<syntaxhighlight lang="python">
class RetryOutputParser(BaseOutputParser[T]):
    """Wrap a parser and try to fix parsing errors.

    Attributes:
        parser: The wrapped parser to use for parsing.
        retry_chain: Runnable to generate corrected completions.
        max_retries: Maximum retry attempts.
        legacy: Whether to use legacy run/arun methods.
    """

    parser: BaseOutputParser[T]
    retry_chain: RunnableSerializable[RetryOutputParserRetryChainInput, str] | Any
    max_retries: int = 1
    legacy: bool = True

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        parser: BaseOutputParser[T],
        prompt: BasePromptTemplate = NAIVE_RETRY_PROMPT,
        max_retries: int = 1,
    ) -> RetryOutputParser[T]:
        """Create RetryOutputParser from an LLM and parser."""

    def parse_with_prompt(self, completion: str, prompt_value: PromptValue) -> T:
        """Parse with retry logic using the original prompt."""


class RetryWithErrorOutputParser(BaseOutputParser[T]):
    """Retry parser that includes the error message in retry prompts.

    Provides more context to the LLM by including what went wrong.
    """

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        parser: BaseOutputParser[T],
        prompt: BasePromptTemplate = NAIVE_RETRY_WITH_ERROR_PROMPT,
        max_retries: int = 1,
    ) -> RetryWithErrorOutputParser[T]:
        """Create RetryWithErrorOutputParser from an LLM and parser."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers.retry import (
    RetryOutputParser,
    RetryWithErrorOutputParser,
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| completion || str || Yes || The LLM output to parse
|-
| prompt_value || PromptValue || Yes || Original prompt for context in retries
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return || T || Parsed output matching the wrapped parser's output type
|}

== Usage Examples ==

=== Basic RetryOutputParser ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers.retry import RetryOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

# Create base parser
base_parser = PydanticOutputParser(pydantic_object=Person)

# Wrap with retry logic
retry_parser = RetryOutputParser.from_llm(
    llm=ChatOpenAI(),
    parser=base_parser,
    max_retries=2,
)

# Parse with retry - needs original prompt for context
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate.from_template(
    "Extract person info: {text}\n{format_instructions}"
)
prompt_value = prompt.format_prompt(
    text="John is 25 years old",
    format_instructions=base_parser.get_format_instructions(),
)

# If parsing fails, retry_parser will ask LLM to fix it
result = retry_parser.parse_with_prompt(
    completion='{"name": "John", "age": "twenty-five"}',  # Invalid
    prompt_value=prompt_value,
)
</syntaxhighlight>

=== RetryWithErrorOutputParser for Better Debugging ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers.retry import RetryWithErrorOutputParser

# This variant includes the error in the retry prompt
retry_parser = RetryWithErrorOutputParser.from_llm(
    llm=ChatOpenAI(),
    parser=base_parser,
    max_retries=3,  # Try up to 3 times
)

# The retry prompt will include:
# - Original prompt
# - Failed completion
# - Error message (e.g., "age must be an integer")
result = retry_parser.parse_with_prompt(completion, prompt_value)
</syntaxhighlight>

=== Custom Retry Prompt ===
<syntaxhighlight lang="python">
from langchain_core.prompts import PromptTemplate

custom_retry_prompt = PromptTemplate.from_template("""
The following output was invalid:
{completion}

Original request:
{prompt}

Error: {error}

Please provide a corrected JSON response:
""")

retry_parser = RetryWithErrorOutputParser.from_llm(
    llm=ChatOpenAI(),
    parser=base_parser,
    prompt=custom_retry_prompt,
    max_retries=2,
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
* [[related_to::Implementation:langchain-ai_langchain_OutputFixingParser]]

