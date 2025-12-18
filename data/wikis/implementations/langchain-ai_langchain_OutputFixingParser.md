{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Output_Parsing]], [[domain::Error_Handling]], [[domain::LLM_Orchestration]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Output parser wrapper `OutputFixingParser` that uses an LLM to fix parsing errors without needing the original prompt.

=== Description ===

The `OutputFixingParser` wraps another parser and, when parsing fails, invokes an LLM with the format instructions, failed completion, and error message to generate a corrected output. Unlike `RetryOutputParser`, it doesn't require the original prompt - it can fix outputs standalone using only the format instructions.

=== Usage ===

Use this parser when you want automatic output correction without access to the original prompt. It's simpler than `RetryOutputParser` but may be less accurate since it lacks the full context.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/output_parsers/fix.py libs/langchain/langchain_classic/output_parsers/fix.py]
* '''Lines:''' 1-156

=== Signature ===
<syntaxhighlight lang="python">
class OutputFixingParser(BaseOutputParser[T]):
    """Wrap a parser and try to fix parsing errors.

    Attributes:
        parser: The wrapped parser to use for parsing.
        retry_chain: Runnable to generate corrected completions.
        max_retries: Maximum retry attempts (default: 1).
        legacy: Whether to use legacy run/arun methods.
    """

    parser: Any
    retry_chain: RunnableSerializable[OutputFixingParserRetryChainInput, str] | Any
    max_retries: int = 1
    legacy: bool = True

    @classmethod
    def from_llm(
        cls,
        llm: Runnable,
        parser: BaseOutputParser[T],
        prompt: BasePromptTemplate = NAIVE_FIX_PROMPT,
        max_retries: int = 1,
    ) -> OutputFixingParser[T]:
        """Create an OutputFixingParser from an LLM and parser."""

    def parse(self, completion: str) -> T:
        """Parse with automatic retry on failure."""

    async def aparse(self, completion: str) -> T:
        """Async parse with automatic retry on failure."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers import OutputFixingParser
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| completion || str || Yes || LLM output to parse
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return || T || Parsed output matching the wrapped parser's output type
|}

== Usage Examples ==

=== Basic OutputFixingParser ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

# Create base parser
base_parser = PydanticOutputParser(pydantic_object=Person)

# Wrap with fixing logic
fixing_parser = OutputFixingParser.from_llm(
    llm=ChatOpenAI(),
    parser=base_parser,
    max_retries=2,
)

# Parse - will auto-fix if invalid
result = fixing_parser.parse('{"name": "John", "age": "twenty-five"}')
# The LLM will be asked to fix the invalid age
print(result)  # Person(name='John', age=25)
</syntaxhighlight>

=== Difference from RetryOutputParser ===
<syntaxhighlight lang="python">
# OutputFixingParser: doesn't need original prompt
result = fixing_parser.parse(completion)  # Only needs the completion

# RetryOutputParser: needs original prompt for context
result = retry_parser.parse_with_prompt(completion, prompt_value)

# Use OutputFixingParser when you don't have the prompt
# Use RetryOutputParser for better accuracy when you do
</syntaxhighlight>

=== In an LCEL Chain ===
<syntaxhighlight lang="python">
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "Extract person info from: {text}\n{format_instructions}"
)
llm = ChatOpenAI()

# Chain with fixing parser at the end
chain = prompt | llm | fixing_parser

result = chain.invoke({
    "text": "John is 25 years old",
    "format_instructions": base_parser.get_format_instructions(),
})
</syntaxhighlight>

=== Custom Fix Prompt ===
<syntaxhighlight lang="python">
from langchain_core.prompts import PromptTemplate

custom_fix_prompt = PromptTemplate.from_template("""
The following output failed to parse:
{completion}

Error: {error}

Expected format:
{instructions}

Please output ONLY valid JSON that matches the expected format:
""")

fixing_parser = OutputFixingParser.from_llm(
    llm=ChatOpenAI(),
    parser=base_parser,
    prompt=custom_fix_prompt,
    max_retries=3,
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
* [[related_to::Implementation:langchain-ai_langchain_RetryOutputParser]]

