{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai/langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Output Parsing]], [[domain::Error Recovery]], [[domain::LLM Chains]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==
Output parser wrapper that automatically retries parsing failures by sending the original prompt and failed completion back to an LLM for correction.

=== Description ===
The RetryOutputParser is a wrapper around any BaseOutputParser that implements automatic retry logic when parsing fails. When an OutputParserException occurs, it constructs a prompt containing the original prompt, the failed completion, and a request to try again. This new prompt is sent to a language model to generate a corrected response, which is then parsed again. This process can repeat up to a configurable maximum number of retries.

The parser supports both legacy LLMChain-style invocation (using `run`/`arun` methods) and modern RunnableSerializable invocation (using `invoke`/`ainvoke` methods). It requires both the completion and the original prompt to function, so it cannot be used with the standard `parse()` method - only `parse_with_prompt()` is supported.

=== Usage ===
Use RetryOutputParser when you have a structured output parser that may fail on malformed LLM responses, and you want to automatically attempt to fix parsing errors by asking the LLM to try again. This is particularly useful for structured output tasks where the model occasionally produces invalid JSON, missing fields, or incorrectly formatted responses. For cases where you want to provide the specific error message back to the LLM (which may help it fix the issue more effectively), use RetryWithErrorOutputParser instead.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai/langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/output_parsers/retry.py libs/langchain/langchain_classic/output_parsers/retry.py]

=== Signature ===
<syntaxhighlight lang="python">
class RetryOutputParser(BaseOutputParser[T]):
    parser: Annotated[BaseOutputParser[T], SkipValidation()]
    retry_chain: Annotated[
        RunnableSerializable[RetryOutputParserRetryChainInput, str] | Any,
        SkipValidation(),
    ]
    max_retries: int = 1
    legacy: bool = True
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers.retry import RetryOutputParser
</syntaxhighlight>

== I/O Contract ==

=== Input Parameters ===
{| class="wikitable"
|-
! Parameter
! Type
! Default
! Description
|-
| parser
| BaseOutputParser[T]
| Required
| The underlying parser to use for parsing the output
|-
| retry_chain
| RunnableSerializable or LLMChain
| Required
| The chain to use for generating retry attempts
|-
| max_retries
| int
| 1
| Maximum number of times to retry parsing
|-
| legacy
| bool
| True
| Whether to use legacy run/arun methods or modern invoke/ainvoke
|}

=== Constructor Method ===
{| class="wikitable"
|-
! Method
! Returns
! Description
|-
| from_llm(llm, parser, prompt=NAIVE_RETRY_PROMPT, max_retries=1)
| RetryOutputParser[T]
| Create a RetryOutputParser from a language model and parser
|}

=== Core Methods ===
{| class="wikitable"
|-
! Method
! Returns
! Description
|-
| parse_with_prompt(completion: str, prompt_value: PromptValue)
| T
| Parse the output with retry logic, using the original prompt for context
|-
| aparse_with_prompt(completion: str, prompt_value: PromptValue)
| T
| Async version of parse_with_prompt
|-
| get_format_instructions()
| str
| Delegates to the underlying parser's format instructions
|}

=== Output ===
Returns the parsed output of type T as defined by the underlying parser, or raises OutputParserException if all retry attempts are exhausted.

== Usage Examples ==

=== Basic Usage with Pydantic Parser ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers.retry import RetryOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class User(BaseModel):
    name: str = Field(description="User's full name")
    age: int = Field(description="User's age in years")


# Create the underlying parser
base_parser = PydanticOutputParser(pydantic_object=User)

# Create retry parser with LLM
llm = ChatOpenAI(temperature=0)
retry_parser = RetryOutputParser.from_llm(
    llm=llm,
    parser=base_parser,
    max_retries=2
)

# Use with prompt
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="Extract user info:\n{format_instructions}\n\n{query}",
    input_variables=["query"],
    partial_variables={"format_instructions": base_parser.get_format_instructions()}
)

# When parsing fails, it will automatically retry
prompt_value = prompt.format_prompt(query="John is 30 years old")
completion = '{"name": "John", "age": "thirty"}'  # Invalid: age should be int

try:
    result = retry_parser.parse_with_prompt(completion, prompt_value)
    print(result)
except Exception as e:
    print(f"Failed after retries: {e}")
</syntaxhighlight>

=== Using in a Chain ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers.retry import RetryOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class Response(BaseModel):
    answer: str
    confidence: float


# Set up parser chain
base_parser = PydanticOutputParser(pydantic_object=Response)
llm = ChatOpenAI()
retry_parser = RetryOutputParser.from_llm(llm, base_parser, max_retries=3)

# Create a chain that uses the retry parser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "Answer this question: {question}\n\n{format_instructions}"
)

chain = prompt | llm

# Execute and parse with retry
question = "What is the capital of France?"
prompt_value = prompt.format_prompt(
    question=question,
    format_instructions=base_parser.get_format_instructions()
)
response = chain.invoke({"question": question, "format_instructions": base_parser.get_format_instructions()})
parsed = retry_parser.parse_with_prompt(response.content, prompt_value)
print(parsed)
</syntaxhighlight>

=== Custom Retry Prompt ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers.retry import RetryOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

custom_retry_prompt = PromptTemplate.from_template("""
The following completion was incorrect:

Original Prompt:
{prompt}

Failed Completion:
{completion}

Please provide a corrected version that follows the format exactly.
""")

llm = ChatOpenAI()
retry_parser = RetryOutputParser.from_llm(
    llm=llm,
    parser=base_parser,
    prompt=custom_retry_prompt,
    max_retries=2
)
</syntaxhighlight>

== Related Pages ==
* [[langchain-ai_langchain_RetryWithErrorOutputParser]] - Enhanced version that includes error details in retry prompts
* [[langchain-ai_langchain_BaseOutputParser]] - Base class for all output parsers
* [[langchain-ai_langchain_PydanticOutputParser]] - Commonly used with retry parsers for structured output
* [[principle::Error Recovery in LLM Pipelines]]
* [[environment::Production LLM Systems]]
