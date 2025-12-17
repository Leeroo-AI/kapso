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
Enhanced output parser wrapper that retries parsing failures by providing the LLM with the original prompt, failed completion, and the specific error message that was raised.

=== Description ===
RetryWithErrorOutputParser extends the retry pattern by including detailed error information in the retry prompt. When an OutputParserException occurs, it constructs a prompt containing the original prompt, the failed completion, AND the error message (as a string representation of the exception). This additional context helps the LLM understand specifically what went wrong, potentially leading to more effective corrections.

Like RetryOutputParser, this wrapper supports both legacy LLMChain-style invocation and modern RunnableSerializable invocation patterns. It requires the original prompt context and therefore only works with `parse_with_prompt()` and `aparse_with_prompt()` methods.

The default retry prompt template includes placeholders for `{prompt}`, `{completion}`, and `{error}`, allowing the LLM to see exactly what constraint was violated.

=== Usage ===
Use RetryWithErrorOutputParser instead of the basic RetryOutputParser when parsing errors contain useful diagnostic information that can help the LLM fix the issue. This is especially valuable for validation errors with specific field requirements, schema violations, or type mismatches where the error message provides clear guidance on what needs to be corrected.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai/langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/output_parsers/retry.py libs/langchain/langchain_classic/output_parsers/retry.py]

=== Signature ===
<syntaxhighlight lang="python">
class RetryWithErrorOutputParser(BaseOutputParser[T]):
    parser: Annotated[BaseOutputParser[T], SkipValidation()]
    retry_chain: Annotated[
        RunnableSerializable[RetryWithErrorOutputParserRetryChainInput, str] | Any,
        SkipValidation(),
    ]
    max_retries: int = 1
    legacy: bool = True
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers.retry import RetryWithErrorOutputParser
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
| The chain to use for generating retry attempts with error context
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
| from_llm(llm, parser, prompt=NAIVE_RETRY_WITH_ERROR_PROMPT, max_retries=1)
| RetryWithErrorOutputParser[T]
| Create a RetryWithErrorOutputParser from a language model and parser
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
| Parse output with error-aware retry logic
|-
| aparse_with_prompt(completion: str, prompt_value: PromptValue)
| T
| Async version of parse_with_prompt with error details
|-
| get_format_instructions()
| str
| Delegates to the underlying parser's format instructions
|}

=== Output ===
Returns the parsed output of type T as defined by the underlying parser, or raises OutputParserException if all retry attempts fail.

== Usage Examples ==

=== Basic Usage with Error Context ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers.retry import RetryWithErrorOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, validator


class Product(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Price in USD", gt=0)
    quantity: int = Field(description="Quantity in stock", ge=0)

    @validator('price')
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Price must be greater than 0')
        return v


# Create parser with error feedback
base_parser = PydanticOutputParser(pydantic_object=Product)
llm = ChatOpenAI(temperature=0)
retry_parser = RetryWithErrorOutputParser.from_llm(
    llm=llm,
    parser=base_parser,
    max_retries=3
)

# The error message helps the LLM understand what to fix
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="Extract product info:\n{format_instructions}\n\n{query}",
    input_variables=["query"],
    partial_variables={"format_instructions": base_parser.get_format_instructions()}
)

prompt_value = prompt.format_prompt(query="Widget costs $25 with 100 units available")
bad_completion = '{"name": "Widget", "price": -25, "quantity": 100}'  # Negative price

# The retry will include the validation error, helping the LLM correct it
result = retry_parser.parse_with_prompt(bad_completion, prompt_value)
print(result)  # Should succeed after retry with corrected positive price
</syntaxhighlight>

=== Custom Error Prompt Template ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers.retry import RetryWithErrorOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

custom_error_prompt = PromptTemplate.from_template("""
You previously generated an invalid response.

Original Instructions:
{prompt}

Your Previous Response:
{completion}

Error That Occurred:
{error}

Please carefully read the error message and generate a corrected response that addresses the specific issue mentioned.
""")

llm = ChatOpenAI(model="gpt-4")
retry_parser = RetryWithErrorOutputParser.from_llm(
    llm=llm,
    parser=base_parser,
    prompt=custom_error_prompt,
    max_retries=2
)
</syntaxhighlight>

=== Comparing with Basic RetryOutputParser ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers.retry import (
    RetryOutputParser,
    RetryWithErrorOutputParser
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class Email(BaseModel):
    subject: str = Field(description="Email subject line")
    recipient: str = Field(description="Valid email address")

    @validator('recipient')
    def validate_email(cls, v):
        if '@' not in v or '.' not in v.split('@')[-1]:
            raise ValueError('Must be a valid email address with @ and domain')
        return v


base_parser = PydanticOutputParser(pydantic_object=Email)
llm = ChatOpenAI()

# Basic retry - only knows parsing failed
basic_retry = RetryOutputParser.from_llm(llm, base_parser)

# Error retry - knows WHY it failed (missing @ or domain)
error_retry = RetryWithErrorOutputParser.from_llm(llm, base_parser)

# The error_retry parser is more likely to succeed because it can see
# the specific validation message about email format requirements
</syntaxhighlight>

=== Async Usage ===
<syntaxhighlight lang="python">
import asyncio
from langchain_classic.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers.retry import RetryWithErrorOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class Answer(BaseModel):
    response: str
    confidence: float


async def parse_with_retry():
    base_parser = PydanticOutputParser(pydantic_object=Answer)
    llm = ChatOpenAI()
    retry_parser = RetryWithErrorOutputParser.from_llm(
        llm=llm,
        parser=base_parser,
        max_retries=2
    )

    from langchain_core.prompts import PromptTemplate
    prompt = PromptTemplate(
        template="Answer: {query}\n{format_instructions}",
        input_variables=["query"],
        partial_variables={"format_instructions": base_parser.get_format_instructions()}
    )

    prompt_value = prompt.format_prompt(query="What is 2+2?")
    completion = '{"response": "4", "confidence": "high"}'  # String instead of float

    # Will retry with error message about type mismatch
    result = await retry_parser.aparse_with_prompt(completion, prompt_value)
    return result


result = asyncio.run(parse_with_retry())
print(result)
</syntaxhighlight>

=== Handling Multiple Validation Errors ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers.retry import RetryWithErrorOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, validator


class Address(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    state: str = Field(description="Two-letter state code")
    zip_code: str = Field(description="5-digit ZIP code")

    @validator('state')
    def validate_state(cls, v):
        if len(v) != 2 or not v.isupper():
            raise ValueError('State must be 2 uppercase letters')
        return v

    @validator('zip_code')
    def validate_zip(cls, v):
        if not v.isdigit() or len(v) != 5:
            raise ValueError('ZIP code must be exactly 5 digits')
        return v


base_parser = PydanticOutputParser(pydantic_object=Address)
llm = ChatOpenAI()
retry_parser = RetryWithErrorOutputParser.from_llm(
    llm=llm,
    parser=base_parser,
    max_retries=3
)

# With multiple validation rules, error messages are crucial
# The LLM sees exactly which field failed and what constraint was violated
</syntaxhighlight>

== Related Pages ==
* [[langchain-ai_langchain_RetryOutputParser]] - Basic retry parser without error context
* [[langchain-ai_langchain_BaseOutputParser]] - Base class for all output parsers
* [[langchain-ai_langchain_PydanticOutputParser]] - Structured output parser with validation
* [[principle::Error Recovery in LLM Pipelines]]
* [[principle::Providing Context in Retry Logic]]
* [[environment::Production LLM Systems]]
