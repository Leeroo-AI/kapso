{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Output_Parsing]], [[domain::Structured_Output]], [[domain::LLM_Orchestration]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Output parser `StructuredOutputParser` that parses LLM responses into dictionaries based on defined response schemas.

=== Description ===

The `StructuredOutputParser` allows you to define expected output fields with names, descriptions, and types using `ResponseSchema` objects. It generates format instructions for the LLM and parses JSON markdown responses into dictionaries. This provides a simpler alternative to Pydantic-based parsing for basic structured outputs.

=== Usage ===

Use this parser when you need simple structured output without full Pydantic validation. For more complex validation, consider using `PydanticOutputParser` or the newer `with_structured_output` method on chat models.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/output_parsers/structured.py libs/langchain/langchain_classic/output_parsers/structured.py]
* '''Lines:''' 1-116

=== Signature ===
<syntaxhighlight lang="python">
class ResponseSchema(BaseModel):
    """Schema for a response field.

    Attributes:
        name: Field name in the output.
        description: Description of what this field contains.
        type: Data type (default: "string").
    """

    name: str
    description: str
    type: str = "string"


class StructuredOutputParser(BaseOutputParser[dict[str, Any]]):
    """Parse LLM output to a structured dictionary.

    Attributes:
        response_schemas: List of schemas defining expected output fields.
    """

    response_schemas: list[ResponseSchema]

    @classmethod
    def from_response_schemas(
        cls,
        response_schemas: list[ResponseSchema],
    ) -> StructuredOutputParser:
        """Create parser from a list of ResponseSchema."""

    def get_format_instructions(self, only_json: bool = False) -> str:
        """Get format instructions to include in prompts."""

    def parse(self, text: str) -> dict[str, Any]:
        """Parse LLM output to dictionary."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| text || str || Yes || LLM output containing JSON in markdown
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return || dict[str, Any] || Parsed dictionary with field names as keys
|}

== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

# Define expected output fields
response_schemas = [
    ResponseSchema(name="answer", description="The answer to the question"),
    ResponseSchema(name="confidence", description="Confidence level 1-10", type="int"),
    ResponseSchema(name="sources", description="List of sources used", type="List[string]"),
]

# Create parser
parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Get format instructions for the prompt
format_instructions = parser.get_format_instructions()
print(format_instructions)
# The output should be a Markdown code snippet formatted in the following schema...
# ```json
# {
#     "answer": string  // The answer to the question
#     "confidence": int  // Confidence level 1-10
#     "sources": List[string]  // List of sources used
# }
# ```
</syntaxhighlight>

=== With LLM Chain ===
<syntaxhighlight lang="python">
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

prompt = PromptTemplate(
    template="Answer the question.\n{format_instructions}\n\nQuestion: {question}",
    input_variables=["question"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

llm = ChatOpenAI()
chain = prompt | llm | parser

result = chain.invoke({"question": "What is the capital of France?"})
print(result)
# {'answer': 'Paris', 'confidence': 10, 'sources': ['Wikipedia']}
</syntaxhighlight>

=== Parsing Raw Output ===
<syntaxhighlight lang="python">
# Parse LLM output directly
llm_output = """```json
{
    "answer": "Paris",
    "confidence": 10,
    "sources": ["Wikipedia", "Britannica"]
}
```"""

result = parser.parse(llm_output)
print(result["answer"])  # "Paris"
print(result["sources"])  # ["Wikipedia", "Britannica"]
</syntaxhighlight>

=== Modern Alternative ===
<syntaxhighlight lang="python">
# For chat models, use with_structured_output instead
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class Answer(BaseModel):
    answer: str
    confidence: int
    sources: list[str]

llm = ChatOpenAI()
structured_llm = llm.with_structured_output(Answer)

result = structured_llm.invoke("What is the capital of France?")
# Answer(answer='Paris', confidence=10, sources=['...'])
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]

