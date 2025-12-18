{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Output Parsing]], [[domain::YAML]], [[domain::Pydantic Validation]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Output parser `YamlOutputParser` that parses YAML-formatted LLM output into Pydantic models.

=== Description ===

The `YamlOutputParser` class extracts YAML content from LLM responses (optionally within markdown code blocks) and validates it against a Pydantic model schema. This enables type-safe structured output from LLMs using YAML's human-readable format, which can be easier for models to generate than JSON.

=== Usage ===

Use this parser when you want structured output from LLMs in YAML format, particularly for complex nested structures or when the YAML format is more natural for the domain (like configuration files).

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/output_parsers/yaml.py libs/langchain/langchain_classic/output_parsers/yaml.py]
* '''Lines:''' 1-70

=== Signature ===
<syntaxhighlight lang="python">
class YamlOutputParser(BaseOutputParser[T]):
    """Parse YAML output using a Pydantic model."""

    pydantic_object: type[T]
    """The Pydantic model to parse."""
    pattern: re.Pattern = re.compile(r"^```(?:ya?ml)?(?P<yaml>[^`]*)", ...)
    """Regex to extract YAML from code blocks."""

    def parse(self, text: str) -> T:
        """Parse YAML text into Pydantic model."""

    def get_format_instructions(self) -> str:
        """Return YAML format instructions with schema."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers import YamlOutputParser
</syntaxhighlight>

== I/O Contract ==

=== Configuration ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| pydantic_object || type[BaseModel] || Yes || Pydantic model class for validation
|}

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| text || str || Yes || LLM output containing YAML (optionally in code block)
|}

=== Outputs ===
{| class="wikitable"
|-
! Type !! Description
|-
| T (Pydantic model) || Validated instance of the Pydantic model
|}

=== Exceptions ===
{| class="wikitable"
|-
! Exception !! Condition
|-
| OutputParserException || YAML parsing fails or validation fails
|}

== Usage Examples ==

=== Basic YAML Parsing ===
<syntaxhighlight lang="python">
from pydantic import BaseModel
from langchain_classic.output_parsers import YamlOutputParser

class Person(BaseModel):
    name: str
    age: int
    occupation: str

parser = YamlOutputParser(pydantic_object=Person)

# Parse YAML from code block
yaml_text = """```yaml
name: Alice
age: 30
occupation: Engineer
```"""

result = parser.parse(yaml_text)
print(result)  # Person(name='Alice', age=30, occupation='Engineer')
print(result.name)  # 'Alice'
</syntaxhighlight>

=== Nested Structures ===
<syntaxhighlight lang="python">
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str
    country: str

class Employee(BaseModel):
    name: str
    department: str
    address: Address

parser = YamlOutputParser(pydantic_object=Employee)

yaml_text = """
name: Bob Smith
department: Engineering
address:
  street: 123 Main St
  city: San Francisco
  country: USA
"""

result = parser.parse(yaml_text)
print(result.address.city)  # 'San Francisco'
</syntaxhighlight>

=== With LLM Chain ===
<syntaxhighlight lang="python">
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

class Recipe(BaseModel):
    name: str
    ingredients: list[str]
    prep_time_minutes: int
    instructions: list[str]

parser = YamlOutputParser(pydantic_object=Recipe)

prompt = PromptTemplate.from_template(
    "Create a simple recipe for {dish}.\n\n"
    "{format_instructions}"
)

llm = ChatOpenAI(model="gpt-4")
chain = prompt | llm | parser

recipe = chain.invoke({
    "dish": "pasta carbonara",
    "format_instructions": parser.get_format_instructions()
})

print(recipe.name)
print(f"Prep time: {recipe.prep_time_minutes} minutes")
for ingredient in recipe.ingredients:
    print(f"  - {ingredient}")
</syntaxhighlight>

=== Format Instructions ===
<syntaxhighlight lang="python">
# Get detailed format instructions
instructions = parser.get_format_instructions()
print(instructions)
# Output includes:
# - JSON schema description
# - YAML examples with arrays and objects
# - Standard YAML formatting conventions
</syntaxhighlight>

== Related Pages ==
* [[related_to::Implementation:langchain-ai_langchain_format_instructions]]
* [[related_to::Implementation:langchain-ai_langchain_StructuredOutputParser]]
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]

