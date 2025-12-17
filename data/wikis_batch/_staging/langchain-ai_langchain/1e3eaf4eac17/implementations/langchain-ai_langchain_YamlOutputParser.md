# YamlOutputParser

## Metadata
- **Package**: `langchain-classic`
- **Module**: `langchain_classic.output_parsers.yaml`
- **Source File**: `/tmp/praxium_repo_wjjl6pl8/libs/langchain/langchain_classic/output_parsers/yaml.py`
- **Type**: Output Parser
- **Base Class**: `BaseOutputParser[T]` (Generic with Pydantic model)
- **Dependencies**: `pyyaml`, `pydantic`

## Overview

`YamlOutputParser` is a type-safe parser that validates LLM YAML outputs against Pydantic models. It extracts YAML from markdown code blocks, parses it, and validates the structure using Pydantic's model validation. This provides both format flexibility (YAML is more human-readable than JSON) and strong type safety through Pydantic.

### Key Characteristics

- **Output Type**: `T` (Generic Pydantic model type)
- **Pydantic Validation**: Full type checking and validation
- **YAML Format**: Accepts YAML in markdown code blocks or plain text
- **Flexible Input**: Handles ``` ```yaml```, ``` ```yml```, or bare YAML
- **JSON Schema**: Generates format instructions from Pydantic JSON schema
- **Type Safety**: Returns validated Pydantic model instances

## Code Reference

### Class Definition

```python
class YamlOutputParser(BaseOutputParser[T]):
    """Parse YAML output using a Pydantic model."""

    pydantic_object: type[T]
    """The Pydantic model to parse."""
    pattern: re.Pattern = re.compile(
        r"^```(?:ya?ml)?(?P<yaml>[^`]*)",
        re.MULTILINE | re.DOTALL,
    )
    """Regex pattern to match yaml code blocks
    within triple backticks with optional yaml or yml prefix."""

    @override
    def parse(self, text: str) -> T:
        try:
            # Greedy search for 1st yaml candidate.
            match = re.search(self.pattern, text.strip())
            # If no backticks were present, try to parse the entire output as yaml.
            yaml_str = match.group("yaml") if match else text

            json_object = yaml.safe_load(yaml_str)
            return self.pydantic_object.model_validate(json_object)

        except (yaml.YAMLError, ValidationError) as e:
            name = self.pydantic_object.__name__
            msg = f"Failed to parse {name} from completion {text}. Got: {e}"
            raise OutputParserException(msg, llm_output=text) from e

    @override
    def get_format_instructions(self) -> str:
        # Copy schema to avoid altering original Pydantic schema.
        schema = dict(self.pydantic_object.model_json_schema().items())

        # Remove extraneous fields.
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        # Ensure yaml in context is well-formed with double quotes.
        schema_str = json.dumps(reduced_schema)

        return YAML_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    @property
    def _type(self) -> str:
        return "yaml"

    @property
    @override
    def OutputType(self) -> type[T]:
        return self.pydantic_object
```

## Input/Output Contract

### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pydantic_object` | `type[T]` | Required | Pydantic model class defining the expected structure |
| `pattern` | `re.Pattern` | Compiled regex | Pattern to extract YAML from markdown code blocks |

### Pattern Details

Default pattern: `r"^```(?:ya?ml)?(?P<yaml>[^`]*)"`
- Matches ``` ```yaml```, ``` ```yml```, or ``` ``` ``` at start
- Captures everything until closing ``` ``` ```
- Uses `MULTILINE` and `DOTALL` flags

### Methods

#### `parse(text: str) -> T`

Parses YAML and validates against Pydantic model.

**Arguments:**
- `text` (str): LLM output containing YAML (with or without code blocks)

**Returns:**
- `T`: Validated Pydantic model instance

**Raises:**
- `OutputParserException`: If YAML is invalid or validation fails

**Process:**
1. Search for YAML in markdown code blocks
2. If no code blocks, treat entire text as YAML
3. Parse YAML to Python dict using `yaml.safe_load()`
4. Validate dict against Pydantic model using `model_validate()`
5. Return validated model instance

#### `get_format_instructions() -> str`

Generates comprehensive YAML format instructions with examples.

**Returns:**
- `str`: Multi-line instructions including JSON schema and YAML examples

**Schema Processing:**
1. Extracts JSON schema from Pydantic model
2. Removes "title" and "type" fields (extraneous for instructions)
3. Converts schema to JSON string
4. Formats into YAML_FORMAT_INSTRUCTIONS template

#### `OutputType` Property

Returns the Pydantic model class type.

**Returns:**
- `type[T]`: The Pydantic model class

## Examples

### Basic Usage

```python
from langchain_classic.output_parsers.yaml import YamlOutputParser
from pydantic import BaseModel, Field

# Define Pydantic model
class Person(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years")
    city: str = Field(description="City of residence")

# Create parser
parser = YamlOutputParser(pydantic_object=Person)

# Parse YAML in code block
response = """```yaml
name: Alice Smith
age: 25
city: Seattle
```"""

result = parser.parse(response)
# Returns: Person(name="Alice Smith", age=25, city="Seattle")
print(result.name)  # "Alice Smith"
print(result.age)   # 25
```

### Without Code Blocks

```python
# Parser handles plain YAML too
response = """name: Bob Jones
age: 30
city: Portland"""

result = parser.parse(response)
# Returns: Person(name="Bob Jones", age=30, city="Portland")
```

### Complex Nested Structure

```python
from typing import List

class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class Contact(BaseModel):
    name: str
    age: int
    email: str
    addresses: List[Address]
    active: bool

parser = YamlOutputParser(pydantic_object=Contact)

response = """```yaml
name: Charlie Brown
age: 35
email: charlie@example.com
active: true
addresses:
  - street: 123 Main St
    city: Boston
    zip_code: "02101"
  - street: 456 Oak Ave
    city: Cambridge
    zip_code: "02139"
```"""

result = parser.parse(response)
# Returns: Contact instance with nested Address objects
print(result.addresses[0].city)  # "Boston"
```

### With Field Validation

```python
from pydantic import BaseModel, Field, field_validator

class Product(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Price in USD", gt=0)
    quantity: int = Field(description="Quantity in stock", ge=0)

    @field_validator('price')
    @classmethod
    def validate_price(cls, v):
        if v > 10000:
            raise ValueError('Price too high')
        return v

parser = YamlOutputParser(pydantic_object=Product)

# Valid product
response = """```yaml
name: Laptop
price: 999.99
quantity: 50
```"""

result = parser.parse(response)
# Returns: Product(name="Laptop", price=999.99, quantity=50)
```

### Format Instructions

```python
# Get format instructions for LLM
instructions = parser.get_format_instructions()
print(instructions)

# Output includes:
# - JSON schema description
# - Example YAML structures
# - Formatting guidelines
# - Requirement to enclose in triple backticks
```

### With LLM Chain

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: int = Field(description="Rating from 1-10", ge=1, le=10)
    summary: str = Field(description="Brief review summary")
    recommend: bool = Field(description="Would you recommend it?")

parser = YamlOutputParser(pydantic_object=MovieReview)
llm = ChatOpenAI()

prompt = PromptTemplate(
    template="""Write a movie review for: {movie}

{format_instructions}""",
    input_variables=["movie"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser

result = chain.invoke({"movie": "The Matrix"})
# Returns: MovieReview instance with all fields validated
print(f"{result.title}: {result.rating}/10")
```

### Error Handling

```python
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel

class User(BaseModel):
    username: str
    age: int

parser = YamlOutputParser(pydantic_object=User)

# Invalid YAML
try:
    response = """```yaml
username: alice
age: not a number
```"""
    parser.parse(response)
except OutputParserException as e:
    print(e)
    # "Failed to parse User from completion... validation error"

# Missing required field
try:
    response = """```yaml
username: bob
```"""
    parser.parse(response)
except OutputParserException as e:
    print(e)
    # "Failed to parse User... Field required"
```

### Optional Fields and Defaults

```python
from typing import Optional

class Config(BaseModel):
    host: str = Field(description="Server host")
    port: int = Field(default=8080, description="Server port")
    ssl: bool = Field(default=False, description="Enable SSL")
    timeout: Optional[int] = Field(default=None, description="Timeout in seconds")

parser = YamlOutputParser(pydantic_object=Config)

# Minimal YAML (uses defaults)
response = """```yaml
host: localhost
```"""

result = parser.parse(response)
# Returns: Config(host="localhost", port=8080, ssl=False, timeout=None)
```

### Enum Fields

```python
from enum import Enum

class Status(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"

class Account(BaseModel):
    username: str
    status: Status

parser = YamlOutputParser(pydantic_object=Account)

response = """```yaml
username: alice
status: active
```"""

result = parser.parse(response)
# Returns: Account(username="alice", status=Status.ACTIVE)
```

### Different Code Block Formats

```python
parser = YamlOutputParser(pydantic_object=Person)

# All these work:

# Format 1: ```yaml
response1 = """```yaml
name: Alice
age: 25
city: Seattle
```"""

# Format 2: ```yml
response2 = """```yml
name: Alice
age: 25
city: Seattle
```"""

# Format 3: ``` (no language)
response3 = """```
name: Alice
age: 25
city: Seattle
```"""

# Format 4: No code blocks
response4 = """name: Alice
age: 25
city: Seattle"""

# All parse successfully
result = parser.parse(response1)  # Works
result = parser.parse(response2)  # Works
result = parser.parse(response3)  # Works
result = parser.parse(response4)  # Works
```

### With OutputFixingParser

```python
from langchain_classic.output_parsers.fix import OutputFixingParser

base_parser = YamlOutputParser(pydantic_object=Person)

# Wrap for automatic error correction
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI()
)

# Handles malformed YAML
response = """```yaml
name: Alice
age: twenty-five  # Invalid type
city: Seattle
```"""

result = fixing_parser.parse(response)
# Fixing parser asks LLM to correct the type error
# Returns: Person(name="Alice", age=25, city="Seattle")
```

## Related Pages

- `BaseOutputParser` - Base class for all output parsers
- `PydanticOutputParser` - Similar parser for JSON format
- `StructuredOutputParser` - Schema-based JSON parser
- `OutputFixingParser` - Wrapper for automatic error correction
- Pydantic documentation - For model definition
- YAML specification - For format details

## Implementation Notes

### Design Decisions

1. **Pydantic Integration**: Leverages Pydantic for robust validation and type safety
2. **Flexible Input**: Accepts YAML with or without code blocks
3. **Safe Loading**: Uses `yaml.safe_load()` to prevent code injection
4. **JSON Schema**: Generates instructions from Pydantic's JSON schema
5. **Greedy Matching**: Finds first YAML block, ignores subsequent content
6. **Multiple Prefixes**: Supports both `yaml` and `yml` code block prefixes

### YAML vs JSON

**Advantages of YAML:**
- More human-readable (no quotes, cleaner syntax)
- Supports comments
- Better for multi-line strings
- Less punctuation (easier for LLMs to generate correctly)

**Disadvantages:**
- Indentation-sensitive (can cause parsing errors)
- More ambiguous (e.g., yes/no vs true/false)
- Slower parsing than JSON

### Pydantic Validation Benefits

Using Pydantic provides:
- **Type Validation**: Ensures correct types (int, str, bool, etc.)
- **Field Validation**: Custom validators for business logic
- **Default Values**: Automatic application of defaults
- **Optional Fields**: Proper handling of optional data
- **Nested Models**: Recursive validation of complex structures
- **Type Coercion**: Automatic type conversion where appropriate
- **Clear Errors**: Detailed validation error messages

### Pattern Matching

The regex pattern `r"^```(?:ya?ml)?(?P<yaml>[^`]*)"``:
- `^` - Start of string
- ``` ``` ``` - Literal triple backticks
- `(?:ya?ml)?` - Optional non-capturing group for "yaml" or "yml"
- `(?P<yaml>[^`]*)` - Named capture group: everything until backtick
- `MULTILINE | DOTALL` - Allows matching across lines and newlines

### Performance Considerations

- YAML parsing slower than JSON (more complex grammar)
- Pydantic validation adds overhead but catches errors early
- Regex matching is fast (one pass through text)
- No caching of parsed models (fresh validation each time)

### Limitations

1. **YAML Only**: Only supports YAML format (not TOML, JSON, etc.)
2. **First Match**: Only parses first YAML block found
3. **No Streaming**: Must have complete YAML before parsing
4. **Indentation Sensitive**: YAML errors from incorrect indentation
5. **Safe Load Only**: Doesn't support custom YAML tags
6. **Schema in Instructions**: JSON schema may confuse LLMs expecting YAML

### Security Considerations

- Uses `yaml.safe_load()` (prevents arbitrary code execution)
- No unsafe YAML features (tags, constructors)
- Pydantic validation provides additional safety
- Still vulnerable to denial-of-service via large YAML (no size limits)

### Use Cases

- Structured data extraction with type safety
- Configuration parsing from LLM output
- Form data collection with validation
- API response parsing
- Complex nested data structures
- Type-validated information retrieval
- Schema-enforced content generation
- Data pipeline validation steps

### Best Practices

1. **Use Pydantic Features**: Leverage validators, defaults, optional fields
2. **Clear Descriptions**: Use `Field(description=...)` for better LLM guidance
3. **Validate Types**: Define precise types (int, float, List[str], etc.)
4. **Provide Examples**: Include example YAML in prompts
5. **Handle Errors**: Always wrap parse() in try-except
6. **Test Edge Cases**: Verify behavior with optional fields, lists, nested models
7. **Consider JSON**: If indentation is problematic, use PydanticOutputParser instead
8. **Use OutputFixingParser**: Wrap for production robustness

### Comparison with PydanticOutputParser

| Feature | YamlOutputParser | PydanticOutputParser |
|---------|------------------|----------------------|
| Format | YAML | JSON |
| Readability | High (minimal syntax) | Medium (quotes, brackets) |
| LLM Friendliness | High (simpler) | Medium (more punctuation) |
| Parsing Speed | Slower | Faster |
| Error Prone | Indentation issues | Bracket/comma issues |
| Comments | Supported | Not supported |
| Use Case | Human-readable configs | Programmatic data |

### Integration Patterns

**Pattern 1: Fallback to JSON**
```python
try:
    result = yaml_parser.parse(text)
except OutputParserException:
    result = json_parser.parse(text)
```

**Pattern 2: Progressive Validation**
```python
def parse_and_enrich(parser, text):
    result = parser.parse(text)
    # Add computed fields
    result.timestamp = datetime.now()
    return result
```

**Pattern 3: Batch Processing**
```python
def parse_multiple(parser, texts):
    results = []
    for text in texts:
        try:
            results.append(parser.parse(text))
        except OutputParserException:
            continue  # Skip invalid entries
    return results
```

### Debugging Tips

1. **Print Raw YAML**: Debug extraction by printing yaml_str before parsing
2. **Validate Schema**: Use `model_json_schema()` to inspect generated schema
3. **Test Pydantic Separately**: Validate model with test data outside parser
4. **Check Indentation**: YAML errors often due to indentation
5. **Use Verbose Mode**: Enable Pydantic's validation errors for details
