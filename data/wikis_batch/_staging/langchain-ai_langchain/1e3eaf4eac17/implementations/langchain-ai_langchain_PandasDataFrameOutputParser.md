# PandasDataFrameOutputParser

## Metadata
- **Package**: `langchain-classic`
- **Module**: `langchain_classic.output_parsers.pandas_dataframe`
- **Source File**: `/tmp/praxium_repo_wjjl6pl8/libs/langchain/langchain_classic/output_parsers/pandas_dataframe.py`
- **Type**: Output Parser
- **Base Class**: `BaseOutputParser[dict[str, Any]]`
- **Dependencies**: `pandas`

## Overview

`PandasDataFrameOutputParser` is a specialized parser that enables natural language queries against Pandas DataFrames. It parses LLM outputs into DataFrame operations (row/column access, aggregations, filtering) and executes them to return results. This allows users to interact with structured data using conversational queries.

### Key Characteristics

- **Output Type**: `dict[str, Any]` (operation results)
- **DataFrame Operations**: Supports column access, row access, and aggregations
- **Array Parameters**: Handles index ranges and column filters
- **Format Instructions**: Provides comprehensive examples for LLM guidance
- **Validation**: Validates operations and column names
- **Dynamic Schema**: Format instructions adapt to DataFrame columns

## Code Reference

### Key Methods

#### `parse_array(array: str, original_request_params: str) -> tuple[list[int | str], str]`

Parses array parameters from request strings.

**Supported Formats:**
- `[1,3,5]` - Comma-separated integers
- `[1..5]` - Integer range (inclusive)
- `[col1,col2]` - Comma-separated column names

**Returns:**
- Tuple of (parsed array, stripped request params)

#### `parse(request: str) -> dict[str, Any]`

Parses and executes DataFrame operations.

**Request Format:**
```
<operation>:<parameter>[optional_array]
```

**Examples:**
- `column:age` - Get age column
- `row:5` - Get row 5
- `mean:salary[1..10]` - Mean of salary for rows 1-10
- `column:name[0,2,4]` - Get name column for specific rows

## Input/Output Contract

### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataframe` | `pd.DataFrame` | Required | Pandas DataFrame to query |

### Validation

- **DataFrame Type**: Must be a Pandas DataFrame or DataFrame-compatible object
- **Non-Empty**: DataFrame cannot be empty
- **Type Checking**: Validates using `issubclass(type(val), pd.DataFrame)`

### Supported Operations

#### 1. Column Access

**Format**: `column:<column_name>[optional_rows]`

**Examples:**
- `column:age` - Returns entire age column
- `column:age[1,3,5]` - Returns age for rows 1, 3, 5
- `column:name[0..4]` - Returns name for rows 0 through 4

#### 2. Row Access

**Format**: `row:<row_index>[optional_columns]`

**Examples:**
- `row:5` - Returns entire row 5
- `row:2[name,age]` - Returns name and age from row 2
- `row:0[name]` - Returns only name from row 0

#### 3. Aggregation Operations

**Format**: `<operation>:<column_name>[optional_rows]`

**Supported Operations**: Any valid Pandas Series method (mean, sum, median, std, min, max, count, etc.)

**Examples:**
- `mean:salary` - Mean of entire salary column
- `sum:revenue[1..10]` - Sum of revenue for rows 1-10
- `max:temperature[0,5,10]` - Max temperature from specific rows

### Array Parameter Formats

| Format | Example | Description |
|--------|---------|-------------|
| Comma-separated integers | `[1,3,5]` | Specific row indices |
| Integer range | `[1..5]` | Rows 1 through 5 (inclusive) |
| Column names | `[name,age]` | Specific columns |

### Error Responses

The parser expects LLM to return error strings starting with:
- `Invalid column:` - When column doesn't exist
- `Invalid operation:` - When operation isn't a valid DataFrame method

## Examples

### Basic Column Access

```python
from langchain_classic.output_parsers.pandas_dataframe import PandasDataFrameOutputParser
import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
})

parser = PandasDataFrameOutputParser(dataframe=df)

# Get entire column
result = parser.parse("column:name")
# Returns: {"name": pd.Series(['Alice', 'Bob', 'Charlie'])}

# Get specific rows
result = parser.parse("column:age[0,2]")
# Returns: {"age": pd.Series([25, 35])}
```

### Row Access

```python
# Get entire row
result = parser.parse("row:1")
# Returns: {"1": pd.Series({'name': 'Bob', 'age': 30, 'salary': 60000})}

# Get specific columns from row
result = parser.parse("row:1[name,age]")
# Returns: {"1": pd.Series({'name': 'Bob', 'age': 30})}

# Get single value
result = parser.parse("row:0[name]")
# Returns: {"0": 'Alice'}
```

### Aggregation Operations

```python
# Mean of entire column
result = parser.parse("mean:salary")
# Returns: {"mean": 60000.0}

# Sum of subset
result = parser.parse("sum:salary[0..1]")
# Returns: {"sum": 110000}

# Max value
result = parser.parse("max:age")
# Returns: {"max": 35}
```

### Range Queries

```python
# Range format [start..end]
result = parser.parse("column:name[0..1]")
# Returns: {"name": pd.Series(['Alice', 'Bob'])}

# Mean of range
result = parser.parse("mean:age[1..2]")
# Returns: {"mean": 32.5}
```

### Format Instructions

```python
# Get instructions for LLM
instructions = parser.get_format_instructions()
print(instructions)

# Output includes:
# - Operation format examples
# - Available columns from DataFrame
# - Array parameter syntax
# - Valid operation types
```

### With LLM Chain

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

df = pd.DataFrame({
    'product': ['A', 'B', 'C'],
    'price': [10, 20, 30],
    'quantity': [100, 200, 150]
})

parser = PandasDataFrameOutputParser(dataframe=df)
llm = ChatOpenAI()

prompt = PromptTemplate(
    template="""Given the following dataframe, answer the question:

Question: {question}

{format_instructions}""",
    input_variables=["question"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser

# Natural language queries
result = chain.invoke({"question": "What is the average price?"})
# LLM returns: "mean:price"
# Parser executes and returns: {"mean": 20.0}

result = chain.invoke({"question": "Show me the first two products"})
# LLM returns: "column:product[0..1]"
# Parser executes and returns: {"product": pd.Series(['A', 'B'])}
```

### Error Handling

```python
from langchain_core.exceptions import OutputParserException

parser = PandasDataFrameOutputParser(dataframe=df)

# Invalid column
try:
    parser.parse("column:invalid_column")
except OutputParserException as e:
    print(e)
    # "Requested index invalid_column is out of bounds."

# Invalid operation
try:
    parser.parse("invalid_op:name")
except OutputParserException as e:
    print(e)
    # "Unsupported request type 'invalid_op'."

# Malformed request
try:
    parser.parse("columnname")  # Missing colon
except OutputParserException as e:
    print(e)
    # "Request 'columnname' is not correctly formatted."
```

### Complex Queries

```python
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=10),
    'temperature': [20, 22, 21, 23, 24, 22, 20, 19, 21, 23],
    'humidity': [60, 65, 63, 68, 70, 66, 62, 60, 64, 67]
})

parser = PandasDataFrameOutputParser(dataframe=df)

# Average temperature for first week
result = parser.parse("mean:temperature[0..6]")
# Returns: {"mean": 21.71}

# Get specific dates
result = parser.parse("column:date[0,5,9]")
# Returns specific date values

# Standard deviation
result = parser.parse("std:humidity")
# Returns: {"std": 3.27}
```

### DataFrame Validation

```python
# Empty DataFrame raises error
try:
    empty_df = pd.DataFrame()
    parser = PandasDataFrameOutputParser(dataframe=empty_df)
except ValueError as e:
    print(e)
    # "DataFrame cannot be empty."

# Non-DataFrame raises error
try:
    parser = PandasDataFrameOutputParser(dataframe=[1, 2, 3])
except TypeError as e:
    print(e)
    # "Wrong type for 'dataframe', must be a subclass of Pandas DataFrame"
```

## Related Pages

- `BaseOutputParser` - Base class for all output parsers
- `StructuredOutputParser` - Parse structured JSON output
- `format_instructions.py` - Contains PANDAS_DATAFRAME_FORMAT_INSTRUCTIONS template
- Pandas DataFrame documentation

## Implementation Notes

### Design Decisions

1. **Operation-Parameter Format**: Uses `operation:parameter[array]` syntax for clear structure
2. **Dictionary Results**: Returns operations as dictionary keys for consistency
3. **Dynamic Column Validation**: Validates columns against actual DataFrame schema
4. **Flexible Arrays**: Supports multiple array formats for different use cases
5. **Pandas Integration**: Leverages Pandas' built-in operations via `getattr()`

### Parsing Logic

The parser follows this workflow:
1. Split request by `:` to separate operation and parameters
2. Check for array parameters in brackets `[...]`
3. If array present, parse it and extract base parameter
4. Execute appropriate operation based on request type
5. Return results as dictionary

### Array Parsing

Three formats supported:
1. **Integer list**: `[1,3,5]` → Uses `DataFrame.index.isin()`
2. **Integer range**: `[1..10]` → Converts to `list(range(1, 11))`
3. **String list**: `[col1,col2]` → Uses `DataFrame.columns.intersection()`

### Index vs. Column Operations

- **Column operations**: Filter by row indices, return column data
- **Row operations**: Filter by column names, return row data
- **Aggregations**: Apply operation to filtered column data

### Performance Considerations

- Operations execute directly on DataFrame (efficient)
- Index filtering using `.isin()` is optimized in Pandas
- Range operations convert to lists (memory consideration for large ranges)
- No caching of results (each parse executes fresh)

### Limitations

1. **Single Operation**: Can only execute one operation per parse call
2. **No Chaining**: Cannot combine multiple operations
3. **Limited Filtering**: Only supports index/column-based filtering
4. **No Conditional Logic**: Cannot filter by value conditions
5. **Return Format**: Always returns dictionary (not raw values)
6. **Column Name Constraints**: Column names with special characters may cause issues
7. **Array Size**: Large arrays in format string may hit parsing limits

### Security Considerations

- Uses `getattr()` for operation dispatch (could be exploited)
- No sandboxing of operations
- Could access any DataFrame method (including private ones)
- Recommendation: Validate DataFrame operations against whitelist in production

### Use Cases

- Natural language data exploration
- Interactive data analysis chatbots
- SQL-like queries via natural language
- Educational tools for teaching Pandas
- Quick data insights without writing code
- Dashboard query interfaces
- Data science assistants

### Best Practices

1. **Validate Operations**: Whitelist allowed operations for production
2. **Column Naming**: Use simple, alphanumeric column names
3. **Size Limits**: Consider DataFrame size (parser holds entire DF in memory)
4. **Error Handling**: Wrap parse calls in try-except for robustness
5. **Format Instructions**: Always provide format instructions to LLM
6. **Result Validation**: Verify results match expected types/values
7. **Performance**: For large DataFrames, consider filtering before parsing

### Integration Patterns

**Pattern 1: Conversational Data Analysis**
```python
chain = prompt | llm | parser
history = []
while True:
    question = input("Ask about the data: ")
    result = chain.invoke({"question": question, "history": history})
    history.append((question, result))
```

**Pattern 2: Multi-Step Analysis**
```python
# Chain multiple queries
results = []
for query in ["mean:salary", "max:age", "sum:revenue"]:
    results.append(parser.parse(query))
```

**Pattern 3: Validation Layer**
```python
ALLOWED_OPS = {"mean", "sum", "max", "min", "column", "row"}

def safe_parse(request):
    operation = request.split(":")[0]
    if operation not in ALLOWED_OPS:
        raise ValueError(f"Operation {operation} not allowed")
    return parser.parse(request)
```
