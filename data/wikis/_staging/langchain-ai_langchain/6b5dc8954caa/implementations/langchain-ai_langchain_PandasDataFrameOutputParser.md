{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Output Parsing]], [[domain::Data Analysis]], [[domain::Pandas Integration]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Output parser `PandasDataFrameOutputParser` that interprets LLM output as DataFrame query operations.

=== Description ===

The `PandasDataFrameOutputParser` class parses LLM-generated query strings and executes them against a Pandas DataFrame. It supports column/row selection, array indexing, range slicing, and DataFrame operations like mean, sum, etc. The parser validates operations and provides meaningful error messages for invalid queries.

=== Usage ===

Use this parser to enable natural language queries over tabular data. The LLM generates structured query strings that the parser executes against the DataFrame, enabling conversational data analysis.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/output_parsers/pandas_dataframe.py libs/langchain/langchain_classic/output_parsers/pandas_dataframe.py]
* '''Lines:''' 1-172

=== Signature ===
<syntaxhighlight lang="python">
class PandasDataFrameOutputParser(BaseOutputParser[dict[str, Any]]):
    """Parse an output using Pandas DataFrame format."""

    dataframe: Any  # pd.DataFrame

    def parse(self, request: str) -> dict[str, Any]:
        """Parse query string and execute against DataFrame."""

    def get_format_instructions(self) -> str:
        """Return instructions with available columns."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.output_parsers import PandasDataFrameOutputParser
</syntaxhighlight>

== I/O Contract ==

=== Configuration ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| dataframe || pd.DataFrame || Yes || The DataFrame to query against
|}

=== Query Syntax ===
{| class="wikitable"
|-
! Format !! Example !! Description
|-
| column:name || column:salary || Get entire column
|-
| row:n || row:5 || Get entire row
|-
| column:name[n,m] || column:age[1,3] || Get column values at specific rows
|-
| row:n[cols] || row:1[name,age] || Get row values for specific columns
|-
| operation:col[range] || mean:salary[0..10] || Apply operation to column slice
|}

=== Outputs ===
{| class="wikitable"
|-
! Type !! Description
|-
| dict[str, Any] || Query result keyed by column/operation name
|}

== Usage Examples ==

=== Basic DataFrame Queries ===
<syntaxhighlight lang="python">
import pandas as pd
from langchain_classic.output_parsers import PandasDataFrameOutputParser

# Create sample DataFrame
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "salary": [50000, 60000, 70000],
})

parser = PandasDataFrameOutputParser(dataframe=df)

# Get column
result = parser.parse("column:name")
print(result)  # {'name': Series(['Alice', 'Bob', 'Charlie'])}

# Get row
result = parser.parse("row:1")
print(result)  # {'1': Series(name='Bob', age=30, salary=60000)}

# Get column with row filter
result = parser.parse("column:salary[0,2]")
print(result)  # {'salary': Series([50000, 70000])}
</syntaxhighlight>

=== Aggregation Operations ===
<syntaxhighlight lang="python">
# Calculate mean salary
result = parser.parse("mean:salary")
print(result)  # {'mean': 60000.0}

# Mean for specific rows
result = parser.parse("mean:salary[0..1]")
print(result)  # {'mean': 55000.0}

# Sum of ages
result = parser.parse("sum:age")
print(result)  # {'sum': 90}
</syntaxhighlight>

=== With LLM Chain ===
<syntaxhighlight lang="python">
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

parser = PandasDataFrameOutputParser(dataframe=df)

prompt = PromptTemplate.from_template(
    "Given a DataFrame with employee data, translate this question "
    "into a DataFrame query:\n\n"
    "Question: {question}\n\n"
    "{format_instructions}"
)

llm = ChatOpenAI(model="gpt-4")
chain = prompt | llm | parser

result = chain.invoke({
    "question": "What is the average salary?",
    "format_instructions": parser.get_format_instructions()
})
print(result)  # {'mean': 60000.0}
</syntaxhighlight>

=== Error Handling ===
<syntaxhighlight lang="python">
from langchain_core.exceptions import OutputParserException

try:
    # Invalid column
    parser.parse("column:invalid_col")
except OutputParserException as e:
    print(e)  # "Invalid column" error message

try:
    # Invalid operation
    parser.parse("invalid_op:salary")
except OutputParserException as e:
    print(e)  # "Unsupported request type" error
</syntaxhighlight>

== Related Pages ==
* [[related_to::Implementation:langchain-ai_langchain_format_instructions]]
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]

