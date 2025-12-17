# File: `libs/langchain/langchain_classic/output_parsers/pandas_dataframe.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 171 |
| Classes | `PandasDataFrameOutputParser` |
| Imports | langchain_classic, langchain_core, pydantic, re, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Parse LLM text output into pandas DataFrame queries and operations, enabling natural language data extraction from tabular data.

**Mechanism:** Takes a pandas DataFrame and parses LLM output in the format "operation:target[optional_array]" where operations include column/row access and DataFrame methods (mean, sum, etc.). Uses regex to parse array specifications ([1,3,5] or [0..4] or [col_names]), validates column names and indices against the DataFrame, and returns results as dictionaries.

**Significance:** Enables natural language querying of pandas DataFrames by having LLMs output structured query strings that can be parsed and executed, bridging conversational AI with data analysis workflows.
