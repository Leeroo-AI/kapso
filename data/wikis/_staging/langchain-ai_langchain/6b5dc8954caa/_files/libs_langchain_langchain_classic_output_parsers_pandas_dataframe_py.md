# File: `libs/langchain/langchain_classic/output_parsers/pandas_dataframe.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 171 |
| Classes | `PandasDataFrameOutputParser` |
| Imports | langchain_classic, langchain_core, pydantic, re, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Parses LLM output into Pandas DataFrame operations like row/column selection and aggregations.

**Mechanism:** Validates dataframe at initialization. Parses colon-separated requests (e.g., "column:name[1,2]", "mean:col[0..5]") using regex for array notation supporting comma-separated values, ranges, and column names. Executes operations on dataframe and returns results as dictionary.

**Significance:** Enables natural language querying of dataframes through LLMs. Supports complex operations with array parameters while validating bounds and column names, making data analysis accessible through conversational interfaces.
