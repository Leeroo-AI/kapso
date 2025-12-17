# File: `examples/online_serving/utils.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 26 |
| Functions | `get_first_model` |
| Imports | openai |

## Understanding

**Status:** ✅ Explored

**Purpose:** Shared utility functions for examples

**Mechanism:** Provides helper function `get_first_model` that retrieves the first available model from a vLLM server via the OpenAI models list endpoint. Includes proper error handling with informative messages for connection issues.

**Significance:** Common utility reused across multiple example files to avoid code duplication. Simplifies example code by handling model discovery logic.
