# File: `examples/online_serving/utils.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 26 |
| Functions | `get_first_model` |
| Imports | openai |

## Understanding

**Status:** ✅ Explored

**Purpose:** Shared utility for example scripts to get model name

**Mechanism:** Provides get_first_model() function that queries /v1/models endpoint, handles APIConnectionError with detailed troubleshooting message, validates that at least one model exists, and returns the first model ID. Used by multiple example scripts to avoid hardcoding model names.

**Significance:** DRY (Don't Repeat Yourself) utility shared across examples. Improves example robustness by auto-discovering available models and providing helpful error messages. Makes examples more portable - they work with any model without modification. Shows good practices for error handling and user guidance in client code.
