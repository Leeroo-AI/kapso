# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/gpt4o/utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 249 |
| Classes | `CustomNamespace`, `ArgumentParser`, `APIException` |
| Functions | `str2bool`, `get_args`, `is_float`, `is_bool`, `parse_args`, `compare_dict_structure`, `dict_to_sorted_str`, `truncate_long_strings`, `... +2 more` |
| Imports | argparse, json, os, pandas, sys, typing |

## Understanding

**Status:** Explored

**Purpose:** Provides utility functions and classes for argument parsing, file loading, data manipulation, and API response handling across the gpt4o toolkit.

**Mechanism:** Key components:
- `ArgumentParser` class: Custom argparse wrapper supporting standard LLM parameters (model, temperature, top-p, max-tokens, etc.) plus unknown argument handling with automatic type conversion
- `CustomNamespace`: Extends argparse.Namespace with a `pop()` method for convenient parameter extraction
- `get_args()`, `parse_args()`: Functions for command-line argument parsing
- Type utilities: `str2bool()`, `is_float()`, `is_bool()` for string-to-type conversion
- Data utilities:
  - `truncate_long_strings(d, max_len)`: Recursively truncates strings in nested dicts/lists for logging
  - `compare_dict_structure()`: Compares structural equivalence of dictionaries
  - `dict_to_sorted_str()`: Deterministic string representation of dicts
  - `load_file2list()`: Multi-format file loader supporting JSONL, CSV, Excel, JSON with pandas, includes UUID-based deduplication
- `APIException`: Custom exception class with optional error codes
- `openai_ret_wrapper()`: Converts Claude API responses (via MIT channel) to OpenAI-compatible format

**Significance:** Foundational utility module that standardizes common operations across the toolkit. Provides essential infrastructure for CLI tools (argument parsing), data pipelines (file loading), debugging (string truncation), and API compatibility (response wrapping).
