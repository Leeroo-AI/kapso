# File: `inference/file_tools/utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 542 |
| Classes | `PydanticJSONEncoder` |
| Functions | `append_signal_handler`, `get_local_ip`, `hash_sha256`, `print_traceback`, `has_chinese_chars`, `has_chinese_messages`, `get_basename_from_url`, `is_http_url`, `... +25 more` |
| Imports | base64, copy, hashlib, io, json, json5, os, pydantic, qwen_agent, re, ... +9 more |

## Understanding

**Status:** Explored

**Purpose:** Comprehensive utility library providing helper functions for file handling, URL processing, text manipulation, image encoding, message formatting, and JSON operations used throughout the DeepResearch system.

**Mechanism:** Provides a wide range of utilities: `hash_sha256()` for content hashing and caching; `is_http_url()`, `get_basename_from_url()`, `sanitize_chrome_file_path()` for URL/path handling; `save_url_to_local_work_dir()` for downloading files; `has_chinese_chars()` for language detection; `encode_image_as_base64()` and `resize_image()` for image processing; `format_as_multimodal_message()` and `format_as_text_message()` for converting messages between formats; `extract_code()` for parsing code blocks; `json_loads()` with JSON5 fallback; and `build_text_completion_prompt()` for constructing chat prompts.

**Significance:** Foundation utility module that provides shared functionality across the entire codebase. Critical for consistent file handling, message formatting, and data transformation operations required by the agent, tools, and file processing components.
