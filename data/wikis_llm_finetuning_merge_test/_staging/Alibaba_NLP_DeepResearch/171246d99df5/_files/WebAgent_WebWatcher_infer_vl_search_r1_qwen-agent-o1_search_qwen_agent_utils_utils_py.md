# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/utils/utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 551 |
| Classes | `PydanticJSONEncoder` |
| Functions | `append_signal_handler`, `get_local_ip`, `hash_sha256`, `print_traceback`, `has_chinese_chars`, `has_chinese_messages`, `get_basename_from_url`, `is_http_url`, `... +26 more` |
| Imports | base64, copy, hashlib, io, json, json5, os, pydantic, qwen_agent, re, ... +9 more |

## Understanding

**Status:** Explored

**Purpose:** Comprehensive utility module providing helper functions for file handling, URL processing, message formatting, JSON operations, image handling, and various common operations used throughout the Qwen agent framework.

**Mechanism:** Key function categories:
1. **System utilities**: `append_signal_handler()` for signal chaining, `get_local_ip()` for network discovery, `hash_sha256()` for hashing, `print_traceback()` for error logging
2. **Text analysis**: `has_chinese_chars()` and `has_chinese_messages()` for language detection using Unicode range checking
3. **URL/Path handling**: `get_basename_from_url()`, `is_http_url()`, `is_image()`, `sanitize_chrome_file_path()`, `sanitize_windows_file_path()` for cross-platform path normalization
4. **File operations**: `save_url_to_local_work_dir()` for downloading files, `save_text_to_file()`, `read_text_from_file()` with charset detection, `get_file_type()` for detecting PDF/DOCX/HTML/etc.
5. **JSON utilities**: `json_loads()` with json5 fallback, `PydanticJSONEncoder` for serializing Pydantic models, `json_dumps_pretty()` and `json_dumps_compact()`
6. **Message formatting**: `format_as_multimodal_message()` and `format_as_text_message()` for converting between message formats, handling file/image/video content with localized upload indicators (Chinese/English)
7. **Image utilities**: `encode_image_as_base64()`, `load_image_from_base64()`, `resize_image()` for image processing
8. **Code extraction**: `extract_code()` for pulling code from markdown blocks or JSON, `extract_urls()` and `extract_markdown_urls()`
9. **Chat utilities**: `build_text_completion_prompt()` for constructing Qwen chat format prompts with im_start/im_end markers

**Significance:** Central utility library that nearly every component in the agent framework depends on. Provides the glue code for handling multimodal content, cross-platform compatibility, message schema conversions, and common operations needed when building LLM agents that interact with files, URLs, and various content types.
