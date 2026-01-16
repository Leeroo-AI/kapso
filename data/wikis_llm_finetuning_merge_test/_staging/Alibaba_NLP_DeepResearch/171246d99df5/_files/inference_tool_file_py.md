# File: `inference/tool_file.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 141 |
| Classes | `FileParser` |
| Functions | `file_parser` |
| Imports | asyncio, bdb, copy, file_tools, json, json5, openai, os, pdb, qwen_agent, ... +4 more |

## Understanding

**Status:** Explored

**Purpose:** Agent tool wrapper that provides a unified interface for parsing multiple user-uploaded files, handling both document files and multimedia content.

**Mechanism:** The `FileParser` class (registered as "parse_file" tool) accepts a list of file names and routes them appropriately: regular document files (PDF, DOCX, PPTX, etc.) are processed via the `file_parser()` async function using `SingleFileParser`, while audio files (.mp3) are routed to the `VideoAgent` for transcription and analysis. The `file_parser()` function resolves paths (handling both URLs and local files), calls the parser, and aggregates results. Output is compressed if exceeding token limits using the `compress()` function.

**Significance:** Tool interface that exposes document parsing capabilities to the ReAct agent. Acts as a dispatcher that unifies document and multimedia file handling, allowing the agent to use a single tool call to process diverse file types uploaded by users.
