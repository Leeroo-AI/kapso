# File: `inference/file_tools/video_agent.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 92 |
| Classes | `VideoAgent` |
| Functions | `video_analysis` |
| Imports | asyncio, copy, file_tools, json, json5, openai, os, qwen_agent, re, sys, ... +1 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** Explored

**Purpose:** High-level agent tool that orchestrates video and audio file analysis, enabling the research agent to process multimedia content and answer user queries about video/audio files.

**Mechanism:** The `VideoAgent` class is a registered tool that accepts a query and a list of files. The `video_analysis()` async function iterates through provided file paths, creates parameters for each, and delegates to the `VideoAnalysis` tool for actual processing. Results from multiple files are aggregated with file-specific headers. The tool handles both batch processing of multiple files and error handling for individual file failures.

**Significance:** Bridge component that extends the research agent's capabilities to multimedia content. Enables answering questions about audio recordings (like lectures) or video content by delegating to the lower-level VideoAnalysis tool while providing a simpler interface for multi-file processing.
