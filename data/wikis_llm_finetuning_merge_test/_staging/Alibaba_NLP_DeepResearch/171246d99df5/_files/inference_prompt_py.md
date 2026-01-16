# File: `inference/prompt.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 51 |

## Understanding

**Status:** Explored

**Purpose:** Central repository for system prompts that define the behavior of the DeepResearch agent and its information extraction capabilities.

**Mechanism:** Contains two key prompts: 1) `SYSTEM_PROMPT` - Defines the agent as a "deep research assistant" and specifies available tools (search, visit, PythonInterpreter, google_scholar, parse_file) with their JSON schemas. Instructs the agent to conduct multi-source investigations and wrap final answers in `<answer></answer>` tags. Includes current date placeholder. 2) `EXTRACTOR_PROMPT` - Template for webpage content extraction that guides the model to locate relevant sections, extract key evidence, and produce a structured JSON summary with "rational", "evidence", and "summary" fields.

**Significance:** Core configuration file that shapes the agent's identity and behavior. The system prompt establishes the research methodology, tool usage format, and output structure. The extractor prompt ensures consistent information extraction from web pages.
