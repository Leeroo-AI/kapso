# File: `WebAgent/WebDancer/demos/assistant_qwq_chat.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 140 |
| Functions | `init_dev_search_agent_service`, `app_gui` |
| Imports | demos, os, qwen_agent |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Main entry point for launching the WebDancer demo application with a Gradio web interface.

**Mechanism:** Contains two key functions: (1) `init_dev_search_agent_service()` - creates a SearchAgent configured with OpenAI-compatible LLM backend (via TextChatAtOAI), defines the system prompt as a "Web Information Seeking Master" with persistent search principles, and sets up the agent with search/visit tools. (2) `app_gui()` - initializes the WebDancer-QwQ-32B agent and launches a Gradio-based WebUI on port 7860. The system prompt includes time awareness and Chinese language support. Provides sample queries in both English and Chinese covering diverse topics (sports, travel, AI, etc.).

**Significance:** Primary application launcher for WebDancer. This is the main file users would run to start the demo (`python assistant_qwq_chat.py`), providing the complete web-based chat interface for interacting with the search agent.
