# File: `WebAgent/WebDancer/demos/gui/web_ui.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 389 |
| Classes | `WebUI` |
| Imports | os, pprint, qwen_agent, re, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides the `WebUI` class that creates a Gradio-based web chat interface for interacting with agents.

**Mechanism:** The `WebUI` class accepts one or more agents and builds a complete chat interface. Key methods: (1) `__init__()` - configures user/agent avatars, names, descriptions, and suggested prompts, (2) `run()` - launches the Gradio interface with a Chatbot component, MultimodalInput for text/files, agent selector dropdown (for multi-agent), and example prompts, (3) `add_text()` - handles user input including text and file attachments (images, audio, video), (4) `agent_run()` - executes the selected agent's `run()` method and streams responses to the chatbot, handling function call formatting via `convert_fncall_to_text()`, (5) `add_mention()` - enables @agent mentions for multi-agent scenarios. Supports LaTeX rendering and configurable concurrency limits.

**Significance:** Core UI infrastructure for WebDancer. This provides the complete web-based chat experience, enabling users to interact with search agents through a browser.
