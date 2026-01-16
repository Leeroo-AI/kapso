# File: `WebAgent/WebWalker/src/app.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 271 |
| Classes | `VisitPage` |
| Functions | `extract_links_with_text` |
| Imports | PIL, agent, asyncio, base64, bs4, json, json5, os, qwen_agent, re, ... +2 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides a Streamlit-based web UI for interactive WebWalker demonstrations, allowing users to explore websites and ask questions through a visual interface with screenshots.

**Mechanism:** The application creates a Streamlit interface with: (1) Sidebar controls for max action count and example selection; (2) Input forms for website URL and query; (3) Real-time visualization of agent actions including thoughts, button clicks, and screenshots. The `extract_links_with_text()` function parses HTML using BeautifulSoup to find clickable links/buttons, filtering to same-domain URLs and storing button-to-URL mappings in BUTTON_URL_ADIC.json. The `VisitPage` tool class (registered with Qwen Agent) handles button clicks by looking up URLs and fetching page content via async `get_info()`. Screenshots are saved as base64 PNGs and displayed in the UI. The agent runs with the query and website, streaming thoughts and memory updates.

**Significance:** User-facing demonstration component that makes WebWalker accessible to non-technical users. Provides visual feedback of the agent's exploration process with screenshots and step-by-step reasoning display.
