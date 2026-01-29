# Gated MCP Server Design

## Overview

The Gated MCP Server provides selective tool exposure for different Claude Code agents across the Kapso system. Each agent can enable only the tools it needs via preset configurations.

## Problem Statement

Multiple Claude Code agents exist across the system, each needing different subsets of tools:

| Use Case | Needs |
|----------|-------|
| KnowledgeMerger | Full KG access (search, index, edit) |
| Ideation Agent | Idea search + web research |
| Implementation Agent | Code search + web research |
| Context Manager | Read-only search (idea + code) |

**Current limitation**: The existing MCP server exposes ALL tools, and access is controlled only via `allowed_tools` in Claude Code config. This leads to:
1. All tools still registered (visible in tool list)
2. No way to configure tool behavior per-agent
3. Manual sync required between presets and `allowed_tools`

## Solution: Gate-Based Architecture

### Core Concepts

1. **Gate**: A group of related tools with shared backend and configuration
2. **Preset**: A named configuration that enables specific gates with custom params
3. **Lazy Initialization**: Backends only initialize when their gate's tools are called

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Gated MCP Server                                 │
│                                                                     │
│  Environment Variables:                                             │
│  ├─ MCP_PRESET: "ideation" (uses preset config)                    │
│  │   OR                                                             │
│  ├─ MCP_ENABLED_GATES: "kg,idea" (manual gate list)                │
│  └─ KG_INDEX_PATH: "/path/to/.index" (backend config)              │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Shared Backend (Singleton, Lazy)                │   │
│  │  KGGraphSearch instance - shared by kg, idea, code gates    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│  ┌───────────────────────────┼──────────────────────────────────┐  │
│  │                    Gate Registry                              │  │
│  │       ┌───────────────────┼───────────────────────┐          │  │
│  │       ▼                   ▼                       ▼          │  │
│  │  ┌─────────┐        ┌─────────┐             ┌─────────┐      │  │
│  │  │ kg_gate │        │idea_gate│             │code_gate│      │  │
│  │  │(config) │        │(config) │             │(config) │      │  │
│  │  └─────────┘        └─────────┘             └─────────┘      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Research Gate (Separate Backend)                 │  │
│  │  Researcher instance - independent, uses OpenAI web_search   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Gates

### KG Gate (`kg`)

Full Knowledge Graph operations for reading and writing wiki pages.

**Tools:**
- `search_knowledge` - Semantic search with filters
- `get_wiki_page` - Retrieve page by title/ID
- `kg_index` - Index pages into KG
- `kg_edit` - Edit existing pages
- `get_page_structure` - Get section definitions

**Config params:**
- `include_content`: bool (default True) - Include full content in search results

**Backend:** KGGraphSearch (shared singleton)

### Idea Gate (`idea`)

Search for conceptual knowledge (Principles + Heuristics).

Uses KGGraphSearch directly with page type filter: `["Principle", "Heuristic"]`

**Tools:**
- `wiki_idea_search` - Search principles and heuristics

**Config params:**
- `top_k`: int (default 5) - Default number of results
- `use_llm_reranker`: bool (default True) - Use LLM for reranking
- `include_content`: bool (default True) - Include full content

**Backend:** KGGraphSearch (shared singleton) with page type filter

### Code Gate (`code`)

Search for code knowledge (Implementations + Environments).

Uses KGGraphSearch directly with page type filter: `["Implementation", "Environment"]`

**Tools:**
- `wiki_code_search` - Search implementations and environments

**Config params:**
- `top_k`: int (default 5) - Default number of results
- `use_llm_reranker`: bool (default True) - Use LLM for reranking
- `include_content`: bool (default True) - Include full code content

**Backend:** KGGraphSearch (shared singleton) with page type filter

### Research Gate (`research`)

Deep web research using OpenAI's web_search tool.

**Tools:**
- `research_idea` - Research conceptual ideas from web
- `research_implementation` - Research code implementations from web
- `research_study` - Generate comprehensive research report

**Config params:**
- `default_depth`: str (default "deep") - Research depth
- `default_top_k`: int (default 5) - Default number of results

**Backend:** Researcher (independent singleton)

## Presets

### `merger`

For KnowledgeMerger - needs full KG access.

```python
gates = {
    "kg": GateConfig(enabled=True, params={"include_content": True}),
}
```

### `ideation`

For ideation phase - needs idea search + research.

```python
gates = {
    "idea": GateConfig(enabled=True, params={"top_k": 10, "use_llm_reranker": True}),
    "research": GateConfig(enabled=True, params={"default_depth": "deep", "default_top_k": 5}),
}
```

### `implementation`

For implementation phase - needs code search + research.

```python
gates = {
    "code": GateConfig(enabled=True, params={"top_k": 5, "include_content": True}),
    "research": GateConfig(enabled=True, params={"default_depth": "deep", "default_top_k": 3}),
}
```

### `context`

For context managers - read-only search.

```python
gates = {
    "idea": GateConfig(enabled=True, params={"top_k": 5, "use_llm_reranker": False}),
    "code": GateConfig(enabled=True, params={"top_k": 5, "include_content": False}),
}
```

### `full`

All tools enabled with default settings (for debugging/admin).

```python
gates = {
    "kg": GateConfig(enabled=True),
    "idea": GateConfig(enabled=True),
    "code": GateConfig(enabled=True),
    "research": GateConfig(enabled=True),
}
```

## File Structure

```
src/knowledge/gated_mcp/
├── __init__.py           # Public exports
├── server.py             # Main MCP server entry point
├── presets.py            # Preset definitions and helper functions
├── backends.py           # Shared backend singletons (lazy init)
├── gates/
│   ├── __init__.py       # Gate exports
│   ├── base.py           # ToolGate abstract base class
│   ├── kg_gate.py        # KG tools implementation
│   ├── idea_gate.py      # Idea search implementation
│   ├── code_gate.py      # Code search implementation
│   └── research_gate.py  # Research tools implementation
└── README.md             # Usage documentation
```

## Key Design Decisions

### 1. Lazy Backend Initialization

Backends are initialized on first tool call, not at server startup. This:
- Prevents failures when gates are disabled
- Reduces startup time
- Allows graceful degradation

### 2. Shared KGGraphSearch Backend

The `kg`, `idea`, and `code` gates all share a single KGGraphSearch instance:
- Reduces memory usage
- Single connection to Weaviate/Neo4j
- Consistent caching

### 3. Async/Sync Bridge

MCP handlers are async, but search backends are sync. We use `run_in_executor`:
```python
async def handle_call(self, ...):
    result = await self._run_sync(lambda: self._search.search(...))
```

### 4. Tool Name Collision Detection

Server startup validates no two gates provide the same tool name:
```python
if tool.name in tool_to_gate:
    raise ValueError(f"Tool name collision: '{tool.name}'")
```

### 5. Auto-Generated `allowed_tools`

Helper function generates Claude Code's `allowed_tools` from preset:
```python
tools = get_allowed_tools_for_preset("ideation", "gated-knowledge")
# Returns: ["Read", "Write", "Bash", "mcp__gated-knowledge__wiki_idea_search", ...]
```

## Usage

### Running the Server

```bash
# With preset
MCP_PRESET=ideation python -m src.knowledge.gated_mcp.server

# With custom gates
MCP_ENABLED_GATES=idea,code python -m src.knowledge.gated_mcp.server

# With KG index path
KG_INDEX_PATH=/path/to/.index MCP_PRESET=merger python -m src.knowledge.gated_mcp.server
```

### Connecting to Claude Code

```python
from src.knowledge.gated_mcp import get_allowed_tools_for_preset

# Build MCP config
mcp_servers = {
    "gated-knowledge": {
        "command": "python",
        "args": ["-m", "src.knowledge.gated_mcp.server"],
        "cwd": str(project_root),
        "env": {"MCP_PRESET": "ideation", "KG_INDEX_PATH": "..."},
    }
}

# Auto-generate allowed_tools
allowed_tools = get_allowed_tools_for_preset("ideation", "gated-knowledge")

# Configure agent
agent_specific = {
    "allowed_tools": allowed_tools,
    "mcp_servers": mcp_servers,
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_PRESET` | Preset name to use | (none, falls back to `MCP_ENABLED_GATES`) |
| `MCP_ENABLED_GATES` | Comma-separated gate names | (none, falls back to "full") |
| `KG_INDEX_PATH` | Path to .index file for KG config | (none) |
| `OPENAI_API_KEY` | For embeddings and research | Required |
| `NEO4J_URI` | Neo4j connection | `bolt://localhost:7687` |
| `WEAVIATE_URL` | Weaviate server | `http://localhost:8081` |

## Error Handling

### Empty Results

Each gate provides helpful guidance when no results are found:
```
No conceptual knowledge found for: "query"

Try:
- Broader search terms
- Different phrasing
- Use wiki_code_search for implementation details
```

### Backend Initialization Failure

If a backend fails to initialize, the error is logged and re-raised with context:
```
RuntimeError: KGGraphSearch initialization failed: Connection refused
```

### Tool Name Collision

Server startup fails fast if two gates provide the same tool name:
```
ValueError: Tool name collision: 'search_knowledge' in both 'kg' and 'custom' gates
```
