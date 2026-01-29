# Gated MCP Implementation Tasks

## Overview

Implementation tasks for the Gated MCP Server. Tasks are ordered by dependency.

---

## Phase 1: Core Infrastructure

### Task 1.1: Create Directory Structure

**Priority:** High  
**Estimated Complexity:** Low

Create the base directory structure:

```
src/knowledge/gated_mcp/
├── __init__.py
├── server.py
├── presets.py
├── backends.py
├── gates/
│   ├── __init__.py
│   ├── base.py
│   ├── kg_gate.py
│   ├── idea_gate.py
│   ├── code_gate.py
│   └── research_gate.py
└── README.md
```

**Acceptance Criteria:**
- [ ] All directories created
- [ ] Empty `__init__.py` files in place

---

### Task 1.2: Implement `presets.py`

**Priority:** High  
**Estimated Complexity:** Low  
**Dependencies:** Task 1.1

Implement preset configuration system:

- [ ] `GateConfig` dataclass with `enabled` and `params`
- [ ] `Preset` dataclass with `name`, `description`, `gates`
- [ ] `PRESETS` dictionary with all preset definitions:
  - `merger`
  - `ideation`
  - `implementation`
  - `context`
  - `full`
- [ ] `GATE_TOOL_NAMES` mapping for allowed_tools generation
- [ ] `get_preset(name)` function
- [ ] `get_allowed_tools_for_preset(preset_name, mcp_server_name)` function

**Acceptance Criteria:**
- [ ] All presets defined with correct gate configurations
- [ ] `get_allowed_tools_for_preset` returns correct tool list
- [ ] Unit tests pass

---

### Task 1.3: Implement `backends.py`

**Priority:** High  
**Estimated Complexity:** Medium  
**Dependencies:** Task 1.1

Implement shared backend singletons with lazy initialization:

- [ ] Thread-safe singleton pattern with `threading.Lock`
- [ ] `get_kg_search_backend()` - lazy init KGGraphSearch
  - [ ] Read `KG_INDEX_PATH` env var for config
  - [ ] Parse `.index` file if present
  - [ ] Handle initialization errors gracefully
- [ ] `get_researcher_backend()` - lazy init Researcher
- [ ] `reset_backends()` - for testing

**Acceptance Criteria:**
- [ ] Backends only initialize on first call
- [ ] Thread-safe (no race conditions)
- [ ] Proper error messages on failure
- [ ] `reset_backends()` cleans up properly

---

### Task 1.4: Implement `gates/base.py`

**Priority:** High  
**Estimated Complexity:** Low  
**Dependencies:** Task 1.2

Implement the `ToolGate` abstract base class:

- [ ] `name` and `description` class attributes
- [ ] `__init__(config: Optional[GateConfig])` 
- [ ] `get_param(key, default)` helper method
- [ ] Abstract `get_tools() -> List[Tool]`
- [ ] Abstract `handle_call(tool_name, arguments) -> Optional[List[TextContent]]`
- [ ] `get_tool_names() -> List[str]` helper
- [ ] `_run_sync(func, *args, **kwargs)` async helper for sync backends

**Acceptance Criteria:**
- [ ] ABC properly defined
- [ ] `_run_sync` correctly wraps sync functions in executor

---

## Phase 2: Gate Implementations

### Task 2.1: Implement `gates/kg_gate.py`

**Priority:** High  
**Estimated Complexity:** Medium  
**Dependencies:** Task 1.3, Task 1.4

Implement the KG gate with all tools:

- [ ] `KGGate` class extending `ToolGate`
- [ ] `get_tools()` returning 5 tools:
  - [ ] `search_knowledge`
  - [ ] `get_wiki_page`
  - [ ] `kg_index`
  - [ ] `kg_edit`
  - [ ] `get_page_structure`
- [ ] `handle_call()` dispatching to handlers
- [ ] `_handle_search()` - uses `include_content` param
- [ ] `_handle_get_page()`
- [ ] `_handle_index()` - directory and single page modes
- [ ] `_handle_edit()`
- [ ] `_handle_get_structure()` - reads from wiki_structure/
- [ ] `_format_search_results()` helper
- [ ] `_format_page()` helper

**Acceptance Criteria:**
- [ ] All 5 tools work correctly
- [ ] Uses shared KGGraphSearch backend
- [ ] Respects `include_content` config param
- [ ] Proper error handling and messages

---

### Task 2.2: Implement `gates/idea_gate.py`

**Priority:** High  
**Estimated Complexity:** Low  
**Dependencies:** Task 1.3, Task 1.4

Implement the idea search gate using KGGraphSearch directly with page type filter.

- [ ] `IdeaGate` class extending `ToolGate`
- [ ] Define `IDEA_TYPES = ["Principle", "Heuristic"]` (from PageType enum)
- [ ] `get_tools()` returning `wiki_idea_search`
  - [ ] Dynamic default `top_k` from config in description
- [ ] `handle_call()` implementation
  - [ ] Use KGGraphSearch directly with `KGSearchFilters(page_types=IDEA_TYPES)`
  - [ ] Use `top_k`, `use_llm_reranker`, `include_content` from config
  - [ ] Helpful empty result message
- [ ] `_format_results()` helper for output formatting

**Acceptance Criteria:**
- [ ] Tool works correctly
- [ ] Uses KGGraphSearch directly (not WikiIdeaSearch wrapper)
- [ ] Filters by page_types: ["Principle", "Heuristic"]
- [ ] Respects config params
- [ ] Helpful error messages

---

### Task 2.3: Implement `gates/code_gate.py`

**Priority:** High  
**Estimated Complexity:** Low  
**Dependencies:** Task 1.3, Task 1.4

Implement the code search gate using KGGraphSearch directly with page type filter.

- [ ] `CodeGate` class extending `ToolGate`
- [ ] Define `CODE_TYPES = ["Implementation", "Environment"]` (from PageType enum)
- [ ] `get_tools()` returning `wiki_code_search`
- [ ] `handle_call()` implementation
  - [ ] Use KGGraphSearch directly with `KGSearchFilters(page_types=CODE_TYPES)`
  - [ ] Use `top_k`, `use_llm_reranker`, `include_content` from config
  - [ ] Different formatting based on `include_content`
- [ ] `_format_results()` helper for output formatting

**Acceptance Criteria:**
- [ ] Tool works correctly
- [ ] Uses KGGraphSearch directly (not WikiCodeSearch wrapper)
- [ ] Filters by page_types: ["Implementation", "Environment"]
- [ ] Respects `include_content` param (full vs overview)
- [ ] Helpful error messages

---

### Task 2.4: Implement `gates/research_gate.py`

**Priority:** High  
**Estimated Complexity:** Medium  
**Dependencies:** Task 1.3, Task 1.4

Implement the research gate:

- [ ] `ResearchGate` class extending `ToolGate`
- [ ] `get_tools()` returning 3 tools:
  - [ ] `research_idea`
  - [ ] `research_implementation`
  - [ ] `research_study`
- [ ] `handle_call()` dispatching to handlers
- [ ] `_handle_idea()` with empty result handling
- [ ] `_handle_implementation()` with empty result handling
- [ ] `_handle_study()`

**Acceptance Criteria:**
- [ ] All 3 tools work correctly
- [ ] Uses Researcher backend
- [ ] Respects `default_depth` and `default_top_k` params
- [ ] Helpful empty result messages

---

### Task 2.5: Implement `gates/__init__.py`

**Priority:** Low  
**Estimated Complexity:** Low  
**Dependencies:** Tasks 2.1-2.4

Export all gate classes:

- [ ] Export `ToolGate`, `KGGate`, `IdeaGate`, `CodeGate`, `ResearchGate`

---

## Phase 3: Server Implementation

### Task 3.1: Implement `server.py`

**Priority:** High  
**Estimated Complexity:** Medium  
**Dependencies:** Phase 2

Implement the main MCP server:

- [ ] `GATE_CLASSES` registry mapping names to classes
- [ ] `_resolve_configuration()` function
  - [ ] Check `MCP_PRESET` env var first
  - [ ] Fall back to `MCP_ENABLED_GATES`
  - [ ] Default to "full" preset
- [ ] `create_gated_mcp_server()` function
  - [ ] Initialize only enabled gates with their configs
  - [ ] Build tool registry with collision detection
  - [ ] Register `list_tools` handler
  - [ ] Register `call_tool` handler with error handling
- [ ] `run_server()` async function
- [ ] `main()` entry point

**Acceptance Criteria:**
- [ ] Server starts with correct preset
- [ ] Only enabled gates' tools are registered
- [ ] Tool name collisions detected at startup
- [ ] Proper error handling in tool calls

---

### Task 3.2: Implement `__init__.py`

**Priority:** Low  
**Estimated Complexity:** Low  
**Dependencies:** Task 3.1

Export public API:

- [ ] Export `PRESETS`, `Preset`, `GateConfig`
- [ ] Export `get_preset`, `get_allowed_tools_for_preset`
- [ ] Export `create_gated_mcp_server`

---

## Phase 4: Documentation & Testing

### Task 4.1: Create README.md

**Priority:** Medium  
**Estimated Complexity:** Low  
**Dependencies:** Phase 3

Write usage documentation:

- [ ] Quick start examples
- [ ] Environment variables reference
- [ ] Preset descriptions
- [ ] Troubleshooting guide

---

### Task 4.2: Unit Tests for Presets

**Priority:** Medium  
**Estimated Complexity:** Low  
**Dependencies:** Task 1.2

Create `tests/test_gated_mcp_presets.py`:

- [ ] Test `get_preset()` returns correct preset
- [ ] Test `get_preset()` raises for unknown preset
- [ ] Test `get_allowed_tools_for_preset()` returns correct tools
- [ ] Test all presets have valid gate names

---

### Task 4.3: Unit Tests for Backends

**Priority:** Medium  
**Estimated Complexity:** Medium  
**Dependencies:** Task 1.3

Create `tests/test_gated_mcp_backends.py`:

- [ ] Test lazy initialization (backend not created until called)
- [ ] Test singleton behavior (same instance returned)
- [ ] Test `reset_backends()` clears state
- [ ] Test error handling when backend init fails

---

### Task 4.4: Unit Tests for Gates

**Priority:** Medium  
**Estimated Complexity:** Medium  
**Dependencies:** Phase 2

Create `tests/test_gated_mcp_gates.py`:

- [ ] Test each gate's `get_tools()` returns expected tools
- [ ] Test each gate's `get_tool_names()` matches
- [ ] Test config params are respected
- [ ] Test `_run_sync()` works correctly

---

### Task 4.5: Integration Tests (Unit)

**Priority:** Medium  
**Estimated Complexity:** High  
**Dependencies:** Phase 3

Create `tests/test_gated_mcp_integration.py`:

- [ ] Test server creation with each preset
- [ ] Test tool calls work end-to-end (mocked backends)
- [ ] Test preset switching via env vars
- [ ] Test `get_allowed_tools_for_preset` matches actual tools

---

### Task 4.6: Standalone Claude Code Integration Test - Idea Search

**Priority:** Medium  
**Estimated Complexity:** Medium  
**Dependencies:** Phase 3

Create `tests/test_gated_mcp_idea_search.py`:

Standalone script to test idea search gate with Claude Code + Bedrock.

- [ ] Similar structure to `test_claude_code_agent_bedrock.py`
- [ ] Configure Claude Code with Bedrock (us.anthropic.claude-opus-4-5-20251101-v1:0)
- [ ] Configure gated MCP server with `MCP_PRESET=ideation`
- [ ] Use `get_allowed_tools_for_preset("ideation", "gated-knowledge")` for allowed_tools
- [ ] Simple task: "Search for principles about LoRA fine-tuning"
- [ ] Verify agent can call `wiki_idea_search` tool
- [ ] Print search results
- [ ] Skip if `KAPSO_RUN_BEDROCK_TESTS != 1` or no credentials
- [ ] Runnable standalone: `python tests/test_gated_mcp_idea_search.py`

**Example structure:**

```python
"""
Test gated MCP idea search with Claude Code + Bedrock.

Run: python tests/test_gated_mcp_idea_search.py
"""
import os
import tempfile
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.execution.coding_agents.base import CodingAgentConfig
from src.execution.coding_agents.adapters.claude_code_agent import ClaudeCodeCodingAgent
from src.knowledge.gated_mcp.presets import get_allowed_tools_for_preset

def run_idea_search_test() -> bool:
    project_root = Path(__file__).parent.parent
    workspace = tempfile.mkdtemp(prefix="gated_mcp_test_")
    
    # MCP server config
    mcp_servers = {
        "gated-knowledge": {
            "command": "python",
            "args": ["-m", "src.knowledge.gated_mcp.server"],
            "cwd": str(project_root),
            "env": {
                "PYTHONPATH": str(project_root),
                "MCP_PRESET": "ideation",
                "KG_INDEX_PATH": os.environ.get("KG_INDEX_PATH", ""),
            },
        }
    }
    
    config = CodingAgentConfig(
        agent_type="claude_code",
        model="us.anthropic.claude-opus-4-5-20251101-v1:0",
        agent_specific={
            "use_bedrock": True,
            "aws_region": os.environ.get("AWS_REGION", "us-east-1"),
            "mcp_servers": mcp_servers,
            "allowed_tools": get_allowed_tools_for_preset("ideation", "gated-knowledge"),
            "timeout": 120,
        }
    )
    
    agent = ClaudeCodeCodingAgent(config)
    agent.initialize(workspace)
    
    result = agent.generate_code(
        "Search for principles about LoRA fine-tuning using the wiki_idea_search tool. "
        "Report what you find."
    )
    
    agent.cleanup()
    shutil.rmtree(workspace, ignore_errors=True)
    
    return result.success

if __name__ == "__main__":
    success = run_idea_search_test()
    exit(0 if success else 1)
```

**Acceptance Criteria:**
- [ ] Script runs standalone
- [ ] Properly skips when credentials missing
- [ ] Claude Code successfully calls `wiki_idea_search`
- [ ] Results printed to stdout

---

### Task 4.7: Standalone Claude Code Integration Test - Code Search

**Priority:** Medium  
**Estimated Complexity:** Medium  
**Dependencies:** Phase 3

Create `tests/test_gated_mcp_code_search.py`:

Standalone script to test code search gate with Claude Code + Bedrock.

- [ ] Similar structure to Task 4.6
- [ ] Configure gated MCP server with `MCP_PRESET=implementation`
- [ ] Use `get_allowed_tools_for_preset("implementation", "gated-knowledge")` for allowed_tools
- [ ] Simple task: "Search for implementation details about QLoRA training"
- [ ] Verify agent can call `wiki_code_search` tool
- [ ] Print search results
- [ ] Runnable standalone: `python tests/test_gated_mcp_code_search.py`

**Acceptance Criteria:**
- [ ] Script runs standalone
- [ ] Properly skips when credentials missing
- [ ] Claude Code successfully calls `wiki_code_search`
- [ ] Results printed to stdout

---

### Task 4.8: Standalone Claude Code Integration Test - Research

**Priority:** Medium  
**Estimated Complexity:** Medium  
**Dependencies:** Phase 3

Create `tests/test_gated_mcp_research.py`:

Standalone script to test research gate with Claude Code + Bedrock.

- [ ] Similar structure to Task 4.6
- [ ] Configure gated MCP server with `MCP_PRESET=ideation` (includes research)
- [ ] Simple task: "Research recent advances in LoRA fine-tuning from the web"
- [ ] Verify agent can call `research_idea` tool
- [ ] Print research results
- [ ] Runnable standalone: `python tests/test_gated_mcp_research.py`

**Acceptance Criteria:**
- [ ] Script runs standalone
- [ ] Properly skips when credentials missing
- [ ] Claude Code successfully calls `research_idea`
- [ ] Results printed to stdout

---

### Task 4.9: Standalone Claude Code Integration Test - Full Preset

**Priority:** Low  
**Estimated Complexity:** Medium  
**Dependencies:** Phase 3

Create `tests/test_gated_mcp_full.py`:

Standalone script to test full preset with all gates enabled.

- [ ] Configure gated MCP server with `MCP_PRESET=full`
- [ ] Task that uses multiple tools: idea search, code search, research
- [ ] Verify all tools accessible
- [ ] Runnable standalone: `python tests/test_gated_mcp_full.py`

**Acceptance Criteria:**
- [ ] Script runs standalone
- [ ] All gates' tools accessible
- [ ] Multi-tool task completes successfully

---

## Phase 5: Integration with Existing Code

### Task 5.1: Update KnowledgeMerger

**Priority:** High  
**Estimated Complexity:** Low  
**Dependencies:** Phase 3

Update `src/knowledge/learners/merger/knowledge_merger.py`:

- [ ] Import `get_allowed_tools_for_preset`
- [ ] Use `MCP_PRESET=merger` instead of manual gate config
- [ ] Use `get_allowed_tools_for_preset()` for `allowed_tools`
- [ ] Update MCP server path to `src.knowledge.gated_mcp.server`

---

### Task 5.2: Deprecate Old MCP Server (Optional)

**Priority:** Low  
**Estimated Complexity:** Low  
**Dependencies:** Task 5.1

Consider deprecating `src/knowledge/wiki_mcps/mcp_server.py`:

- [ ] Add deprecation warning
- [ ] Update any other references
- [ ] Document migration path

---

## Summary

| Phase | Tasks | Priority |
|-------|-------|----------|
| Phase 1: Core Infrastructure | 4 tasks | High |
| Phase 2: Gate Implementations | 5 tasks | High |
| Phase 3: Server Implementation | 2 tasks | High |
| Phase 4: Documentation & Testing | 9 tasks | Medium |
| Phase 5: Integration | 2 tasks | High/Low |

**Total: 22 tasks**

### Recommended Implementation Order

1. Task 1.1 (directory structure)
2. Task 1.2 (presets)
3. Task 1.3 (backends)
4. Task 1.4 (base gate)
5. Tasks 2.1-2.4 (gate implementations) - can be parallelized
6. Task 2.5 (gates init)
7. Task 3.1 (server)
8. Task 3.2 (init)
9. Task 5.1 (update KnowledgeMerger)
10. Tasks 4.1-4.5 (docs and unit tests)
11. Tasks 4.6-4.9 (standalone Claude Code integration tests)
