# Task: Migrate KnowledgeMerger from wiki_mcps to gated_mcp

## Overview

The `KnowledgeMerger` in `src/knowledge/learners/merger/knowledge_merger.py` currently uses `src/knowledge/wiki_mcps/mcp_server.py` for MCP tools. We want to migrate it to use `src/knowledge/gated_mcp/server.py` instead, then remove `wiki_mcps`.

## Current State

### wiki_mcps (to be removed)
- Location: `src/knowledge/wiki_mcps/`
- Files: `__init__.py`, `mcp_server.py`, `mcp_config_template.json`, `README.md`
- Tools provided:
  - `search_knowledge`
  - `get_wiki_page`
  - `list_page_types`
  - `search_with_context`
  - `kg_index`
  - `kg_edit`
  - `get_page_structure`

### gated_mcp (replacement)
- Location: `src/knowledge/gated_mcp/`
- Server: `src/knowledge/gated_mcp/server.py`
- KG Gate (`kg_gate.py`) provides:
  - `search_knowledge` ✓
  - `get_wiki_page` ✓
  - `kg_index` ✓
  - `kg_edit` ✓
  - `get_page_structure` ✓

### Missing in gated_mcp
- `list_page_types` - Not in KGGate (minor, can be added or skipped)
- `search_with_context` - Not in KGGate (minor, can be added or skipped)

## Changes Required

### 1. Update KnowledgeMerger (`knowledge_merger.py`)

**Current code (lines 393-400):**
```python
mcp_servers = {
    "kg-graph-search": {
        "command": "python",
        "args": ["-m", "src.knowledge.wiki_mcps.mcp_server"],
        "cwd": str(project_root),
        "env": mcp_env,
    }
}
```

**New code:**
```python
mcp_servers = {
    "kg-graph-search": {
        "command": "python",
        "args": ["-m", "src.knowledge.gated_mcp.server"],
        "cwd": str(project_root),
        "env": {
            **mcp_env,
            "MCP_ENABLED_GATES": "kg",  # Only enable KG gate for merger
        },
    }
}
```

**Also update allowed_tools (lines 403-410):**
The tool names stay the same, so no changes needed to `allowed_tools`.

### 2. Update Environment Variables

The gated_mcp server uses different env vars:
- `MCP_ENABLED_GATES` - Comma-separated gate names (e.g., "kg")
- `KG_INDEX_PATH` - Same as wiki_mcps ✓

### 3. Update Tests

**File: `tests/test_mcp_integration.py` (line 68)**
```python
# Old
"args": ["-m", "src.knowledge.wiki_mcps.mcp_server"],

# New
"args": ["-m", "src.knowledge.gated_mcp.server"],
```

**File: `tests/test_get_page_structure.py` (line 16)**
```python
# Old
from src.knowledge.wiki_mcps.mcp_server import _handle_get_page_structure

# New - Need to test via KGGate instead
from src.knowledge.gated_mcp.gates.kg_gate import KGGate
```

### 4. Remove wiki_mcps

After migration is complete:
```bash
rm -rf src/knowledge/wiki_mcps/
```

### 5. Update Documentation

Update any references in:
- `src/execution/coding_agents/adapters/claude_code_agent.py` (docstring example only)

## Implementation Steps

1. [ ] Update `knowledge_merger.py` to use gated_mcp server
2. [ ] Update `tests/test_mcp_integration.py`
3. [ ] Update `tests/test_get_page_structure.py` to test via KGGate
4. [ ] Update docstring example in `claude_code_agent.py`
5. [ ] Verify all tests pass
6. [ ] Remove `src/knowledge/wiki_mcps/` directory
7. [ ] Update any remaining references

## Verification

After changes:
```bash
# Test KnowledgeMerger
python -c "from src.knowledge.learners.merger import KnowledgeMerger; print('OK')"

# Test gated_mcp server
python -m src.knowledge.gated_mcp.server --help

# Run tests
pytest tests/test_mcp_integration.py -v
pytest tests/test_get_page_structure.py -v
```

## Notes

- The gated_mcp server is more flexible (supports multiple gates)
- For KnowledgeMerger, we only need the `kg` gate
- Tool names are identical, so no prompt changes needed
- The `list_page_types` and `search_with_context` tools are not used by KnowledgeMerger
