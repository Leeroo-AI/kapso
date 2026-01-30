# Task: Migrate KnowledgeMerger from wiki_mcps to gated_mcp - COMPLETED

## Summary

Successfully migrated from `src/knowledge/wiki_mcps` to `src/knowledge/gated_mcp` and removed the old module.

## Changes Made

### 1. Updated `knowledge_merger.py`
- Changed MCP server from `src.knowledge.wiki_mcps.mcp_server` to `src.knowledge.gated_mcp.server`
- Added `MCP_ENABLED_GATES=kg` environment variable
- Removed `KG_SEARCH_BACKEND` env var (not needed for gated_mcp)

### 2. Updated `tests/test_mcp_integration.py`
- Changed MCP server reference to `src.knowledge.gated_mcp.server`
- Updated env vars to use `MCP_ENABLED_GATES=kg`

### 3. Updated `tests/test_get_page_structure.py`
- Changed from importing `_handle_get_page_structure` from wiki_mcps
- Now tests via `KGGate.handle_call()` from gated_mcp

### 4. Updated `claude_code_agent.py`
- Updated docstring example to use `src.knowledge.gated_mcp.server`
- Updated env var example to show `MCP_ENABLED_GATES`

### 5. Removed `src/knowledge/wiki_mcps/`
- Deleted entire directory (was: `__init__.py`, `mcp_server.py`, `mcp_config_template.json`, `README.md`)

## Verification

All tests passed:
- `test_get_page_structure.py` - All page types work correctly
- `KnowledgeMerger` import works
- `gated_mcp.server` import works

## Notes

- The gated_mcp KGGate provides all the same tools as wiki_mcps:
  - `search_knowledge` ✓
  - `get_wiki_page` ✓
  - `kg_index` ✓
  - `kg_edit` ✓
  - `get_page_structure` ✓
- `list_page_types` and `search_with_context` were not used by KnowledgeMerger
