# Task: Convert repo_memory CLI to MCP Gate

## Status: COMPLETED

## Goal

Replace the CLI-based repo_memory access with an MCP gate to solve path issues when running from different locations.

## Summary of Changes

### Phase 1: Create RepoMemoryGate
- [x] Created `src/gated_mcp/gates/repo_memory_gate.py`
  - Implemented `RepoMemoryGate` class extending `ToolGate`
  - Tools exposed:
    - `get_repo_memory_section(section_id, repo_root?)` - Get specific section content (no truncation)
    - `list_repo_memory_sections(repo_root?)` - List available section IDs
    - `get_repo_memory_summary(repo_root?)` - Get summary + TOC
  - Uses lazy imports for `RepoMemoryManager`
  - `repo_root` resolution: params > env var `REPO_MEMORY_ROOT` > CWD

### Phase 2: Register the Gate
- [x] Updated `src/gated_mcp/gates/__init__.py` - Import and export `RepoMemoryGate`
- [x] Updated `src/gated_mcp/server.py` - Added `RepoMemoryGate` to `GATE_CLASSES`
- [x] Updated `src/gated_mcp/presets.py` - Added `repo_memory` gate definition and `repo_root` param to `get_mcp_config`

### Phase 3: Update Search Strategies
- [x] Updated `src/execution/search_strategies/generic/strategy.py`
  - Added `repo_memory` to default ideation gates
  - Added `repo_memory` to default implementation gates
  - Pass `repo_root` parameter to MCP config
  - Removed Bash from ideation allowed tools (no longer needed for repo_memory)
- [x] Updated `src/execution/search_strategies/benchmark_tree_search.py`
  - Updated repo_memory access instructions to use MCP tools

### Phase 4: Update Prompts
- [x] Updated `src/execution/search_strategies/generic/prompts/ideation_claude_code.md`
  - Replaced CLI instructions with MCP tool usage
  - Removed Bash from available tools
- [x] Updated `src/execution/search_strategies/generic/prompts/implementation_claude_code.md`
  - Same changes as ideation prompt

### Phase 5: Update Observability
- [x] Updated `src/execution/memories/repo_memory/observation.py`
  - Updated `extract_repo_memory_sections_consulted()` to detect MCP tool calls
  - Added patterns for `get_repo_memory_section` calls

### Phase 6: Cleanup
- [x] Updated `docs/evolve/repo-memory.mdx` - Updated CLI references to MCP tools
- [x] Updated `archive/basic_linear_search.py` - Updated references

### Phase 7: Testing
- [x] Tested MCP gate directly - All tools work correctly
- [x] Tested integration - Imports work, gate is registered

### Phase 8: Remove CLI
- [x] Deleted `src/execution/memories/repo_memory/cli.py`
- [x] Replaced `tests/test_repo_memory_cli_standalone.py` with `tests/test_repo_memory_gate.py`

## Notes

- No truncation on section content (user requirement)
- CLI deleted - no backward compatibility needed
- `repo_root` resolution order: params > env var > CWD
