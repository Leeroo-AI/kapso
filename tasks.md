# Experiment Memory Migration - COMPLETED

## Goal
Replace the context_manager module with a new experiment_memory module that uses MCP tools for experiment history access.

## Status: COMPLETED

All changes have been applied successfully.

## New Architecture

### Experiment History Access
- **Before**: Context managers injected experiment history as formatted strings into prompts
- **After**: Agents query experiment history dynamically via MCP tools

### New Module: `src/experiment_memory/`
- `__init__.py`: Module exports
- `store.py`: ExperimentHistoryStore with dual storage:
  - JSON file for basic retrieval (top, recent)
  - Weaviate for semantic search (optional)

### New MCP Gate: `experiment_history`
- `src/knowledge/gated_mcp/gates/experiment_history_gate.py`: Gate implementation
- Tools:
  - `get_top_experiments`: Get best experiments by score
  - `get_recent_experiments`: Get most recent experiments
  - `search_similar_experiments`: Semantic search for similar experiments

## Files Changed

### Created
- `src/experiment_memory/__init__.py`
- `src/experiment_memory/store.py`
- `src/execution/types.py` (moved from context_manager/types.py)
- `src/knowledge/gated_mcp/gates/experiment_history_gate.py`

### Updated
- `src/execution/orchestrator.py`: Uses ExperimentHistoryStore, removed context_manager
- `src/execution/search_strategies/basic_linear_search.py`: Uses experiment_history MCP gate
- `src/execution/prompts/ideation_claude_code.md`: Updated to use MCP tools for history
- `src/knowledge/gated_mcp/presets.py`: Added experiment_history gate
- `src/knowledge/gated_mcp/server.py`: Registered ExperimentHistoryGate
- `src/knowledge/gated_mcp/gates/__init__.py`: Added ExperimentHistoryGate export
- `src/config.yaml`: Removed context_manager sections, added experiment_history gate
- `src/core/__init__.py`: Updated import path for ContextData
- `src/execution/search_strategies/*.py`: Updated import paths

### Archived (in `/archive/`)
- `__init__.py`
- `_template.py`
- `base.py`
- `basic_context_manager.py`
- `cognitive_context_manager.py`
- `context_manager.yaml`
- `factory.py`
- `kg_enriched_context_manager.py`
- `token_efficient_context_manager.py`
- `README.md`
- `test_basic_context_manager.py`

## How It Works

1. **Orchestrator** creates `ExperimentHistoryStore` at startup
2. After each experiment, orchestrator calls `store.add_experiment(node)`
3. Store persists to JSON and optionally indexes in Weaviate
4. **BasicLinearSearch** includes `experiment_history` gate in ideation
5. **Ideation agent** uses MCP tools to query past experiments:
   - `get_top_experiments(5)` - see what worked best
   - `get_recent_experiments(5)` - see recent attempts
   - `search_similar_experiments("my idea")` - check if similar was tried

## Configuration

### Environment Variables
- `EXPERIMENT_HISTORY_PATH`: Path to JSON file (default: `.kapso/experiment_history.json`)
- `WEAVIATE_URL`: Weaviate URL for semantic search (optional)

### Config.yaml
```yaml
search_strategy:
  type: "basic_linear_search"
  params:
    ideation_gates: ["idea", "code", "research", "experiment_history"]
```
