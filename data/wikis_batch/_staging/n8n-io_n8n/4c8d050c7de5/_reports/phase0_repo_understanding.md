# Phase 0: Repository Understanding Report

## Summary
- Files explored: 60/60
- Completion: 100%

## Repository Overview

The n8n repository's Python codebase consists of two main packages:

1. **AI Workflow Builder Evaluations** (10 files, ~2,462 lines)
   - Path: `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/`
   - Purpose: Programmatic evaluation of AI-generated workflows using graph edit distance

2. **Python Task Runner** (50 files, ~4,636 lines)
   - Path: `packages/@n8n/task-runner-python/`
   - Purpose: Isolated Python code execution engine for n8n workflows

## Key Discoveries

### AI Workflow Builder Evaluations

**Architecture:**
- Uses **NetworkX** for graph algorithms and **graph edit distance** to compare AI-generated workflows against ground truth
- Configurable cost functions with presets (strict/standard/lenient)
- CLI tool for batch workflow comparisons

**Core Components:**
- `graph_builder.py` - Converts n8n workflow JSON to directed graphs
- `similarity.py` - Implements optimized graph edit distance algorithm
- `cost_functions.py` - Domain-specific edit operation costs
- `config_loader.py` - YAML-based configuration with presets

**Key Patterns:**
- Trigger nodes have highest priority (critical changes)
- Expression normalization for `$fromAI()` calls
- Name hash matching prevents inappropriate node swapping

### Python Task Runner

**Architecture:**
- **Security-first design** with defense-in-depth:
  - Static AST analysis (`task_analyzer.py`)
  - Runtime import validation (`import_validation.py`)
  - Subprocess isolation with forkserver context
  - Configurable allowlists/denylists

**Core Components:**
- `main.py` - Application entry point
- `task_runner.py` - WebSocket-based task orchestrator
- `task_executor.py` - Isolated subprocess code execution
- `task_state.py` - Task lifecycle management

**Communication Patterns:**
- WebSocket to broker for task coordination
- Length-prefixed IPC pipes for subprocess results
- RPC callbacks for mid-execution operations (print capture)

**Configuration:**
- All settings via N8N_RUNNERS_* environment variables
- Support for Docker secrets (VAR_FILE pattern)
- Configurable security policies, timeouts, Sentry integration

**Error Hierarchy (14 types):**
- Configuration errors
- Security violations
- Task lifecycle errors (cancelled/killed/timeout)
- IPC errors (pipe content/length)
- Result handling errors

## Main Entry Points

| File | Purpose |
|------|---------|
| `task-runner-python/src/main.py` | Task runner application entry |
| `ai-workflow-builder.ee/.../src/__main__.py` | Evaluation CLI entry |
| `task-runner-python/src/task_runner.py` | Core orchestration class |
| `ai-workflow-builder.ee/.../src/compare_workflows.py` | Workflow comparison CLI |

## Core Modules Identified

### Task Runner Core
| Module | Purpose |
|--------|---------|
| `task_runner.py` | WebSocket connection, task distribution |
| `task_executor.py` | Subprocess management, code wrapping |
| `task_analyzer.py` | Static security analysis via AST |
| `shutdown.py` | Graceful termination coordination |

### Task Runner Configuration
| Module | Purpose |
|--------|---------|
| `config/task_runner_config.py` | Central configuration hub |
| `config/security_config.py` | Sandbox security settings |
| `constants.py` | Default values and magic strings |

### AI Workflow Builder Core
| Module | Purpose |
|--------|---------|
| `similarity.py` | GED algorithm implementation |
| `graph_builder.py` | Workflow to graph conversion |
| `cost_functions.py` | Edit operation cost calculations |

## Architecture Patterns Observed

1. **Dataclass-based Configuration** - All configs use Python dataclasses with `from_env()` factory methods

2. **Type-safe Message Protocol** - Union types with literal discriminators for broker/runner/pipe messages

3. **Offer-based Task Distribution** - Runners advertise capacity; broker selects winners for load balancing

4. **Background Thread Pattern** - PipeReader runs in background thread to avoid blocking WebSocket

5. **LRU Caching** - 500-entry cache for validation results in TaskAnalyzer

6. **Continue-on-Fail** - Errors converted to result items with error field for workflow resilience

## Test Coverage Summary

**Unit Tests:**
- Environment variable parsing and Docker secrets
- Sentry error filtering logic
- Static security analysis (AST-based)
- IPC pipe reading mechanics

**Integration Tests:**
- End-to-end code execution (all_items, per_item modes)
- RPC message handling
- Health check endpoint
- WebSocket connection resilience

## Recommendations for Next Phase

### Suggested Workflows to Document

1. **Task Execution Flow**
   - WebSocket connection → Task offer → Task acceptance → Subprocess spawn → Result collection → Completion message

2. **Security Validation Flow**
   - Code received → AST analysis → Import validation → Runtime builtin filtering → Isolated execution

3. **Workflow Comparison Flow**
   - Load workflows → Build graphs → Compute GED → Calculate similarity → Generate report

### Key APIs to Trace

1. `TaskRunner.start()` - Main orchestration loop
2. `TaskExecutor.execute()` - Subprocess code execution
3. `TaskAnalyzer.validate()` - Static security analysis
4. `similarity.calculate_similarity()` - GED computation

### Important Files for Anchoring Phase

1. `task_runner.py` - Central to understanding task distribution
2. `task_executor.py` - Central to understanding code isolation
3. `similarity.py` - Central to understanding workflow evaluation
4. `config_loader.py` - Central to understanding comparison configuration

## Statistics

| Category | Files | Lines |
|----------|-------|-------|
| AI Workflow Builder | 10 | ~2,462 |
| Task Runner Core | 15 | ~1,851 |
| Task Runner Config | 4 | ~206 |
| Task Runner Messages | 4 | ~216 |
| Task Runner Errors | 15 | ~175 |
| Task Runner Tests | 12 | ~2,188 |
| **Total** | **60** | **7,098** |
