# Phase 1a: Anchoring Report

## Summary
- Workflows created: 3
- Total steps documented: 25

## Workflows Created

| Workflow | Source Files | Steps | Rough APIs |
|----------|--------------|-------|------------|
| Python_Task_Execution | main.py, task_runner.py, task_executor.py, task_analyzer.py | 8 | TaskRunner.start, TaskExecutor.create_process, TaskAnalyzer.validate |
| Security_Validation_Pipeline | task_analyzer.py, task_executor.py, import_validation.py, constants.py | 8 | TaskAnalyzer.validate, SecurityValidator, validate_module_import |
| AI_Workflow_Comparison | compare_workflows.py, similarity.py, graph_builder.py, config_loader.py | 9 | load_config, build_workflow_graph, calculate_graph_edit_distance |

## Coverage Summary
- Source files covered: 48 out of 60 (80%)
- Test files: Not covered (12 files) - appropriate for Workflow pages
- Example files documented: N/A (this repo has infrastructure code, not examples)

## Source Files Identified Per Workflow

### n8n-io_n8n_Python_Task_Execution

**Core Files:**
- `packages/@n8n/task-runner-python/src/main.py` - Application entry point (72 lines)
- `packages/@n8n/task-runner-python/src/task_runner.py` - Main task orchestrator (501 lines)
- `packages/@n8n/task-runner-python/src/task_executor.py` - Sandboxed code executor (506 lines)
- `packages/@n8n/task-runner-python/src/task_analyzer.py` - Code security analysis (212 lines)

**Supporting Files:**
- `packages/@n8n/task-runner-python/src/message_serde.py` - Message serialization/deserialization
- `packages/@n8n/task-runner-python/src/pipe_reader.py` - Subprocess pipe reader
- `packages/@n8n/task-runner-python/src/task_state.py` - Task lifecycle state
- `packages/@n8n/task-runner-python/src/nanoid.py` - Unique ID generation
- `packages/@n8n/task-runner-python/src/shutdown.py` - Graceful shutdown handling
- `packages/@n8n/task-runner-python/src/health_check_server.py` - HTTP health endpoint
- `packages/@n8n/task-runner-python/src/sentry.py` - Error tracking integration

**Configuration:**
- `packages/@n8n/task-runner-python/src/config/task_runner_config.py` - Main config
- `packages/@n8n/task-runner-python/src/config/health_check_config.py` - Health check config
- `packages/@n8n/task-runner-python/src/config/sentry_config.py` - Sentry config

**Message Types:**
- `packages/@n8n/task-runner-python/src/message_types/broker.py` - Broker messages
- `packages/@n8n/task-runner-python/src/message_types/runner.py` - Runner messages
- `packages/@n8n/task-runner-python/src/message_types/pipe.py` - Pipe messages

**Error Classes:**
- 14 error class files in `packages/@n8n/task-runner-python/src/errors/`

### n8n-io_n8n_Security_Validation_Pipeline

**Core Files:**
- `packages/@n8n/task-runner-python/src/task_analyzer.py` - AST-based security validator (212 lines)
- `packages/@n8n/task-runner-python/src/task_executor.py` - Runtime sandbox setup (506 lines)
- `packages/@n8n/task-runner-python/src/import_validation.py` - Module import validation (37 lines)
- `packages/@n8n/task-runner-python/src/constants.py` - Security policy definitions (193 lines)

**Configuration:**
- `packages/@n8n/task-runner-python/src/config/security_config.py` - Security settings

**Error Classes:**
- `packages/@n8n/task-runner-python/src/errors/security_violation_error.py`

### n8n-io_n8n_AI_Workflow_Comparison

**Core Files:**
- `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/compare_workflows.py` - CLI tool (334 lines)
- `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/similarity.py` - GED calculation (501 lines)
- `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/graph_builder.py` - Workflow to graph (222 lines)
- `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/config_loader.py` - Config management (389 lines)
- `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/cost_functions.py` - Cost calculation (497 lines)

**Entry Points:**
- `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/__main__.py`
- `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/__init__.py`

## Notes for Phase 1b (Enrichment)

### Files Needing Line-by-Line Tracing

1. **task_runner.py** (501 lines)
   - Complex async state machine with multiple message handlers
   - Need to trace: `_execute_task`, `_handle_task_offer_accept`, `_send_offers`
   - External dependency: `websockets` library

2. **task_executor.py** (506 lines)
   - Core sandbox implementation with security hooks
   - Need to trace: `_all_items`, `_per_item`, `_filter_builtins`, `_sanitize_sys_modules`
   - Uses forkserver multiprocessing context

3. **similarity.py** (501 lines)
   - Graph edit distance algorithm implementation
   - Need to trace: `calculate_graph_edit_distance`, `_extract_operations_from_path`
   - External dependency: `networkx` library

4. **config_loader.py** (389 lines)
   - Complex dataclass hierarchy for configuration
   - Need to trace: `WorkflowComparisonConfig._from_dict`, rule matching logic

### External APIs to Document

1. **websockets** - WebSocket client for broker communication
   - Used in: TaskRunner.start()
   - Key methods: connect(), send(), receive()

2. **networkx** - Graph algorithms for workflow comparison
   - Used in: similarity.py, graph_builder.py
   - Key methods: DiGraph(), optimize_edit_paths(), relabel_nodes()

3. **multiprocessing** - Process isolation for code execution
   - Used in: TaskExecutor
   - Context: forkserver
   - Key: Pipe(), Process()

4. **ast** - Python AST parsing for security analysis
   - Used in: TaskAnalyzer, SecurityValidator
   - Key methods: parse(), NodeVisitor

### Unclear Mappings

1. **Overlap between Python_Task_Execution and Security_Validation_Pipeline**
   - Security validation is a sub-workflow within task execution
   - Consider whether to merge or keep separate (kept separate for clarity)
   - Files task_analyzer.py and task_executor.py appear in both workflows

2. **Error handling flow**
   - 14 error classes defined but not explicitly documented as a workflow step
   - Consider adding an Error Handling principle or heuristic

3. **Message protocol**
   - WebSocket message format spans multiple files (broker.py, runner.py, message_serde.py)
   - May warrant its own Principle page

## Architecture Insights

### Python Task Runner
- **Pattern:** Broker-Runner with WebSocket communication
- **Isolation:** Forkserver subprocess per task
- **Security:** Defense in depth (AST analysis + runtime hooks + sandbox)
- **Modes:** All-items (batch) and Per-item (streaming)

### AI Workflow Comparison
- **Pattern:** Graph-based similarity with configurable cost functions
- **Algorithm:** Optimal Graph Edit Distance via NetworkX
- **Configuration:** YAML/JSON with presets (strict/standard/lenient)
- **Output:** JSON or human-readable summary with PASS/FAIL

## Metrics

| Metric | Value |
|--------|-------|
| Total Workflow Steps | 25 |
| Unique Principles Referenced | 25 |
| Implementation APIs Identified | 25 |
| Source Files Covered | 48/60 (80%) |
| Lines of Code Covered | ~5,500/7,098 (77%) |
