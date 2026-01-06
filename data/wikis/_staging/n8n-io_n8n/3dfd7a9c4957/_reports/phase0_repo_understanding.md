# Phase 0: Repository Understanding Report

## Summary
- Files explored: 60/60
- Completion: 100%

## Key Discoveries

### Main Entry Points Found
- `packages/@n8n/task-runner-python/src/main.py` - Primary entry point for Python task runner service
- `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/__main__.py` - Entry point for workflow comparison CLI

### Core Modules Identified

#### Python Task Runner (`task-runner-python`)
A sophisticated system for executing Python code tasks in a distributed, secure manner:

1. **Core Execution**
   - `task_runner.py` - Main orchestrator managing WebSocket connections, task lifecycle, and executor coordination
   - `task_executor.py` - Sandboxed code executor with restricted builtins and controlled imports
   - `task_analyzer.py` - AST-based code security analysis and validation

2. **Security Layer**
   - `constants.py` - Security policy definitions (allowed/blocked modules)
   - `import_validation.py` - Module import security enforcement
   - `security_config.py` - Security settings configuration

3. **Communication**
   - `message_serde.py` - JSON message serialization/deserialization
   - `pipe_reader.py` - Subprocess pipe communication
   - `message_types/` - Protocol message definitions (broker, runner, pipe)

4. **Infrastructure**
   - `health_check_server.py` - HTTP health endpoint for Kubernetes
   - `sentry.py` - Error tracking integration
   - `shutdown.py` - Graceful shutdown handling

#### AI Workflow Builder Evaluations (`ai-workflow-builder.ee`)
A workflow comparison tool for evaluating AI-generated workflows:

1. **Core Comparison**
   - `graph_builder.py` - Converts n8n workflows to NetworkX graphs
   - `similarity.py` - Graph edit distance calculation
   - `cost_functions.py` - Edit operation cost calculation

2. **Configuration**
   - `config_loader.py` - Flexible comparison rules management

3. **CLI Interface**
   - `compare_workflows.py` - Command-line tool for workflow comparison

### Architecture Patterns Observed

1. **Broker-Runner Pattern**: Task runner uses WebSocket connection to a broker for task dispatch
2. **Subprocess Isolation**: Code execution happens in isolated subprocesses with restricted capabilities
3. **Dataclass Protocol**: Heavy use of Python dataclasses for type-safe message definitions
4. **AST Security**: Pre-execution code analysis using Python AST to detect blocked imports
5. **Graph-Based Comparison**: Workflow evaluation uses NetworkX for sophisticated similarity metrics

### Error Handling Strategy
- 14 custom exception classes covering:
  - Configuration errors
  - Security violations
  - Task lifecycle (cancelled, killed, timeout)
  - Communication errors (pipe, WebSocket)
  - Result handling (missing, read errors)

## File Organization

| Category | Count | Description |
|----------|-------|-------------|
| AI Workflow Evaluation | 10 | Workflow comparison and similarity scoring |
| Task Runner Config | 4 | Configuration dataclasses |
| Task Runner Errors | 15 | Custom exception classes |
| Task Runner Core | 19 | Main execution and communication |
| Test Files | 12 | Unit and integration tests |

## Recommendations for Next Phase

### Suggested Workflows to Document
1. **Task Execution Flow**: How a task goes from broker → runner → executor → result
2. **Security Validation Pipeline**: AST analysis → import validation → sandbox setup
3. **Workflow Comparison Pipeline**: JSON → Graph → Edit Distance → Similarity Score

### Key APIs to Trace
1. `TaskRunner.run()` - Main task processing loop
2. `TaskExecutor.execute()` - Code execution in sandbox
3. `calculate_graph_edit_distance()` - Workflow similarity calculation

### Important Files for Anchoring Phase
1. `task_runner.py` - Central orchestrator (501 lines)
2. `task_executor.py` - Core execution engine (506 lines)
3. `similarity.py` - Primary evaluation logic (501 lines)
4. `constants.py` - Security definitions (193 lines)

## Technical Notes

- The task runner supports two execution modes: "all items" and "per item"
- Health checks work during task execution (non-blocking)
- Sentry integration is optional and configurable
- The workflow comparison uses configurable cost functions for nuanced scoring
- Test infrastructure includes a full local broker simulation
