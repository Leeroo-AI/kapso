# Phase 3: Enrichment Report

## Repository Information

| Field | Value |
|-------|-------|
| **Repository** | n8n-io_n8n |
| **Source** | https://github.com/n8n-io/n8n |
| **Wiki Directory** | `/home/ubuntu/praxium/data/wikis_batch2/_staging/n8n-io_n8n/3dfd7a9c4957` |
| **Execution Date** | 2024-12-18 |
| **Status** | ✅ Complete |

---

## Summary

| Metric | Count |
|--------|-------|
| **Environments Created** | 3 |
| **Heuristics Created** | 6 |
| **Total New Pages** | 9 |
| **Environment Links Added** | 21 |
| **Heuristic Links Added** | 18 |

---

## Environments Created

| Environment | File | Required By | Description |
|-------------|------|-------------|-------------|
| n8n-io_n8n_Python_Task_Runner_Env | [→](../environments/n8n-io_n8n_Python_Task_Runner_Env.md) | TaskRunner_init, TaskRunner_start, TaskRunner_send_offers, TaskRunner_execute_task, TaskAnalyzer_validate, TaskExecutor_create_process, SecurityConfig, TaskAnalyzer_cache | Linux/macOS Python 3.13+ with websockets for task execution |
| n8n-io_n8n_Sandbox_Environment | [→](../environments/n8n-io_n8n_Sandbox_Environment.md) | TaskExecutor_execute, TaskExecutor_sandbox, TaskExecutor_put_result, validate_module_import | Isolated subprocess with filtered builtins and sys.modules |
| n8n-io_n8n_Workflow_Comparison_Env | [→](../environments/n8n-io_n8n_Workflow_Comparison_Env.md) | load_config, load_workflow, build_workflow_graph, relabel_graph_by_structure, calculate_graph_edit_distance, extract_operations_from_path, calculate_max_cost, determine_priority, format_output | Python 3.11+ with NetworkX, NumPy, SciPy for GED |

### Key Environment Requirements Found

1. **Platform Restriction:** Python Task Runner explicitly blocks Windows (forkserver unavailable)
2. **Python Version:** Task Runner requires 3.13+; Workflow Comparison requires 3.11+
3. **Required Credential:** `N8N_RUNNERS_GRANT_TOKEN` is mandatory for task runner
4. **Security Configuration:** Multiple env vars control sandbox behavior (stdlib_allow, external_allow, builtins_deny)

---

## Heuristics Created

| Heuristic | File | Applies To | Description |
|-----------|------|------------|-------------|
| n8n-io_n8n_Validation_Caching_Strategy | [→](../heuristics/n8n-io_n8n_Validation_Caching_Strategy.md) | TaskAnalyzer_cache, TaskAnalyzer_validate, Security_Validation_Pipeline | LRU cache with SHA256 hash keys; 500 entry limit |
| n8n-io_n8n_Pipe_Timeout_Scaling | [→](../heuristics/n8n-io_n8n_Pipe_Timeout_Scaling.md) | TaskExecutor_put_result, TaskRunner_execute_task, Python_Task_Execution | Dynamic timeout: (payload * 0.1 / 100MB/s) + 2s |
| n8n-io_n8n_Print_Output_Truncation | [→](../heuristics/n8n-io_n8n_Print_Output_Truncation.md) | TaskExecutor_put_result, TaskExecutor_execute, Python_Task_Execution | Limit print() to 100 statements |
| n8n-io_n8n_Offer_Validity_Jitter | [→](../heuristics/n8n-io_n8n_Offer_Validity_Jitter.md) | TaskRunner_send_offers, Python_Task_Execution, Task_Offer_Negotiation | Random 0-500ms jitter prevents thundering herd |
| n8n-io_n8n_GED_Performance_Note | [→](../heuristics/n8n-io_n8n_GED_Performance_Note.md) | calculate_graph_edit_distance, AI_Workflow_Comparison, Graph_Edit_Distance | GED exponential but OK for small workflow graphs |
| n8n-io_n8n_Graceful_Shutdown_Timeout | [→](../heuristics/n8n-io_n8n_Graceful_Shutdown_Timeout.md) | TaskRunner_execute_task, Python_Task_Execution, Result_Delivery | Two-phase: wait 10s then SIGTERM/SIGKILL |

### Key Tribal Knowledge Captured

1. **Performance Optimization:** Validation caching avoids redundant AST parsing (500 entries max)
2. **Resource Protection:** Print output truncation prevents pipe overflow (100 statement limit)
3. **Distributed Systems:** Offer validity jitter prevents thundering herd (500ms random)
4. **Scaling:** Pipe timeout scales with payload size (formula: 10% of max / 100MB/s + 2s)
5. **Algorithm Note:** GED is exponential but acceptable for typical workflow graphs (< 50 nodes)
6. **Reliability:** Graceful shutdown with timeout prevents hangs (10s default, then force kill)

---

## Indexes Updated

| Index File | Status | Changes |
|------------|--------|---------|
| `_EnvironmentIndex.md` | ✅ Updated | Added 3 environment entries with connections |
| `_HeuristicIndex.md` | ✅ Updated | Added 6 heuristic entries with connections |

---

## Source Files Analyzed

### For Environments
- `packages/@n8n/task-runner-python/pyproject.toml` - Dependencies and Python version
- `packages/@n8n/task-runner-python/src/constants.py` - Environment variable names and defaults
- `packages/@n8n/task-runner-python/src/config/task_runner_config.py` - Configuration validation
- `packages/@n8n/task-runner-python/src/env.py` - Environment variable reading
- `packages/@n8n/task-runner-python/src/main.py` - Platform check (Windows blocked)
- `packages/@n8n/task-runner-python/src/task_executor.py` - Sandbox construction
- `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/pyproject.toml` - NetworkX dependencies

### For Heuristics
- `packages/@n8n/task-runner-python/src/task_analyzer.py` - LRU caching implementation
- `packages/@n8n/task-runner-python/src/task_executor.py` - Print truncation, pipe I/O
- `packages/@n8n/task-runner-python/src/task_runner.py` - Offer jitter, shutdown handling
- `packages/@n8n/task-runner-python/src/config/task_runner_config.py` - Timeout calculation
- `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/similarity.py` - GED performance note

---

## Notes for Audit Phase

### No Broken Links Expected
- All environment pages reference existing implementations
- All heuristic pages reference existing implementations, workflows, and principles

### Pages That May Need Review
- Environment pages contain extensive configuration documentation that could become outdated
- Heuristic pages reference specific line numbers that may shift with code changes

### Potential Future Enhancements
1. Link environment pages from implementation pages (bidirectional)
2. Add `uses_heuristic` links to workflow pages
3. Consider extracting more heuristics from comments in non-Python files (TypeScript backend)

---

## Files Created

```
environments/
├── n8n-io_n8n_Python_Task_Runner_Env.md
├── n8n-io_n8n_Sandbox_Environment.md
└── n8n-io_n8n_Workflow_Comparison_Env.md

heuristics/
├── n8n-io_n8n_Validation_Caching_Strategy.md
├── n8n-io_n8n_Pipe_Timeout_Scaling.md
├── n8n-io_n8n_Print_Output_Truncation.md
├── n8n-io_n8n_Offer_Validity_Jitter.md
├── n8n-io_n8n_GED_Performance_Note.md
└── n8n-io_n8n_Graceful_Shutdown_Timeout.md
```

---

**Phase 3 Enrichment Complete.** The wiki now includes environment requirements and tribal knowledge extracted from the n8n Python codebase.
