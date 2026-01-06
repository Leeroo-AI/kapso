# Environment Index: n8n-io_n8n

> Tracks Environment pages and which pages require them.
> **Update IMMEDIATELY** after creating or modifying a Environment page.

## Summary

| Environment | Required By |
|-------------|-------------|
| Python_Task_Runner_Env | 8 implementations |
| Sandbox_Environment | 4 implementations |
| Workflow_Comparison_Env | 9 implementations |

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| n8n-io_n8n_Python_Task_Runner_Env | [→](./environments/n8n-io_n8n_Python_Task_Runner_Env.md) | ✅Impl:n8n-io_n8n_TaskRunner_init, ✅Impl:n8n-io_n8n_TaskRunner_start, ✅Impl:n8n-io_n8n_TaskRunner_send_offers, ✅Impl:n8n-io_n8n_TaskRunner_execute_task, ✅Impl:n8n-io_n8n_TaskAnalyzer_validate, ✅Impl:n8n-io_n8n_TaskExecutor_create_process, ✅Impl:n8n-io_n8n_SecurityConfig, ✅Impl:n8n-io_n8n_TaskAnalyzer_cache | Linux/macOS Python 3.13+ with websockets |
| n8n-io_n8n_Sandbox_Environment | [→](./environments/n8n-io_n8n_Sandbox_Environment.md) | ✅Impl:n8n-io_n8n_TaskExecutor_execute, ✅Impl:n8n-io_n8n_TaskExecutor_sandbox, ✅Impl:n8n-io_n8n_TaskExecutor_put_result, ✅Impl:n8n-io_n8n_validate_module_import | Isolated subprocess with filtered builtins |
| n8n-io_n8n_Workflow_Comparison_Env | [→](./environments/n8n-io_n8n_Workflow_Comparison_Env.md) | ✅Impl:n8n-io_n8n_load_config, ✅Impl:n8n-io_n8n_load_workflow, ✅Impl:n8n-io_n8n_build_workflow_graph, ✅Impl:n8n-io_n8n_relabel_graph_by_structure, ✅Impl:n8n-io_n8n_calculate_graph_edit_distance, ✅Impl:n8n-io_n8n_extract_operations_from_path, ✅Impl:n8n-io_n8n_calculate_max_cost, ✅Impl:n8n-io_n8n_determine_priority, ✅Impl:n8n-io_n8n_format_output | Python 3.11+ with NetworkX, NumPy, SciPy |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
