# Implementation Index: n8n-io_n8n

> Index of all Implementation pages and their connections to Principles and Environments.

## Python Task Execution Implementations

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| n8n-io_n8n_TaskRunner_start | [→](./implementations/n8n-io_n8n_TaskRunner_start.md) | ✅Principle:n8n-io_n8n_WebSocket_Connection, ✅Env:n8n-io_n8n_Python_Task_Runner | WebSocket connection with broker |
| n8n-io_n8n_TaskRunner_send_offers | [→](./implementations/n8n-io_n8n_TaskRunner_send_offers.md) | ✅Principle:n8n-io_n8n_Offer_Based_Distribution, ✅Env:n8n-io_n8n_Python_Task_Runner | Capacity-based offer generation |
| n8n-io_n8n_TaskRunner_handle_task_offer_accept | [→](./implementations/n8n-io_n8n_TaskRunner_handle_task_offer_accept.md) | ✅Principle:n8n-io_n8n_Task_Acceptance, ✅Env:n8n-io_n8n_Python_Task_Runner | Offer validation, task creation |
| n8n-io_n8n_TaskAnalyzer_validate | [→](./implementations/n8n-io_n8n_TaskAnalyzer_validate.md) | ✅Principle:n8n-io_n8n_Static_Security_Analysis, ✅Env:n8n-io_n8n_Python_Task_Runner | AST-based code validation |
| n8n-io_n8n_TaskExecutor_create_process | [→](./implementations/n8n-io_n8n_TaskExecutor_create_process.md) | ✅Principle:n8n-io_n8n_Subprocess_Isolation, ✅Env:n8n-io_n8n_Python_Task_Runner | Forkserver process creation |
| n8n-io_n8n_TaskExecutor_all_items | [→](./implementations/n8n-io_n8n_TaskExecutor_all_items.md) | ✅Principle:n8n-io_n8n_Code_Execution, ✅Env:n8n-io_n8n_Python_Task_Runner | All-items and per-item modes |
| n8n-io_n8n_TaskExecutor_execute_process | [→](./implementations/n8n-io_n8n_TaskExecutor_execute_process.md) | ✅Principle:n8n-io_n8n_Result_Collection, ✅Env:n8n-io_n8n_Python_Task_Runner | PipeReader, timeout handling |
| n8n-io_n8n_TaskRunner_execute_task | [→](./implementations/n8n-io_n8n_TaskRunner_execute_task.md) | ✅Principle:n8n-io_n8n_Task_Completion, ✅Env:n8n-io_n8n_Python_Task_Runner | Full task lifecycle |

## Security Validation Implementations

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| n8n-io_n8n_TaskRunner_handle_task_settings | [→](./implementations/n8n-io_n8n_TaskRunner_handle_task_settings.md) | ✅Principle:n8n-io_n8n_Task_Settings_Reception, ✅Env:n8n-io_n8n_Python_Task_Runner | Settings extraction |
| n8n-io_n8n_TaskAnalyzer_cache | [→](./implementations/n8n-io_n8n_TaskAnalyzer_cache.md) | ✅Principle:n8n-io_n8n_Validation_Caching, ✅Env:n8n-io_n8n_Python_Task_Runner | LRU validation cache |
| n8n-io_n8n_ast_parse | [→](./implementations/n8n-io_n8n_ast_parse.md) | ✅Principle:n8n-io_n8n_AST_Parsing, ✅Env:n8n-io_n8n_Python_Task_Runner | External: Python stdlib ast |
| n8n-io_n8n_SecurityValidator_visit_Import | [→](./implementations/n8n-io_n8n_SecurityValidator_visit_Import.md) | ✅Principle:n8n-io_n8n_Import_Validation, ✅Env:n8n-io_n8n_Python_Task_Runner | Import statement validation |
| n8n-io_n8n_SecurityValidator_visit_Attribute | [→](./implementations/n8n-io_n8n_SecurityValidator_visit_Attribute.md) | ✅Principle:n8n-io_n8n_Pattern_Detection, ✅Env:n8n-io_n8n_Python_Task_Runner | Dangerous pattern detection |
| n8n-io_n8n_TaskAnalyzer_raise_security_error | [→](./implementations/n8n-io_n8n_TaskAnalyzer_raise_security_error.md) | ✅Principle:n8n-io_n8n_Violation_Reporting, ✅Env:n8n-io_n8n_Python_Task_Runner | Error aggregation |
| n8n-io_n8n_TaskExecutor_filter_builtins | [→](./implementations/n8n-io_n8n_TaskExecutor_filter_builtins.md) | ✅Principle:n8n-io_n8n_Builtin_Filtering, ✅Env:n8n-io_n8n_Python_Task_Runner | Builtin restriction |
| n8n-io_n8n_TaskExecutor_create_safe_import | [→](./implementations/n8n-io_n8n_TaskExecutor_create_safe_import.md) | ✅Principle:n8n-io_n8n_Runtime_Import_Validation, ✅Env:n8n-io_n8n_Python_Task_Runner | Runtime import validation |

## Workflow Comparison Implementations

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| n8n-io_n8n_load_workflow | [→](./implementations/n8n-io_n8n_load_workflow.md) | ✅Principle:n8n-io_n8n_Workflow_Loading, ✅Env:n8n-io_n8n_Python_Workflow_Comparison | JSON file loading |
| n8n-io_n8n_load_config | [→](./implementations/n8n-io_n8n_load_config.md) | ✅Principle:n8n-io_n8n_Configuration_Loading, ✅Env:n8n-io_n8n_Python_Workflow_Comparison | Preset/YAML/JSON config |
| n8n-io_n8n_build_workflow_graph | [→](./implementations/n8n-io_n8n_build_workflow_graph.md) | ✅Principle:n8n-io_n8n_Graph_Construction, ✅Env:n8n-io_n8n_Python_Workflow_Comparison | n8n to NetworkX conversion |
| n8n-io_n8n_relabel_graph_by_structure | [→](./implementations/n8n-io_n8n_relabel_graph_by_structure.md) | ✅Principle:n8n-io_n8n_Graph_Relabeling, ✅Env:n8n-io_n8n_Python_Workflow_Comparison | Structural relabeling |
| n8n-io_n8n_calculate_graph_edit_distance | [→](./implementations/n8n-io_n8n_calculate_graph_edit_distance.md) | ✅Principle:n8n-io_n8n_GED_Calculation, ✅Env:n8n-io_n8n_Python_Workflow_Comparison | GED computation |
| n8n-io_n8n_extract_operations_from_path | [→](./implementations/n8n-io_n8n_extract_operations_from_path.md) | ✅Principle:n8n-io_n8n_Edit_Extraction, ✅Env:n8n-io_n8n_Python_Workflow_Comparison | Edit operation extraction |
| n8n-io_n8n_similarity_formula | [→](./implementations/n8n-io_n8n_similarity_formula.md) | ✅Principle:n8n-io_n8n_Similarity_Calculation, ✅Env:n8n-io_n8n_Python_Workflow_Comparison | Pattern: formula |
| n8n-io_n8n_format_output | [→](./implementations/n8n-io_n8n_format_output.md) | ✅Principle:n8n-io_n8n_Result_Formatting, ✅Env:n8n-io_n8n_Python_Workflow_Comparison | JSON/Summary formatting |

---

**Total Implementations:** 24

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
