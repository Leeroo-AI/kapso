# Principle Index: n8n-io_n8n

> Index of all Principle pages and their connections to Implementations and Workflows.

## Python Task Execution Principles

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| n8n-io_n8n_WebSocket_Connection | [→](./principles/n8n-io_n8n_WebSocket_Connection.md) | ✅Impl:n8n-io_n8n_TaskRunner_start, ✅Workflow:n8n-io_n8n_Python_Task_Execution | WebSocket broker communication |
| n8n-io_n8n_Offer_Based_Distribution | [→](./principles/n8n-io_n8n_Offer_Based_Distribution.md) | ✅Impl:n8n-io_n8n_TaskRunner_send_offers, ✅Workflow:n8n-io_n8n_Python_Task_Execution | Pull-based task distribution |
| n8n-io_n8n_Task_Acceptance | [→](./principles/n8n-io_n8n_Task_Acceptance.md) | ✅Impl:n8n-io_n8n_TaskRunner_handle_task_offer_accept, ✅Workflow:n8n-io_n8n_Python_Task_Execution | Offer validation protocol |
| n8n-io_n8n_Static_Security_Analysis | [→](./principles/n8n-io_n8n_Static_Security_Analysis.md) | ✅Impl:n8n-io_n8n_TaskAnalyzer_validate, ✅Workflow:n8n-io_n8n_Python_Task_Execution | AST-based pre-validation |
| n8n-io_n8n_Subprocess_Isolation | [→](./principles/n8n-io_n8n_Subprocess_Isolation.md) | ✅Impl:n8n-io_n8n_TaskExecutor_create_process, ✅Workflow:n8n-io_n8n_Python_Task_Execution | Process-level sandboxing |
| n8n-io_n8n_Code_Execution | [→](./principles/n8n-io_n8n_Code_Execution.md) | ✅Impl:n8n-io_n8n_TaskExecutor_all_items, ✅Workflow:n8n-io_n8n_Python_Task_Execution | Sandboxed execution modes |
| n8n-io_n8n_Result_Collection | [→](./principles/n8n-io_n8n_Result_Collection.md) | ✅Impl:n8n-io_n8n_TaskExecutor_execute_process, ✅Workflow:n8n-io_n8n_Python_Task_Execution | IPC result collection |
| n8n-io_n8n_Task_Completion | [→](./principles/n8n-io_n8n_Task_Completion.md) | ✅Impl:n8n-io_n8n_TaskRunner_execute_task, ✅Workflow:n8n-io_n8n_Python_Task_Execution | Full task lifecycle |

## Security Validation Principles

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| n8n-io_n8n_Task_Settings_Reception | [→](./principles/n8n-io_n8n_Task_Settings_Reception.md) | ✅Impl:n8n-io_n8n_TaskRunner_handle_task_settings, ✅Workflow:n8n-io_n8n_Security_Validation | Config reception |
| n8n-io_n8n_Validation_Caching | [→](./principles/n8n-io_n8n_Validation_Caching.md) | ✅Impl:n8n-io_n8n_TaskAnalyzer_cache, ✅Workflow:n8n-io_n8n_Security_Validation | LRU memoization |
| n8n-io_n8n_AST_Parsing | [→](./principles/n8n-io_n8n_AST_Parsing.md) | ✅Impl:n8n-io_n8n_ast_parse, ✅Workflow:n8n-io_n8n_Security_Validation | Code to AST conversion |
| n8n-io_n8n_Import_Validation | [→](./principles/n8n-io_n8n_Import_Validation.md) | ✅Impl:n8n-io_n8n_SecurityValidator_visit_Import, ✅Workflow:n8n-io_n8n_Security_Validation | Allowlist validation |
| n8n-io_n8n_Pattern_Detection | [→](./principles/n8n-io_n8n_Pattern_Detection.md) | ✅Impl:n8n-io_n8n_SecurityValidator_visit_Attribute, ✅Workflow:n8n-io_n8n_Security_Validation | Dangerous pattern detection |
| n8n-io_n8n_Violation_Reporting | [→](./principles/n8n-io_n8n_Violation_Reporting.md) | ✅Impl:n8n-io_n8n_TaskAnalyzer_raise_security_error, ✅Workflow:n8n-io_n8n_Security_Validation | Error aggregation |
| n8n-io_n8n_Builtin_Filtering | [→](./principles/n8n-io_n8n_Builtin_Filtering.md) | ✅Impl:n8n-io_n8n_TaskExecutor_filter_builtins, ✅Workflow:n8n-io_n8n_Security_Validation | Runtime builtin restriction |
| n8n-io_n8n_Runtime_Import_Validation | [→](./principles/n8n-io_n8n_Runtime_Import_Validation.md) | ✅Impl:n8n-io_n8n_TaskExecutor_create_safe_import, ✅Workflow:n8n-io_n8n_Security_Validation | Dynamic import validation |

## Workflow Comparison Principles

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| n8n-io_n8n_Workflow_Loading | [→](./principles/n8n-io_n8n_Workflow_Loading.md) | ✅Impl:n8n-io_n8n_load_workflow, ✅Workflow:n8n-io_n8n_Workflow_Comparison | File loading |
| n8n-io_n8n_Configuration_Loading | [→](./principles/n8n-io_n8n_Configuration_Loading.md) | ✅Impl:n8n-io_n8n_load_config, ✅Workflow:n8n-io_n8n_Workflow_Comparison | Preset/config loading |
| n8n-io_n8n_Graph_Construction | [→](./principles/n8n-io_n8n_Graph_Construction.md) | ✅Impl:n8n-io_n8n_build_workflow_graph, ✅Workflow:n8n-io_n8n_Workflow_Comparison | JSON to graph |
| n8n-io_n8n_Graph_Relabeling | [→](./principles/n8n-io_n8n_Graph_Relabeling.md) | ✅Impl:n8n-io_n8n_relabel_graph_by_structure, ✅Workflow:n8n-io_n8n_Workflow_Comparison | Structural relabeling |
| n8n-io_n8n_GED_Calculation | [→](./principles/n8n-io_n8n_GED_Calculation.md) | ✅Impl:n8n-io_n8n_calculate_graph_edit_distance, ✅Workflow:n8n-io_n8n_Workflow_Comparison | Graph edit distance |
| n8n-io_n8n_Edit_Extraction | [→](./principles/n8n-io_n8n_Edit_Extraction.md) | ✅Impl:n8n-io_n8n_extract_operations_from_path, ✅Workflow:n8n-io_n8n_Workflow_Comparison | Operation extraction |
| n8n-io_n8n_Similarity_Calculation | [→](./principles/n8n-io_n8n_Similarity_Calculation.md) | ✅Impl:n8n-io_n8n_similarity_formula, ✅Workflow:n8n-io_n8n_Workflow_Comparison | Score normalization |
| n8n-io_n8n_Result_Formatting | [→](./principles/n8n-io_n8n_Result_Formatting.md) | ✅Impl:n8n-io_n8n_format_output, ✅Workflow:n8n-io_n8n_Workflow_Comparison | Output formatting |

---

**Total Principles:** 24

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
