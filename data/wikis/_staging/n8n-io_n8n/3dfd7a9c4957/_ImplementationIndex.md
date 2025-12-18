# Implementation Index: n8n-io_n8n

> Tracks Implementation pages and their connections to Principles, Environments, etc.
> **Update IMMEDIATELY** after creating or modifying a Implementation page.

## Summary

| Workflow | Implementation Count |
|----------|---------------------|
| Python_Task_Execution | 8 |
| Security_Validation_Pipeline | 8 |
| AI_Workflow_Comparison | 9 |
| **Total** | **25** |

## Pages

### Python Task Execution Workflow

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| n8n-io_n8n_TaskRunner_init | [→](./implementations/n8n-io_n8n_TaskRunner_init.md) | ✅Principle:n8n-io_n8n_Runner_Initialization | Initializes runner with config, executor, analyzer |
| n8n-io_n8n_TaskRunner_start | [→](./implementations/n8n-io_n8n_TaskRunner_start.md) | ✅Principle:n8n-io_n8n_Broker_Connection | WebSocket connection to broker |
| n8n-io_n8n_TaskRunner_send_offers | [→](./implementations/n8n-io_n8n_TaskRunner_send_offers.md) | ✅Principle:n8n-io_n8n_Task_Offer_Negotiation | Capacity-based task offer protocol |
| n8n-io_n8n_TaskAnalyzer_validate | [→](./implementations/n8n-io_n8n_TaskAnalyzer_validate.md) | ✅Principle:n8n-io_n8n_Security_Validation | AST-based code security analysis |
| n8n-io_n8n_TaskExecutor_create_process | [→](./implementations/n8n-io_n8n_TaskExecutor_create_process.md) | ✅Principle:n8n-io_n8n_Subprocess_Creation | ForkServer subprocess isolation |
| n8n-io_n8n_TaskExecutor_execute | [→](./implementations/n8n-io_n8n_TaskExecutor_execute.md) | ✅Principle:n8n-io_n8n_Sandboxed_Execution | Sandboxed code execution modes |
| n8n-io_n8n_TaskExecutor_put_result | [→](./implementations/n8n-io_n8n_TaskExecutor_put_result.md) | ✅Principle:n8n-io_n8n_Result_Serialization | Length-prefixed JSON IPC |
| n8n-io_n8n_TaskRunner_execute_task | [→](./implementations/n8n-io_n8n_TaskRunner_execute_task.md) | ✅Principle:n8n-io_n8n_Result_Delivery | End-to-end task orchestration |

### Security Validation Pipeline Workflow

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| n8n-io_n8n_SecurityConfig | [→](./implementations/n8n-io_n8n_SecurityConfig.md) | ✅Principle:n8n-io_n8n_Security_Configuration | Dataclass with allow/deny lists |
| n8n-io_n8n_TaskAnalyzer_cache | [→](./implementations/n8n-io_n8n_TaskAnalyzer_cache.md) | ✅Principle:n8n-io_n8n_Validation_Caching | LRU cache with hash keys |
| n8n-io_n8n_ast_parse | [→](./implementations/n8n-io_n8n_ast_parse.md) | ✅Principle:n8n-io_n8n_AST_Parsing | Python stdlib AST parser wrapper |
| n8n-io_n8n_SecurityValidator_imports | [→](./implementations/n8n-io_n8n_SecurityValidator_imports.md) | ✅Principle:n8n-io_n8n_Import_Analysis | Import statement validation |
| n8n-io_n8n_SecurityValidator_patterns | [→](./implementations/n8n-io_n8n_SecurityValidator_patterns.md) | ✅Principle:n8n-io_n8n_Dangerous_Pattern_Detection | Blocked names/attributes detection |
| n8n-io_n8n_SecurityViolationError | [→](./implementations/n8n-io_n8n_SecurityViolationError.md) | ✅Principle:n8n-io_n8n_Violation_Aggregation | Custom exception for violations |
| n8n-io_n8n_validate_module_import | [→](./implementations/n8n-io_n8n_validate_module_import.md) | ✅Principle:n8n-io_n8n_Runtime_Import_Validation | Runtime import validation |
| n8n-io_n8n_TaskExecutor_sandbox | [→](./implementations/n8n-io_n8n_TaskExecutor_sandbox.md) | ✅Principle:n8n-io_n8n_Sandbox_Environment | Sandbox setup methods |

### AI Workflow Comparison Workflow

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| n8n-io_n8n_load_config | [→](./implementations/n8n-io_n8n_load_config.md) | ✅Principle:n8n-io_n8n_Comparison_Configuration | YAML/JSON config loading |
| n8n-io_n8n_load_workflow | [→](./implementations/n8n-io_n8n_load_workflow.md) | ✅Principle:n8n-io_n8n_Workflow_Parsing | JSON workflow parsing |
| n8n-io_n8n_build_workflow_graph | [→](./implementations/n8n-io_n8n_build_workflow_graph.md) | ✅Principle:n8n-io_n8n_Graph_Construction | Workflow to NetworkX graph |
| n8n-io_n8n_relabel_graph_by_structure | [→](./implementations/n8n-io_n8n_relabel_graph_by_structure.md) | ✅Principle:n8n-io_n8n_Graph_Relabeling | Structural ID assignment |
| n8n-io_n8n_calculate_graph_edit_distance | [→](./implementations/n8n-io_n8n_calculate_graph_edit_distance.md) | ✅Principle:n8n-io_n8n_Graph_Edit_Distance | GED with custom costs |
| n8n-io_n8n_extract_operations_from_path | [→](./implementations/n8n-io_n8n_extract_operations_from_path.md) | ✅Principle:n8n-io_n8n_Edit_Operation_Extraction | Edit path to operations |
| n8n-io_n8n_calculate_max_cost | [→](./implementations/n8n-io_n8n_calculate_max_cost.md) | ✅Principle:n8n-io_n8n_Similarity_Scoring | Max cost for normalization |
| n8n-io_n8n_determine_priority | [→](./implementations/n8n-io_n8n_determine_priority.md) | ✅Principle:n8n-io_n8n_Priority_Assignment | Priority classification |
| n8n-io_n8n_format_output | [→](./implementations/n8n-io_n8n_format_output.md) | ✅Principle:n8n-io_n8n_Output_Formatting | JSON and summary formatters |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
