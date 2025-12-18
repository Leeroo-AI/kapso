# Principle Index: n8n-io_n8n

> Tracks Principle pages and their connections to Implementations, Workflows, etc.
> **Update IMMEDIATELY** after creating or modifying a Principle page.

## Summary

| Workflow | Principle Count |
|----------|----------------|
| Python_Task_Execution | 8 |
| Security_Validation_Pipeline | 8 |
| AI_Workflow_Comparison | 9 |
| **Total** | **25** |

## Pages

### Python Task Execution Workflow

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| n8n-io_n8n_Runner_Initialization | [→](./principles/n8n-io_n8n_Runner_Initialization.md) | ✅Impl:n8n-io_n8n_TaskRunner_init | Configuration-driven component initialization |
| n8n-io_n8n_Broker_Connection | [→](./principles/n8n-io_n8n_Broker_Connection.md) | ✅Impl:n8n-io_n8n_TaskRunner_start | WebSocket task distribution architecture |
| n8n-io_n8n_Task_Offer_Negotiation | [→](./principles/n8n-io_n8n_Task_Offer_Negotiation.md) | ✅Impl:n8n-io_n8n_TaskRunner_send_offers | Capacity-based task distribution |
| n8n-io_n8n_Security_Validation | [→](./principles/n8n-io_n8n_Security_Validation.md) | ✅Impl:n8n-io_n8n_TaskAnalyzer_validate | Static analysis with caching |
| n8n-io_n8n_Subprocess_Creation | [→](./principles/n8n-io_n8n_Subprocess_Creation.md) | ✅Impl:n8n-io_n8n_TaskExecutor_create_process | Process isolation via ForkServer |
| n8n-io_n8n_Sandboxed_Execution | [→](./principles/n8n-io_n8n_Sandboxed_Execution.md) | ✅Impl:n8n-io_n8n_TaskExecutor_execute | Restricted Python runtime |
| n8n-io_n8n_Result_Serialization | [→](./principles/n8n-io_n8n_Result_Serialization.md) | ✅Impl:n8n-io_n8n_TaskExecutor_put_result | Length-prefixed JSON framing |
| n8n-io_n8n_Result_Delivery | [→](./principles/n8n-io_n8n_Result_Delivery.md) | ✅Impl:n8n-io_n8n_TaskRunner_execute_task | End-to-end task lifecycle |

### Security Validation Pipeline Workflow

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| n8n-io_n8n_Security_Configuration | [→](./principles/n8n-io_n8n_Security_Configuration.md) | ✅Impl:n8n-io_n8n_SecurityConfig | Allow/deny list policy model |
| n8n-io_n8n_Validation_Caching | [→](./principles/n8n-io_n8n_Validation_Caching.md) | ✅Impl:n8n-io_n8n_TaskAnalyzer_cache | LRU caching for validation results |
| n8n-io_n8n_AST_Parsing | [→](./principles/n8n-io_n8n_AST_Parsing.md) | ✅Impl:n8n-io_n8n_ast_parse | Python AST for security analysis |
| n8n-io_n8n_Import_Analysis | [→](./principles/n8n-io_n8n_Import_Analysis.md) | ✅Impl:n8n-io_n8n_SecurityValidator_imports | Import statement validation |
| n8n-io_n8n_Dangerous_Pattern_Detection | [→](./principles/n8n-io_n8n_Dangerous_Pattern_Detection.md) | ✅Impl:n8n-io_n8n_SecurityValidator_patterns | Sandbox escape pattern detection |
| n8n-io_n8n_Violation_Aggregation | [→](./principles/n8n-io_n8n_Violation_Aggregation.md) | ✅Impl:n8n-io_n8n_SecurityViolationError | Structured error reporting |
| n8n-io_n8n_Runtime_Import_Validation | [→](./principles/n8n-io_n8n_Runtime_Import_Validation.md) | ✅Impl:n8n-io_n8n_validate_module_import | Dynamic import interception |
| n8n-io_n8n_Sandbox_Environment | [→](./principles/n8n-io_n8n_Sandbox_Environment.md) | ✅Impl:n8n-io_n8n_TaskExecutor_sandbox | Multi-layered sandbox construction |

### AI Workflow Comparison Workflow

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| n8n-io_n8n_Comparison_Configuration | [→](./principles/n8n-io_n8n_Comparison_Configuration.md) | ✅Impl:n8n-io_n8n_load_config | Flexible config with presets |
| n8n-io_n8n_Workflow_Parsing | [→](./principles/n8n-io_n8n_Workflow_Parsing.md) | ✅Impl:n8n-io_n8n_load_workflow | JSON workflow loading |
| n8n-io_n8n_Graph_Construction | [→](./principles/n8n-io_n8n_Graph_Construction.md) | ✅Impl:n8n-io_n8n_build_workflow_graph | Workflow to directed graph |
| n8n-io_n8n_Graph_Relabeling | [→](./principles/n8n-io_n8n_Graph_Relabeling.md) | ✅Impl:n8n-io_n8n_relabel_graph_by_structure | Name-independent comparison |
| n8n-io_n8n_Graph_Edit_Distance | [→](./principles/n8n-io_n8n_Graph_Edit_Distance.md) | ✅Impl:n8n-io_n8n_calculate_graph_edit_distance | GED with domain-specific costs |
| n8n-io_n8n_Edit_Operation_Extraction | [→](./principles/n8n-io_n8n_Edit_Operation_Extraction.md) | ✅Impl:n8n-io_n8n_extract_operations_from_path | Actionable edit operations |
| n8n-io_n8n_Similarity_Scoring | [→](./principles/n8n-io_n8n_Similarity_Scoring.md) | ✅Impl:n8n-io_n8n_calculate_max_cost | Normalized similarity metric |
| n8n-io_n8n_Priority_Assignment | [→](./principles/n8n-io_n8n_Priority_Assignment.md) | ✅Impl:n8n-io_n8n_determine_priority | Edit severity classification |
| n8n-io_n8n_Output_Formatting | [→](./principles/n8n-io_n8n_Output_Formatting.md) | ✅Impl:n8n-io_n8n_format_output | Multi-format output generation |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
