# Phase 2: Excavation + Synthesis Report

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
| **Total Workflows** | 3 |
| **Total Principle-Implementation Pairs** | 25 |
| **Principles Created** | 25 |
| **Implementations Created** | 25 |
| **1:1 Mappings Verified** | ✅ Yes |

---

## Workflow Breakdown

### Workflow 1: Python_Task_Execution

**Description:** End-to-end process for securely executing Python code tasks from broker communication through sandboxed execution to result delivery.

| # | Principle | Implementation | Status |
|---|-----------|----------------|--------|
| 1 | Runner_Initialization | TaskRunner_init | ✅ |
| 2 | Broker_Connection | TaskRunner_start | ✅ |
| 3 | Task_Offer_Negotiation | TaskRunner_send_offers | ✅ |
| 4 | Security_Validation | TaskAnalyzer_validate | ✅ |
| 5 | Subprocess_Creation | TaskExecutor_create_process | ✅ |
| 6 | Sandboxed_Execution | TaskExecutor_execute | ✅ |
| 7 | Result_Serialization | TaskExecutor_put_result | ✅ |
| 8 | Result_Delivery | TaskRunner_execute_task | ✅ |

**Source Files:**
- `packages/@n8n/task-runner-python/src/task_runner.py`
- `packages/@n8n/task-runner-python/src/task_executor.py`
- `packages/@n8n/task-runner-python/src/task_analyzer.py`
- `packages/@n8n/task-runner-python/src/config/task_runner_config.py`

---

### Workflow 2: Security_Validation_Pipeline

**Description:** Multi-layered security validation process combining AST-based static analysis with runtime import validation and sandbox enforcement.

| # | Principle | Implementation | Status |
|---|-----------|----------------|--------|
| 1 | Security_Configuration | SecurityConfig | ✅ |
| 2 | Validation_Caching | TaskAnalyzer_cache | ✅ |
| 3 | AST_Parsing | ast_parse | ✅ |
| 4 | Import_Analysis | SecurityValidator_imports | ✅ |
| 5 | Dangerous_Pattern_Detection | SecurityValidator_patterns | ✅ |
| 6 | Violation_Aggregation | SecurityViolationError | ✅ |
| 7 | Runtime_Import_Validation | validate_module_import | ✅ |
| 8 | Sandbox_Environment | TaskExecutor_sandbox | ✅ |

**Source Files:**
- `packages/@n8n/task-runner-python/src/task_analyzer.py`
- `packages/@n8n/task-runner-python/src/task_executor.py`
- `packages/@n8n/task-runner-python/src/import_validation.py`
- `packages/@n8n/task-runner-python/src/config/security_config.py`
- `packages/@n8n/task-runner-python/src/errors/security_violation_error.py`

---

### Workflow 3: AI_Workflow_Comparison

**Description:** End-to-end process for evaluating AI-generated n8n workflows against ground truth using graph edit distance.

| # | Principle | Implementation | Status |
|---|-----------|----------------|--------|
| 1 | Comparison_Configuration | load_config | ✅ |
| 2 | Workflow_Parsing | load_workflow | ✅ |
| 3 | Graph_Construction | build_workflow_graph | ✅ |
| 4 | Graph_Relabeling | relabel_graph_by_structure | ✅ |
| 5 | Graph_Edit_Distance | calculate_graph_edit_distance | ✅ |
| 6 | Edit_Operation_Extraction | extract_operations_from_path | ✅ |
| 7 | Similarity_Scoring | calculate_max_cost | ✅ |
| 8 | Priority_Assignment | determine_priority | ✅ |
| 9 | Output_Formatting | format_output | ✅ |

**Source Files:**
- `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/compare_workflows.py`
- `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/similarity.py`
- `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/graph_builder.py`
- `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/config_loader.py`

---

## Files Created

### Principles (25 files)

```
principles/
├── n8n-io_n8n_AST_Parsing.md
├── n8n-io_n8n_Broker_Connection.md
├── n8n-io_n8n_Comparison_Configuration.md
├── n8n-io_n8n_Dangerous_Pattern_Detection.md
├── n8n-io_n8n_Edit_Operation_Extraction.md
├── n8n-io_n8n_Graph_Construction.md
├── n8n-io_n8n_Graph_Edit_Distance.md
├── n8n-io_n8n_Graph_Relabeling.md
├── n8n-io_n8n_Import_Analysis.md
├── n8n-io_n8n_Output_Formatting.md
├── n8n-io_n8n_Priority_Assignment.md
├── n8n-io_n8n_Result_Delivery.md
├── n8n-io_n8n_Result_Serialization.md
├── n8n-io_n8n_Runner_Initialization.md
├── n8n-io_n8n_Runtime_Import_Validation.md
├── n8n-io_n8n_Sandbox_Environment.md
├── n8n-io_n8n_Sandboxed_Execution.md
├── n8n-io_n8n_Security_Configuration.md
├── n8n-io_n8n_Security_Validation.md
├── n8n-io_n8n_Similarity_Scoring.md
├── n8n-io_n8n_Subprocess_Creation.md
├── n8n-io_n8n_Task_Offer_Negotiation.md
├── n8n-io_n8n_Validation_Caching.md
├── n8n-io_n8n_Violation_Aggregation.md
└── n8n-io_n8n_Workflow_Parsing.md
```

### Implementations (25 files)

```
implementations/
├── n8n-io_n8n_ast_parse.md
├── n8n-io_n8n_build_workflow_graph.md
├── n8n-io_n8n_calculate_graph_edit_distance.md
├── n8n-io_n8n_calculate_max_cost.md
├── n8n-io_n8n_determine_priority.md
├── n8n-io_n8n_extract_operations_from_path.md
├── n8n-io_n8n_format_output.md
├── n8n-io_n8n_load_config.md
├── n8n-io_n8n_load_workflow.md
├── n8n-io_n8n_relabel_graph_by_structure.md
├── n8n-io_n8n_SecurityConfig.md
├── n8n-io_n8n_SecurityValidator_imports.md
├── n8n-io_n8n_SecurityValidator_patterns.md
├── n8n-io_n8n_SecurityViolationError.md
├── n8n-io_n8n_TaskAnalyzer_cache.md
├── n8n-io_n8n_TaskAnalyzer_validate.md
├── n8n-io_n8n_TaskExecutor_create_process.md
├── n8n-io_n8n_TaskExecutor_execute.md
├── n8n-io_n8n_TaskExecutor_put_result.md
├── n8n-io_n8n_TaskExecutor_sandbox.md
├── n8n-io_n8n_TaskRunner_execute_task.md
├── n8n-io_n8n_TaskRunner_init.md
├── n8n-io_n8n_TaskRunner_send_offers.md
├── n8n-io_n8n_TaskRunner_start.md
└── n8n-io_n8n_validate_module_import.md
```

---

## Indexes Updated

| Index File | Status |
|------------|--------|
| `_PrincipleIndex.md` | ✅ Updated |
| `_ImplementationIndex.md` | ✅ Updated |
| `_WorkflowIndex.md` | ✅ Already complete from Phase 1b |

---

## Page Structure Verification

All pages follow the required MediaWiki format:

### Principle Pages Include:
- ✅ Metadata infobox (Knowledge Sources, Domains, Last Updated)
- ✅ Overview section with Description and Usage
- ✅ Theoretical Basis with code examples
- ✅ Related Pages with `[[implemented_by::Implementation:...]]` link

### Implementation Pages Include:
- ✅ Metadata infobox (Knowledge Sources, Domains, Last Updated)
- ✅ Overview section with Description and Usage
- ✅ Code Reference (Source Location, Signature, Import)
- ✅ I/O Contract (Inputs/Outputs tables)
- ✅ Usage Examples with code
- ✅ Related Pages with `[[implements::Principle:...]]` link

---

## Semantic Link Verification

All 25 pairs have bidirectional semantic links:

| Principle → Implementation | Implementation → Principle |
|---------------------------|---------------------------|
| `[[implemented_by::Implementation:n8n-io_n8n_*]]` | `[[implements::Principle:n8n-io_n8n_*]]` |

---

## Key Technical Concepts Documented

### Security Domain
- AST-based static analysis for code security
- Runtime import interception and validation
- Sandbox environment construction (builtin filtering, module sanitization)
- Allow/deny list policy model

### Graph Theory Domain
- Graph Edit Distance (GED) algorithm
- NetworkX integration for directed graphs
- Structural graph relabeling for name-independent comparison
- Cost-based similarity scoring

### Workflow Execution Domain
- WebSocket-based broker-runner communication
- ForkServer subprocess isolation
- Length-prefixed JSON IPC
- LRU caching for validation results

---

## Notes

1. **Implementation Types Used:**
   - API Doc (24): Functions/classes in the n8n repository
   - Wrapper Doc (1): `ast_parse` wrapping Python stdlib `ast.parse`

2. **Environments Referenced:**
   - `n8n-io_n8n_Python_Task_Runner_Env` - Main task runner environment
   - `n8n-io_n8n_Sandbox_Environment` - Restricted execution environment
   - `n8n-io_n8n_Workflow_Comparison_Env` - AI workflow comparison environment

3. **No Errors Encountered:** All 25 pairs created successfully with proper formatting and links.

---

## Next Steps

Phase 2 is complete. The wiki is ready for:
- Phase 3: Cross-referencing and validation
- Integration with the main wiki system
- User review and feedback
