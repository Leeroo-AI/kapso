# Phase 1: Anchoring Report

## Summary

- Workflows created: 3
- Total steps documented: 24
- Implementation hints captured: 24

## Workflows Created

| Workflow | Source Files | Steps | Implementation APIs |
|----------|--------------|-------|---------------------|
| n8n-io_n8n_Python_Task_Execution | task_runner.py, task_executor.py, task_analyzer.py, pipe_reader.py, message_serde.py | 8 | TaskRunner.start, TaskExecutor.create_process, TaskAnalyzer.validate, PipeReader |
| n8n-io_n8n_Security_Validation | task_analyzer.py, task_executor.py, import_validation.py, constants.py | 8 | TaskAnalyzer.validate, SecurityValidator, _filter_builtins, _create_safe_import |
| n8n-io_n8n_Workflow_Comparison | compare_workflows.py, graph_builder.py, similarity.py, config_loader.py, cost_functions.py | 8 | build_workflow_graph, calculate_graph_edit_distance, load_config, format_output_json |

## Coverage Summary

- Source files covered: 45/60 (75%)
- Core source files documented: 100%
- Test files not covered (as expected)

### Files by Coverage

**Covered by Workflow: n8n-io_n8n_Python_Task_Execution (38 files)**
- task_runner.py - Core task orchestrator
- task_executor.py - Isolated subprocess execution
- task_state.py - Task lifecycle state model
- pipe_reader.py - Subprocess result reader
- message_serde.py - WebSocket message serialization
- All message_types/* files
- All errors/* files
- All config/* files
- main.py, env.py, logs.py, shutdown.py, sentry.py, health_check_server.py, nanoid.py

**Covered by Workflow: n8n-io_n8n_Security_Validation (5 files)**
- task_analyzer.py - Static security validation (AST)
- import_validation.py - Runtime import validation
- security_config.py (shared with Python_Task_Execution)
- constants.py (shared with Python_Task_Execution)
- task_executor.py (shared with Python_Task_Execution)

**Covered by Workflow: n8n-io_n8n_Workflow_Comparison (7 files)**
- compare_workflows.py - CLI workflow comparison tool
- graph_builder.py - n8n to NetworkX converter
- similarity.py - Graph edit distance algorithm
- config_loader.py - Configuration management system
- cost_functions.py - Graph edit distance costs
- __init__.py - Package initialization
- __main__.py - Module execution entry point

**Not Covered (Test files - 15 files)**
- All files in tests/ directories (expected, test files don't need Workflow coverage)

## Implementation Context Captured

| Workflow | Principles | API Docs | Wrapper Docs | Pattern Docs | External Tool Docs |
|----------|------------|----------|--------------|--------------|-------------------|
| Python_Task_Execution | 8 | 8 | 0 | 0 | 0 |
| Security_Validation | 8 | 7 | 0 | 0 | 1 |
| Workflow_Comparison | 8 | 7 | 0 | 1 | 0 |
| **Total** | **24** | **22** | **0** | **1** | **1** |

## Notes for Excavation Phase

### APIs to Extract (with Source Locations)

#### Python Task Runner Package

| API | Source | Used By Principles |
|-----|--------|-------------------|
| `TaskRunner.start` | task_runner.py:L115-146 | WebSocket_Connection |
| `TaskRunner._send_offers_loop` | task_runner.py:L431-473 | Offer_Based_Distribution |
| `TaskRunner._handle_task_offer_accept` | task_runner.py:L253-280 | Task_Acceptance |
| `TaskRunner._execute_task` | task_runner.py:L302-371 | Task_Completion |
| `TaskRunner._handle_task_settings` | task_runner.py:L282-300 | Task_Settings_Reception |
| `TaskAnalyzer.validate` | task_analyzer.py:L172-196 | Static_Security_Analysis |
| `TaskAnalyzer._to_cache_key` | task_analyzer.py:L203-212 | Validation_Caching |
| `SecurityValidator.visit_Import` | task_analyzer.py:L34-50 | Import_Validation |
| `SecurityValidator.visit_Attribute` | task_analyzer.py:L52-71 | Pattern_Detection |
| `TaskExecutor.create_process` | task_executor.py:L56-86 | Subprocess_Isolation |
| `TaskExecutor.execute_process` | task_executor.py:L88-165 | Result_Collection |
| `TaskExecutor._all_items` | task_executor.py:L185-223 | Code_Execution |
| `TaskExecutor._per_item` | task_executor.py:L224-278 | Code_Execution |
| `TaskExecutor._filter_builtins` | task_executor.py:L424-439 | Builtin_Filtering |
| `TaskExecutor._create_safe_import` | task_executor.py:L479-495 | Runtime_Import_Validation |

#### AI Workflow Builder Package

| API | Source | Used By Principles |
|-----|--------|-------------------|
| `load_workflow` | compare_workflows.py:L67-91 | Workflow_Loading |
| `load_config` | config_loader.py | Configuration_Loading |
| `build_workflow_graph` | graph_builder.py:L10-90 | Graph_Construction |
| `_relabel_graph_by_structure` | similarity.py:L421-501 | Graph_Relabeling |
| `calculate_graph_edit_distance` | similarity.py:L19-144 | GED_Calculation |
| `_extract_operations_from_path` | similarity.py:L223-386 | Edit_Extraction |
| `format_output_json` | compare_workflows.py:L94-116 | Result_Formatting |
| `format_output_summary` | compare_workflows.py:L176-255 | Result_Formatting |

### External Dependencies to Document

| Library | Used For | Workflows |
|---------|----------|-----------|
| `websockets` | WebSocket client connection | Python_Task_Execution |
| `networkx` | Graph algorithms, GED calculation | Workflow_Comparison |
| `pyyaml` | YAML config parsing | Workflow_Comparison |
| `ast` (stdlib) | AST parsing for security analysis | Security_Validation |
| `multiprocessing` (stdlib) | Subprocess isolation | Python_Task_Execution |
| `hashlib` (stdlib) | Code hashing for cache keys | Security_Validation |

### User-Defined Patterns to Document

| Pattern | Description | Workflow |
|---------|-------------|----------|
| `similarity_formula` | `1.0 - (edit_cost / max_cost)` similarity calculation | Workflow_Comparison |

### Key Architecture Patterns Discovered

1. **Offer-Based Task Distribution**
   - Runners advertise capacity via offers with validity windows
   - Broker selects runners based on available offers
   - Enables load balancing without central assignment

2. **Defense-in-Depth Security**
   - Static AST analysis before execution
   - Runtime import interception
   - Builtin filtering
   - Environment variable clearing
   - sys.modules sanitization

3. **Length-Prefixed IPC Protocol**
   - 4-byte big-endian length prefix
   - JSON payload for results and errors
   - Background thread for non-blocking reads

4. **Graph Edit Distance for Workflow Comparison**
   - n8n JSON → NetworkX DiGraph conversion
   - Structural relabeling for position-based matching
   - Custom cost functions with trigger prioritization

### Recommended Principle Pages (24 total)

#### Python Task Execution (8 Principles)
1. `n8n-io_n8n_WebSocket_Connection` - Persistent broker connection with reconnection
2. `n8n-io_n8n_Offer_Based_Distribution` - Capacity-based task distribution
3. `n8n-io_n8n_Task_Acceptance` - Offer validation and task state creation
4. `n8n-io_n8n_Static_Security_Analysis` - AST-based code validation
5. `n8n-io_n8n_Subprocess_Isolation` - Forkserver-based process isolation
6. `n8n-io_n8n_Code_Execution` - All-items and per-item execution modes
7. `n8n-io_n8n_Result_Collection` - IPC pipe reading and timeout handling
8. `n8n-io_n8n_Task_Completion` - Result/error reporting and cleanup

#### Security Validation (8 Principles)
1. `n8n-io_n8n_Task_Settings_Reception` - Code extraction from broker messages
2. `n8n-io_n8n_Validation_Caching` - LRU cache for validation results
3. `n8n-io_n8n_AST_Parsing` - Code parsing and syntax checking
4. `n8n-io_n8n_Import_Validation` - Allowlist-based import checking
5. `n8n-io_n8n_Pattern_Detection` - Dangerous attribute/name detection
6. `n8n-io_n8n_Violation_Reporting` - Multi-violation error aggregation
7. `n8n-io_n8n_Builtin_Filtering` - Runtime builtin restriction
8. `n8n-io_n8n_Runtime_Import_Validation` - Dynamic import interception

#### Workflow Comparison (8 Principles)
1. `n8n-io_n8n_Workflow_Loading` - JSON file loading with error handling
2. `n8n-io_n8n_Configuration_Loading` - Preset and custom config loading
3. `n8n-io_n8n_Graph_Construction` - n8n to NetworkX conversion
4. `n8n-io_n8n_Graph_Relabeling` - Structural ID assignment
5. `n8n-io_n8n_GED_Calculation` - Graph edit distance computation
6. `n8n-io_n8n_Edit_Extraction` - Edit operation description generation
7. `n8n-io_n8n_Similarity_Calculation` - Normalized similarity scoring
8. `n8n-io_n8n_Result_Formatting` - JSON and summary output formatting

## Files Created

```
/home/ubuntu/praxium/data/wikis_batch/_staging/n8n-io_n8n/4c8d050c7de5/
├── workflows/
│   ├── n8n-io_n8n_Python_Task_Execution.md
│   ├── n8n-io_n8n_Security_Validation.md
│   └── n8n-io_n8n_Workflow_Comparison.md
├── _WorkflowIndex.md (updated)
├── _RepoMap_n8n-io_n8n.md (updated)
└── _reports/
    └── phase1_anchoring.md (this file)
```

## Completion Status

Phase 1 Anchoring is **COMPLETE**.

All three suggested workflows from Phase 0 have been documented:
1. ✅ Task Execution Flow → `n8n-io_n8n_Python_Task_Execution`
2. ✅ Security Validation Flow → `n8n-io_n8n_Security_Validation`
3. ✅ Workflow Comparison Flow → `n8n-io_n8n_Workflow_Comparison`

Phase 2 (Excavation) can now proceed to create:
- 24 Principle pages
- 24 Implementation pages
- 1 Environment page (n8n-io_n8n_Python)
