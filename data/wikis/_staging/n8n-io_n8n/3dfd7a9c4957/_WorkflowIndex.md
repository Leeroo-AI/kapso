# Workflow Index: n8n-io_n8n

> Comprehensive index of Workflows and their implementation context.
> This index bridges Phase 1 (Anchoring) and Phase 2 (Excavation).
> **Update IMMEDIATELY** after creating or modifying a Workflow page.

---

## Summary

| Workflow | Steps | Principles | Rough APIs |
|----------|-------|------------|------------|
| Python_Task_Execution | 8 | 8 | TaskRunner.start, TaskExecutor.create_process, TaskExecutor.execute_process, TaskAnalyzer.validate |
| Security_Validation_Pipeline | 8 | 8 | TaskAnalyzer.validate, SecurityValidator, validate_module_import, _filter_builtins |
| AI_Workflow_Comparison | 9 | 9 | load_config, build_workflow_graph, calculate_graph_edit_distance |

---

## Workflow: n8n-io_n8n_Python_Task_Execution

**File:** [→](./workflows/n8n-io_n8n_Python_Task_Execution.md)
**Description:** End-to-end process for securely executing Python code tasks from broker communication through sandboxed execution to result delivery.

### Steps Overview

| # | Step Name | Principle | Rough API | Related Files |
|---|-----------|-----------|-----------|---------------|
| 1 | Runner Initialization | Runner_Initialization | `TaskRunner.__init__`, `TaskRunnerConfig.from_env` | main.py, task_runner.py, task_runner_config.py |
| 2 | Broker Connection | Broker_Connection | `TaskRunner.start`, `websockets.connect` | task_runner.py |
| 3 | Task Offer Negotiation | Task_Offer_Negotiation | `TaskRunner._send_offers`, `TaskRunner._handle_task_offer_accept` | task_runner.py |
| 4 | Security Validation | Security_Validation | `TaskAnalyzer.validate` | task_analyzer.py, import_validation.py |
| 5 | Subprocess Creation | Subprocess_Creation | `TaskExecutor.create_process` | task_executor.py |
| 6 | Sandboxed Code Execution | Sandboxed_Execution | `TaskExecutor._all_items`, `TaskExecutor._per_item` | task_executor.py |
| 7 | Result Serialization | Result_Serialization | `TaskExecutor._put_result`, `json.dumps` | task_executor.py |
| 8 | Result Delivery | Result_Delivery | `TaskRunner._execute_task`, `RunnerTaskDone` | task_runner.py |

### Source Files (for enrichment)

- `packages/@n8n/task-runner-python/src/main.py` - Application entry point and startup
- `packages/@n8n/task-runner-python/src/task_runner.py` - Main task orchestrator (501 lines)
- `packages/@n8n/task-runner-python/src/task_executor.py` - Sandboxed code executor (506 lines)
- `packages/@n8n/task-runner-python/src/task_analyzer.py` - Code security analysis (212 lines)
- `packages/@n8n/task-runner-python/src/message_serde.py` - Message serialization
- `packages/@n8n/task-runner-python/src/pipe_reader.py` - Subprocess pipe reader
- `packages/@n8n/task-runner-python/src/config/task_runner_config.py` - Configuration dataclass

### Step 1: Runner_Initialization

| Attribute | Value |
|-----------|-------|
| **Principle** | `n8n-io_n8n_Runner_Initialization` |
| **Implementation** | `n8n-io_n8n_TaskRunner_init` |
| **API Call** | `TaskRunner.__init__(config: TaskRunnerConfig) -> None` |
| **Source Location** | `packages/@n8n/task-runner-python/src/task_runner.py:L74-110` |
| **External Dependencies** | `websockets`, `logging`, `asyncio` |
| **Environment** | `n8n-io_n8n_Python_Task_Runner_Env` |
| **Key Parameters** | `config: TaskRunnerConfig` - Complete configuration including broker URI, concurrency limits, security settings |
| **Inputs** | `TaskRunnerConfig` object from `TaskRunnerConfig.from_env()` |
| **Outputs** | Initialized `TaskRunner` instance with `runner_id`, `executor`, `analyzer`, `security_config` |

### Step 2: Broker_Connection

| Attribute | Value |
|-----------|-------|
| **Principle** | `n8n-io_n8n_Broker_Connection` |
| **Implementation** | `n8n-io_n8n_TaskRunner_start` |
| **API Call** | `TaskRunner.start() -> None` (async) |
| **Source Location** | `packages/@n8n/task-runner-python/src/task_runner.py:L115-146` |
| **External Dependencies** | `websockets`, `websockets.asyncio.client.ClientConnection` |
| **Environment** | `n8n-io_n8n_Python_Task_Runner_Env` |
| **Key Parameters** | Uses `self.websocket_url`, `self.config.grant_token`, `self.config.max_payload_size` |
| **Inputs** | Broker URI and authentication token from config |
| **Outputs** | Active WebSocket connection (`self.websocket_connection`), message listener coroutine started |

### Step 3: Task_Offer_Negotiation

| Attribute | Value |
|-----------|-------|
| **Principle** | `n8n-io_n8n_Task_Offer_Negotiation` |
| **Implementation** | `n8n-io_n8n_TaskRunner_send_offers` |
| **API Call** | `TaskRunner._send_offers() -> None` (async), `TaskRunner._handle_task_offer_accept(message: BrokerTaskOfferAccept) -> None` (async) |
| **Source Location** | `packages/@n8n/task-runner-python/src/task_runner.py:L441-474`, `L253-280` |
| **External Dependencies** | `time`, `random` |
| **Environment** | `n8n-io_n8n_Python_Task_Runner_Env` |
| **Key Parameters** | `OFFER_INTERVAL: 0.25s`, `OFFER_VALIDITY: 5000ms`, `max_concurrency` from config |
| **Inputs** | Current running task count, open offers dictionary |
| **Outputs** | `TaskOffer` objects added to `self.open_offers`, `RunnerTaskOffer` messages sent to broker |

### Step 4: Security_Validation

| Attribute | Value |
|-----------|-------|
| **Principle** | `n8n-io_n8n_Security_Validation` |
| **Implementation** | `n8n-io_n8n_TaskAnalyzer_validate` |
| **API Call** | `TaskAnalyzer.validate(code: str) -> None` |
| **Source Location** | `packages/@n8n/task-runner-python/src/task_analyzer.py:L172-196` |
| **External Dependencies** | `ast`, `hashlib`, `collections.OrderedDict` |
| **Environment** | `n8n-io_n8n_Python_Task_Runner_Env` |
| **Key Parameters** | `code: str` - Python source code to validate |
| **Inputs** | Python code string from task settings |
| **Outputs** | None (returns silently if valid), raises `SecurityViolationError` if violations detected |

### Step 5: Subprocess_Creation

| Attribute | Value |
|-----------|-------|
| **Principle** | `n8n-io_n8n_Subprocess_Creation` |
| **Implementation** | `n8n-io_n8n_TaskExecutor_create_process` |
| **API Call** | `TaskExecutor.create_process(code: str, node_mode: NodeMode, items: Items, security_config: SecurityConfig, query: Query = None) -> tuple[ForkServerProcess, PipeConnection, PipeConnection]` |
| **Source Location** | `packages/@n8n/task-runner-python/src/task_executor.py:L56-86` |
| **External Dependencies** | `multiprocessing`, `multiprocessing.context.ForkServerProcess` |
| **Environment** | `n8n-io_n8n_Python_Task_Runner_Env` |
| **Key Parameters** | `code: str`, `node_mode: NodeMode` (all_items/per_item), `items: Items`, `security_config: SecurityConfig` |
| **Inputs** | Task code, execution mode, input items, security configuration |
| **Outputs** | Tuple of (process, read_connection, write_connection) for subprocess communication |

### Step 6: Sandboxed_Execution

| Attribute | Value |
|-----------|-------|
| **Principle** | `n8n-io_n8n_Sandboxed_Execution` |
| **Implementation** | `n8n-io_n8n_TaskExecutor_execute` |
| **API Call** | `TaskExecutor._all_items(raw_code: str, items: Items, write_conn, security_config: SecurityConfig, query: Query = None) -> None`, `TaskExecutor._per_item(raw_code: str, items: Items, write_conn, security_config: SecurityConfig, _query: Query = None) -> None` |
| **Source Location** | `packages/@n8n/task-runner-python/src/task_executor.py:L186-278` |
| **External Dependencies** | `textwrap`, `io`, `os`, `sys` |
| **Environment** | `n8n-io_n8n_Sandbox_Environment` |
| **Key Parameters** | `raw_code: str`, `items: Items`, `write_conn: Connection`, `security_config: SecurityConfig` |
| **Inputs** | User code wrapped in function, input items, pipe for result delivery |
| **Outputs** | Execution result written to pipe via `_put_result()` or `_put_error()` |

### Step 7: Result_Serialization

| Attribute | Value |
|-----------|-------|
| **Principle** | `n8n-io_n8n_Result_Serialization` |
| **Implementation** | `n8n-io_n8n_TaskExecutor_put_result` |
| **API Call** | `TaskExecutor._put_result(write_fd: int, result: Items, print_args: PrintArgs) -> None`, `TaskExecutor._put_error(write_fd: int, e: BaseException, stderr: str = "", print_args: PrintArgs | None = None) -> None` |
| **Source Location** | `packages/@n8n/task-runner-python/src/task_executor.py:L298-351` |
| **External Dependencies** | `json`, `os` |
| **Environment** | `n8n-io_n8n_Sandbox_Environment` |
| **Key Parameters** | `write_fd: int` - file descriptor, `result: Items` - execution output, `print_args: PrintArgs` - captured print calls |
| **Inputs** | Execution result or exception, captured stderr, print arguments |
| **Outputs** | JSON-encoded message with 4-byte length prefix written to pipe |

### Step 8: Result_Delivery

| Attribute | Value |
|-----------|-------|
| **Principle** | `n8n-io_n8n_Result_Delivery` |
| **Implementation** | `n8n-io_n8n_TaskRunner_execute_task` |
| **API Call** | `TaskRunner._execute_task(task_id: str, task_settings: TaskSettings) -> None` (async) |
| **Source Location** | `packages/@n8n/task-runner-python/src/task_runner.py:L302-371` |
| **External Dependencies** | `asyncio` |
| **Environment** | `n8n-io_n8n_Python_Task_Runner_Env` |
| **Key Parameters** | `task_id: str`, `task_settings: TaskSettings` containing code, items, mode |
| **Inputs** | Task ID, task settings, subprocess result |
| **Outputs** | `RunnerTaskDone` or `RunnerTaskError` message sent to broker via WebSocket |

### Implementation Extraction Guide

| Principle | Implementation | API | Source | Type |
|-----------|----------------|-----|--------|------|
| Runner_Initialization | `TaskRunner_init` | `TaskRunner.__init__` | task_runner.py:L74-110 | API Doc |
| Broker_Connection | `TaskRunner_start` | `TaskRunner.start` | task_runner.py:L115-146 | API Doc |
| Task_Offer_Negotiation | `TaskRunner_send_offers` | `_send_offers`, `_handle_task_offer_accept` | task_runner.py:L441-474, L253-280 | API Doc |
| Security_Validation | `TaskAnalyzer_validate` | `TaskAnalyzer.validate` | task_analyzer.py:L172-196 | API Doc |
| Subprocess_Creation | `TaskExecutor_create_process` | `TaskExecutor.create_process` | task_executor.py:L56-86 | API Doc |
| Sandboxed_Execution | `TaskExecutor_execute` | `_all_items`, `_per_item` | task_executor.py:L186-278 | API Doc |
| Result_Serialization | `TaskExecutor_put_result` | `_put_result`, `_put_error` | task_executor.py:L298-351 | API Doc |
| Result_Delivery | `TaskRunner_execute_task` | `_execute_task` | task_runner.py:L302-371 | API Doc |

---

## Workflow: n8n-io_n8n_Security_Validation_Pipeline

**File:** [→](./workflows/n8n-io_n8n_Security_Validation_Pipeline.md)
**Description:** Multi-layered security validation process combining AST-based static analysis with runtime import validation and sandbox enforcement.

### Steps Overview

| # | Step Name | Principle | Rough API | Related Files |
|---|-----------|-----------|-----------|---------------|
| 1 | Configuration Loading | Security_Configuration | `SecurityConfig`, `TaskRunnerConfig.from_env` | security_config.py, task_runner_config.py |
| 2 | Cache Lookup | Validation_Caching | `TaskAnalyzer._to_cache_key`, `OrderedDict` | task_analyzer.py |
| 3 | AST Parsing | AST_Parsing | `ast.parse` | task_analyzer.py |
| 4 | Import Analysis | Import_Analysis | `SecurityValidator.visit_Import`, `SecurityValidator.visit_ImportFrom` | task_analyzer.py |
| 5 | Dangerous Pattern Detection | Dangerous_Pattern_Detection | `SecurityValidator.visit_Attribute`, `SecurityValidator.visit_Name` | task_analyzer.py |
| 6 | Violation Aggregation | Violation_Aggregation | `SecurityValidator.violations`, `SecurityViolationError` | task_analyzer.py, security_violation_error.py |
| 7 | Runtime Import Validation | Runtime_Import_Validation | `TaskExecutor._create_safe_import`, `validate_module_import` | task_executor.py, import_validation.py |
| 8 | Sandbox Environment Setup | Sandbox_Environment | `TaskExecutor._filter_builtins`, `TaskExecutor._sanitize_sys_modules` | task_executor.py |

### Source Files (for enrichment)

- `packages/@n8n/task-runner-python/src/task_analyzer.py` - AST-based security validator (212 lines)
- `packages/@n8n/task-runner-python/src/task_executor.py` - Runtime sandbox setup (506 lines)
- `packages/@n8n/task-runner-python/src/import_validation.py` - Module import validation (37 lines)
- `packages/@n8n/task-runner-python/src/config/security_config.py` - Security config dataclass (9 lines)
- `packages/@n8n/task-runner-python/src/constants.py` - BLOCKED_NAMES, BLOCKED_ATTRIBUTES (193 lines)
- `packages/@n8n/task-runner-python/src/errors/security_violation_error.py` - Error class

### Step 1: Security_Configuration

| Attribute | Value |
|-----------|-------|
| **Principle** | `n8n-io_n8n_Security_Configuration` |
| **Implementation** | `n8n-io_n8n_SecurityConfig` |
| **API Call** | `SecurityConfig(stdlib_allow: set[str], external_allow: set[str], builtins_deny: set[str], runner_env_deny: bool)` |
| **Source Location** | `packages/@n8n/task-runner-python/src/config/security_config.py:L4-9` |
| **External Dependencies** | `dataclasses` |
| **Environment** | `n8n-io_n8n_Python_Task_Runner_Env` |
| **Key Parameters** | `stdlib_allow: set[str]` - allowed stdlib modules, `external_allow: set[str]` - allowed external packages, `builtins_deny: set[str]` - denied builtins, `runner_env_deny: bool` - block env access |
| **Inputs** | Environment variables: `N8N_RUNNERS_STDLIB_ALLOW`, `N8N_RUNNERS_EXTERNAL_ALLOW`, `N8N_RUNNERS_BUILTINS_DENY`, `N8N_BLOCK_RUNNER_ENV_ACCESS` |
| **Outputs** | `SecurityConfig` dataclass instance with security policy settings |

### Step 2: Validation_Caching

| Attribute | Value |
|-----------|-------|
| **Principle** | `n8n-io_n8n_Validation_Caching` |
| **Implementation** | `n8n-io_n8n_TaskAnalyzer_cache` |
| **API Call** | `TaskAnalyzer._to_cache_key(code: str) -> CacheKey`, `TaskAnalyzer._set_in_cache(cache_key: CacheKey, violations: CachedViolations) -> None` |
| **Source Location** | `packages/@n8n/task-runner-python/src/task_analyzer.py:L203-212` |
| **External Dependencies** | `hashlib`, `collections.OrderedDict` |
| **Environment** | `n8n-io_n8n_Python_Task_Runner_Env` |
| **Key Parameters** | `code: str` - code to hash, `MAX_VALIDATION_CACHE_SIZE: 500` - cache limit |
| **Inputs** | Python code string, current allowlists tuple |
| **Outputs** | `CacheKey` tuple of (code_hash, allowlists_tuple), cached validation results |

### Step 3: AST_Parsing

| Attribute | Value |
|-----------|-------|
| **Principle** | `n8n-io_n8n_AST_Parsing` |
| **Implementation** | `n8n-io_n8n_ast_parse` |
| **API Call** | `ast.parse(code: str) -> ast.Module` |
| **Source Location** | `packages/@n8n/task-runner-python/src/task_analyzer.py:L188` (usage) |
| **External Dependencies** | `ast` (Python stdlib) |
| **Environment** | `n8n-io_n8n_Python_Task_Runner_Env` |
| **Key Parameters** | `code: str` - Python source code |
| **Inputs** | Raw Python code string |
| **Outputs** | `ast.Module` AST tree for traversal by `SecurityValidator` |

### Step 4: Import_Analysis

| Attribute | Value |
|-----------|-------|
| **Principle** | `n8n-io_n8n_Import_Analysis` |
| **Implementation** | `n8n-io_n8n_SecurityValidator_imports` |
| **API Call** | `SecurityValidator.visit_Import(node: ast.Import) -> None`, `SecurityValidator.visit_ImportFrom(node: ast.ImportFrom) -> None` |
| **Source Location** | `packages/@n8n/task-runner-python/src/task_analyzer.py:L34-50` |
| **External Dependencies** | `ast` |
| **Environment** | `n8n-io_n8n_Python_Task_Runner_Env` |
| **Key Parameters** | `node: ast.Import` or `node: ast.ImportFrom` - AST import nodes |
| **Inputs** | AST import nodes from traversal |
| **Outputs** | Module names validated against allowlists, violations added to `self.violations` |

### Step 5: Dangerous_Pattern_Detection

| Attribute | Value |
|-----------|-------|
| **Principle** | `n8n-io_n8n_Dangerous_Pattern_Detection` |
| **Implementation** | `n8n-io_n8n_SecurityValidator_patterns` |
| **API Call** | `SecurityValidator.visit_Attribute(node: ast.Attribute) -> None`, `SecurityValidator.visit_Name(node: ast.Name) -> None`, `SecurityValidator.visit_Call(node: ast.Call) -> None`, `SecurityValidator.visit_Subscript(node: ast.Subscript) -> None` |
| **Source Location** | `packages/@n8n/task-runner-python/src/task_analyzer.py:L52-129` |
| **External Dependencies** | `ast` |
| **Environment** | `n8n-io_n8n_Python_Task_Runner_Env` |
| **Key Parameters** | `BLOCKED_NAMES`: `__loader__`, `__builtins__`, `__globals__`, `__spec__`, `__name__`; `BLOCKED_ATTRIBUTES`: 35+ dangerous attributes |
| **Inputs** | AST nodes for attribute access, name access, calls, subscripts |
| **Outputs** | Security violations appended to `self.violations` list |

### Step 6: Violation_Aggregation

| Attribute | Value |
|-----------|-------|
| **Principle** | `n8n-io_n8n_Violation_Aggregation` |
| **Implementation** | `n8n-io_n8n_SecurityViolationError` |
| **API Call** | `SecurityViolationError(message: str = "Security violations detected", description: str = "")` |
| **Source Location** | `packages/@n8n/task-runner-python/src/errors/security_violation_error.py:L1-9` |
| **External Dependencies** | None |
| **Environment** | `n8n-io_n8n_Python_Task_Runner_Env` |
| **Key Parameters** | `message: str` - error message, `description: str` - newline-joined violation details |
| **Inputs** | List of violation strings from `SecurityValidator.violations` |
| **Outputs** | Raised `SecurityViolationError` exception with formatted description |

### Step 7: Runtime_Import_Validation

| Attribute | Value |
|-----------|-------|
| **Principle** | `n8n-io_n8n_Runtime_Import_Validation` |
| **Implementation** | `n8n-io_n8n_validate_module_import` |
| **API Call** | `validate_module_import(module_path: str, security_config: SecurityConfig) -> tuple[bool, str | None]` |
| **Source Location** | `packages/@n8n/task-runner-python/src/import_validation.py:L7-37` |
| **External Dependencies** | `sys` (for `sys.stdlib_module_names`) |
| **Environment** | `n8n-io_n8n_Sandbox_Environment` |
| **Key Parameters** | `module_path: str` - full module path (e.g., `os.path`), `security_config: SecurityConfig` |
| **Inputs** | Module import path, security configuration with allowlists |
| **Outputs** | Tuple of (is_allowed: bool, error_message: str | None) |

### Step 8: Sandbox_Environment

| Attribute | Value |
|-----------|-------|
| **Principle** | `n8n-io_n8n_Sandbox_Environment` |
| **Implementation** | `n8n-io_n8n_TaskExecutor_sandbox` |
| **API Call** | `TaskExecutor._filter_builtins(security_config: SecurityConfig) -> dict`, `TaskExecutor._sanitize_sys_modules(security_config: SecurityConfig) -> None` |
| **Source Location** | `packages/@n8n/task-runner-python/src/task_executor.py:L424-477` |
| **External Dependencies** | `sys`, `os` |
| **Environment** | `n8n-io_n8n_Sandbox_Environment` |
| **Key Parameters** | `security_config.builtins_deny` - builtins to remove, `security_config.stdlib_allow` / `external_allow` - allowed modules |
| **Inputs** | Security configuration with deny/allow lists |
| **Outputs** | Filtered `__builtins__` dict with safe import wrapper, sanitized `sys.modules` |

### Implementation Extraction Guide

| Principle | Implementation | API | Source | Type |
|-----------|----------------|-----|--------|------|
| Security_Configuration | `SecurityConfig` | `SecurityConfig` dataclass | security_config.py:L4-9 | API Doc |
| Validation_Caching | `TaskAnalyzer_cache` | `_to_cache_key`, `_set_in_cache` | task_analyzer.py:L203-212 | API Doc |
| AST_Parsing | `ast_parse` | `ast.parse` | Python stdlib | Wrapper Doc |
| Import_Analysis | `SecurityValidator_imports` | `visit_Import`, `visit_ImportFrom` | task_analyzer.py:L34-50 | API Doc |
| Dangerous_Pattern_Detection | `SecurityValidator_patterns` | `visit_Attribute`, `visit_Name`, `visit_Call`, `visit_Subscript` | task_analyzer.py:L52-129 | API Doc |
| Violation_Aggregation | `SecurityViolationError` | `SecurityViolationError` | security_violation_error.py:L1-9 | API Doc |
| Runtime_Import_Validation | `validate_module_import` | `validate_module_import` | import_validation.py:L7-37 | API Doc |
| Sandbox_Environment | `TaskExecutor_sandbox` | `_filter_builtins`, `_sanitize_sys_modules` | task_executor.py:L424-477 | API Doc |

---

## Workflow: n8n-io_n8n_AI_Workflow_Comparison

**File:** [→](./workflows/n8n-io_n8n_AI_Workflow_Comparison.md)
**Description:** End-to-end process for evaluating AI-generated n8n workflows against ground truth using graph edit distance.

### Steps Overview

| # | Step Name | Principle | Rough API | Related Files |
|---|-----------|-----------|-----------|---------------|
| 1 | Configuration Loading | Comparison_Configuration | `load_config`, `WorkflowComparisonConfig` | config_loader.py |
| 2 | Workflow JSON Parsing | Workflow_Parsing | `json.load`, `load_workflow` | compare_workflows.py |
| 3 | Graph Construction | Graph_Construction | `build_workflow_graph`, `nx.DiGraph` | graph_builder.py |
| 4 | Graph Relabeling | Graph_Relabeling | `_relabel_graph_by_structure`, `nx.relabel_nodes` | similarity.py |
| 5 | Graph Edit Distance Calculation | Graph_Edit_Distance | `calculate_graph_edit_distance`, `nx.optimize_edit_paths` | similarity.py |
| 6 | Edit Operation Extraction | Edit_Operation_Extraction | `_extract_operations_from_path` | similarity.py |
| 7 | Similarity Score Computation | Similarity_Scoring | `_calculate_max_cost`, similarity formula | similarity.py |
| 8 | Priority Assignment | Priority_Assignment | `_determine_priority` | similarity.py |
| 9 | Output Formatting | Output_Formatting | `format_output_json`, `format_output_summary` | compare_workflows.py |

### Source Files (for enrichment)

- `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/compare_workflows.py` - CLI tool (334 lines)
- `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/similarity.py` - GED calculation (501 lines)
- `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/graph_builder.py` - Workflow to graph (222 lines)
- `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/config_loader.py` - Config management (389 lines)
- `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/cost_functions.py` - Cost calculation (497 lines)

### Step 1: Comparison_Configuration

| Attribute | Value |
|-----------|-------|
| **Principle** | `n8n-io_n8n_Comparison_Configuration` |
| **Implementation** | `n8n-io_n8n_load_config` |
| **API Call** | `load_config(config_source: Optional[str] = None) -> WorkflowComparisonConfig` |
| **Source Location** | `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/config_loader.py:L359-389` |
| **External Dependencies** | `yaml`, `json`, `pathlib.Path` |
| **Environment** | `n8n-io_n8n_Workflow_Comparison_Env` |
| **Key Parameters** | `config_source: Optional[str]` - None (default), "preset:name", or file path |
| **Inputs** | Config source string (preset name or file path) |
| **Outputs** | `WorkflowComparisonConfig` instance with cost weights, ignore rules, similarity groups |

### Step 2: Workflow_Parsing

| Attribute | Value |
|-----------|-------|
| **Principle** | `n8n-io_n8n_Workflow_Parsing` |
| **Implementation** | `n8n-io_n8n_load_workflow` |
| **API Call** | `load_workflow(path: str) -> Dict[str, Any]` |
| **Source Location** | `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/compare_workflows.py:L67-91` |
| **External Dependencies** | `json`, `sys` |
| **Environment** | `n8n-io_n8n_Workflow_Comparison_Env` |
| **Key Parameters** | `path: str` - path to workflow JSON file |
| **Inputs** | File path to n8n workflow JSON |
| **Outputs** | Workflow dictionary with `nodes` and `connections` keys |

### Step 3: Graph_Construction

| Attribute | Value |
|-----------|-------|
| **Principle** | `n8n-io_n8n_Graph_Construction` |
| **Implementation** | `n8n-io_n8n_build_workflow_graph` |
| **API Call** | `build_workflow_graph(workflow: Dict[str, Any], config: Optional[WorkflowComparisonConfig] = None) -> nx.DiGraph` |
| **Source Location** | `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/graph_builder.py:L10-90` |
| **External Dependencies** | `networkx` |
| **Environment** | `n8n-io_n8n_Workflow_Comparison_Env` |
| **Key Parameters** | `workflow: Dict[str, Any]` - workflow JSON, `config: Optional[WorkflowComparisonConfig]` - filtering config |
| **Inputs** | Workflow dictionary with nodes/connections, optional configuration |
| **Outputs** | `nx.DiGraph` with nodes (type, parameters, is_trigger) and edges (connection_type, indices) |

### Step 4: Graph_Relabeling

| Attribute | Value |
|-----------|-------|
| **Principle** | `n8n-io_n8n_Graph_Relabeling` |
| **Implementation** | `n8n-io_n8n_relabel_graph_by_structure` |
| **API Call** | `_relabel_graph_by_structure(graph: nx.DiGraph) -> tuple[nx.DiGraph, Dict[str, str]]` |
| **Source Location** | `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/similarity.py:L421-501` |
| **External Dependencies** | `networkx` |
| **Environment** | `n8n-io_n8n_Workflow_Comparison_Env` |
| **Key Parameters** | `graph: nx.DiGraph` - graph to relabel |
| **Inputs** | Original graph with name-based node IDs |
| **Outputs** | Tuple of (relabeled_graph with structural IDs like `trigger_0`, `node_1`, reverse_mapping dict) |

### Step 5: Graph_Edit_Distance

| Attribute | Value |
|-----------|-------|
| **Principle** | `n8n-io_n8n_Graph_Edit_Distance` |
| **Implementation** | `n8n-io_n8n_calculate_graph_edit_distance` |
| **API Call** | `calculate_graph_edit_distance(g1: nx.DiGraph, g2: nx.DiGraph, config: WorkflowComparisonConfig) -> Dict[str, Any]` |
| **Source Location** | `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/similarity.py:L19-144` |
| **External Dependencies** | `networkx` (`nx.optimize_edit_paths`) |
| **Environment** | `n8n-io_n8n_Workflow_Comparison_Env` |
| **Key Parameters** | `g1: nx.DiGraph` - generated workflow graph, `g2: nx.DiGraph` - ground truth graph, `config: WorkflowComparisonConfig` |
| **Inputs** | Two workflow graphs, configuration with cost functions |
| **Outputs** | Dict with `similarity_score`, `edit_cost`, `max_possible_cost`, `top_edits` |

### Step 6: Edit_Operation_Extraction

| Attribute | Value |
|-----------|-------|
| **Principle** | `n8n-io_n8n_Edit_Operation_Extraction` |
| **Implementation** | `n8n-io_n8n_extract_operations_from_path` |
| **API Call** | `_extract_operations_from_path(node_edit_path: List[tuple], edge_edit_path: List[tuple], g1: nx.DiGraph, g2: nx.DiGraph, config: WorkflowComparisonConfig, g1_name_mapping: Dict[str, str], g2_name_mapping: Dict[str, str]) -> List[Dict[str, Any]]` |
| **Source Location** | `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/similarity.py:L223-386` |
| **External Dependencies** | None |
| **Environment** | `n8n-io_n8n_Workflow_Comparison_Env` |
| **Key Parameters** | `node_edit_path: List[tuple]` - (u, v) tuples for node ops, `edge_edit_path: List[tuple]` - edge operations |
| **Inputs** | Edit paths from NetworkX, graphs, mappings |
| **Outputs** | List of operation dicts with `type`, `description`, `cost`, `priority`, `node_name` |

### Step 7: Similarity_Scoring

| Attribute | Value |
|-----------|-------|
| **Principle** | `n8n-io_n8n_Similarity_Scoring` |
| **Implementation** | `n8n-io_n8n_calculate_max_cost` |
| **API Call** | `_calculate_max_cost(g1: nx.DiGraph, g2: nx.DiGraph, config: WorkflowComparisonConfig) -> float` |
| **Source Location** | `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/similarity.py:L195-220` |
| **External Dependencies** | None |
| **Environment** | `n8n-io_n8n_Workflow_Comparison_Env` |
| **Key Parameters** | Uses `config.node_deletion_cost`, `config.node_insertion_cost`, `config.edge_deletion_cost`, `config.edge_insertion_cost` |
| **Inputs** | Two graphs, configuration |
| **Outputs** | Maximum possible cost (delete all g1 + insert all g2), used for formula: `similarity = 1 - (edit_cost / max_cost)` |

### Step 8: Priority_Assignment

| Attribute | Value |
|-----------|-------|
| **Principle** | `n8n-io_n8n_Priority_Assignment` |
| **Implementation** | `n8n-io_n8n_determine_priority` |
| **API Call** | `_determine_priority(cost: float, config: WorkflowComparisonConfig, node_data: Optional[Dict[str, Any]] = None, operation_type: Optional[str] = None) -> str` |
| **Source Location** | `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/similarity.py:L389-418` |
| **External Dependencies** | None |
| **Environment** | `n8n-io_n8n_Workflow_Comparison_Env` |
| **Key Parameters** | `cost: float`, `node_data` (checks `is_trigger`), `operation_type` |
| **Inputs** | Edit operation cost, optional node data, operation type |
| **Outputs** | Priority level string: `'critical'`, `'major'`, or `'minor'` |

### Step 9: Output_Formatting

| Attribute | Value |
|-----------|-------|
| **Principle** | `n8n-io_n8n_Output_Formatting` |
| **Implementation** | `n8n-io_n8n_format_output` |
| **API Call** | `format_output_json(result: Dict[str, Any], metadata: Dict[str, Any], verbose: bool = False) -> str`, `format_output_summary(result: Dict[str, Any], metadata: Dict[str, Any], verbose: bool = False) -> str` |
| **Source Location** | `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/compare_workflows.py:L94-255` |
| **External Dependencies** | `json` |
| **Environment** | `n8n-io_n8n_Workflow_Comparison_Env` |
| **Key Parameters** | `result: Dict[str, Any]` - comparison result, `metadata: Dict[str, Any]` - graph stats, `verbose: bool` - include details |
| **Inputs** | Comparison result dict, metadata about graphs and config |
| **Outputs** | JSON string or human-readable summary string with similarity %, top edits, pass/fail indicator |

### Implementation Extraction Guide

| Principle | Implementation | API | Source | Type |
|-----------|----------------|-----|--------|------|
| Comparison_Configuration | `load_config` | `load_config` | config_loader.py:L359-389 | API Doc |
| Workflow_Parsing | `load_workflow` | `load_workflow` | compare_workflows.py:L67-91 | API Doc |
| Graph_Construction | `build_workflow_graph` | `build_workflow_graph` | graph_builder.py:L10-90 | API Doc |
| Graph_Relabeling | `relabel_graph_by_structure` | `_relabel_graph_by_structure` | similarity.py:L421-501 | API Doc |
| Graph_Edit_Distance | `calculate_graph_edit_distance` | `calculate_graph_edit_distance` | similarity.py:L19-144 | API Doc |
| Edit_Operation_Extraction | `extract_operations_from_path` | `_extract_operations_from_path` | similarity.py:L223-386 | API Doc |
| Similarity_Scoring | `calculate_max_cost` | `_calculate_max_cost` | similarity.py:L195-220 | API Doc |
| Priority_Assignment | `determine_priority` | `_determine_priority` | similarity.py:L389-418 | API Doc |
| Output_Formatting | `format_output` | `format_output_json`, `format_output_summary` | compare_workflows.py:L94-255 | API Doc |

---

**Legend:** `Type:Name` = page exists | `Type:Name` = page needs creation

**Implementation Types:**
- **API Doc:** Function/class in this repo
- **Wrapper Doc:** External library with repo-specific usage
- **Pattern Doc:** User-defined interface/pattern
- **External Tool Doc:** CLI or external tool
