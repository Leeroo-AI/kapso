# File: `packages/@n8n/task-runner-python/src/import_validation.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 37 |
| Functions | `validate_module_import` |
| Imports | src, sys |

## Understanding

**Status:** ✅ Explored

**Purpose:** Runtime module import validation

**Mechanism:** Validates module imports against allowlists:
1. Takes module path (e.g., "os.path") and security config
2. Extracts root module name (e.g., "os" from "os.path")
3. Determines if module is stdlib using sys.stdlib_module_names
4. Checks if module is in stdlib_allow or external_allow lists
5. Supports wildcard "*" to allow all modules in a category
6. Returns (True, None) if allowed, (False, error_message) if denied
7. Generates helpful error messages listing allowed modules

**Significance:** This is the runtime enforcement of import restrictions, called by TaskExecutor's safe_import hook. Works in conjunction with TaskAnalyzer's static analysis to provide defense-in-depth. The stdlib vs external distinction allows fine-grained control (e.g., allow numpy but not os). The wildcard support enables permissive mode for trusted environments. The error messages help users understand which modules are available in their environment.
