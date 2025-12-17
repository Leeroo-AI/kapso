{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Security]], [[domain::Runtime_Protection]], [[domain::Import_Control]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Concrete tool for creating a runtime import validation wrapper that enforces import allowlists during task execution, preventing unauthorized module imports even after AST analysis.

=== Description ===

The `_create_safe_import()` static method in the `TaskExecutor` class creates a closure that wraps Python's built-in `__import__` function. This wrapper intercepts all import operations at runtime, validates the module name against the security configuration's allowlist, and raises a `SecurityViolationError` if the import is not authorized.

=== Usage ===

This implementation is invoked during execution environment setup when creating the filtered builtins dictionary. The safe import wrapper is injected as `__import__` in the execution namespace, ensuring that all import operations—whether from explicit import statements, `importlib` calls, or dynamic `__import__()` usage—are validated before execution.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/task_executor.py
* '''Lines:''' L479-495

=== Signature ===
<syntaxhighlight lang="python">
@staticmethod
def _create_safe_import(security_config: SecurityConfig) -> Callable:
    """Create a wrapped __import__ function that validates against allowlists."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.task_executor import TaskExecutor
from src.security_config import SecurityConfig
from src.exceptions import SecurityViolationError
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| security_config || SecurityConfig || Yes || Configuration containing import_allowlist and module validation settings
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| safe_import || Callable || Wrapper function matching __import__ signature that validates imports
|}

== Implementation Details ==

=== Complete Implementation ===
<syntaxhighlight lang="python">
@staticmethod
def _create_safe_import(security_config: SecurityConfig):
    original_import = __builtins__["__import__"]

    def safe_import(name, *args, **kwargs):
        is_allowed, error_msg = validate_module_import(name, security_config)
        if not is_allowed:
            raise SecurityViolationError(
                message="Security violation detected",
                description=error_msg
            )
        return original_import(name, *args, **kwargs)

    return safe_import
</syntaxhighlight>

=== Closure Mechanism ===
The function uses a closure to capture both `original_import` and `security_config`:

<syntaxhighlight lang="python">
# Outer function captures security_config
def _create_safe_import(security_config: SecurityConfig):
    # Captures original __import__ from builtins
    original_import = __builtins__["__import__"]

    # Inner function (closure) has access to both captured variables
    def safe_import(name, *args, **kwargs):
        # Uses security_config and original_import
        ...

    # Returns inner function
    return safe_import
</syntaxhighlight>

This design allows:
* Security config to be set once at creation time
* Original import function to be preserved
* No global state modification
* Thread-safe operation (each execution gets its own closure)

=== Import Validation Logic ===
The wrapper calls `validate_module_import()`:
<syntaxhighlight lang="python">
is_allowed, error_msg = validate_module_import(name, security_config)
</syntaxhighlight>

This function performs:
* Top-level module extraction (e.g., "os.path" → "os")
* Allowlist membership check
* Error message generation

Return values:
* `(True, "")`: Module is allowed, proceed with import
* `(False, "Module 'x' is not in the allowlist")`: Module denied, raise exception

=== Exception Handling ===
If validation fails:
<syntaxhighlight lang="python">
raise SecurityViolationError(
    message="Security violation detected",
    description=error_msg
)
</syntaxhighlight>

This exception propagates up, terminating task execution and sending the error back to the n8n broker.

=== Original Import Delegation ===
If validation succeeds, the wrapper delegates to the original `__import__`:
<syntaxhighlight lang="python">
return original_import(name, *args, **kwargs)
</syntaxhighlight>

This preserves all standard import behavior:
* Module caching in `sys.modules`
* Package initialization
* Relative imports (though blocked by AST analysis)
* Custom import hooks

=== __import__ Signature ===
The safe wrapper matches the built-in `__import__` signature:
<syntaxhighlight lang="python">
def __import__(
    name,           # Module name (str)
    globals=None,   # Global namespace (dict)
    locals=None,    # Local namespace (dict)
    fromlist=(),    # Names to import from module (tuple)
    level=0         # Relative import level (int)
):
    ...
</syntaxhighlight>

Using `*args, **kwargs` ensures compatibility with all parameters without explicit declaration.

== Usage Examples ==

=== Basic Import Interception ===
<syntaxhighlight lang="python">
from src.task_executor import TaskExecutor
from src.security_config import SecurityConfig
from src.exceptions import SecurityViolationError

# Configure allowed modules
security_config = SecurityConfig()
security_config.import_allowlist = ["json", "math"]

# Create safe import wrapper
safe_import = TaskExecutor._create_safe_import(security_config)

# Allowed import succeeds
try:
    json_module = safe_import("json")
    print(f"Imported: {json_module.__name__}")  # "json"
except SecurityViolationError as e:
    print(f"Import blocked: {e.description}")

# Disallowed import fails
try:
    os_module = safe_import("os")
except SecurityViolationError as e:
    print(f"Import blocked: {e.description}")
    # Output: Import blocked: Module 'os' is not in the allowlist
</syntaxhighlight>

=== Integration with Execution Environment ===
<syntaxhighlight lang="python">
from src.task_executor import TaskExecutor
from src.security_config import SecurityConfig

security_config = SecurityConfig()
security_config.import_allowlist = ["pandas", "numpy"]

# Create execution namespace with safe import
namespace = {
    "__builtins__": {
        **{k: v for k, v in __builtins__.items()},
        "__import__": TaskExecutor._create_safe_import(security_config)
    }
}

# User code with imports
task_code = """
import pandas as pd
import numpy as np

data = pd.DataFrame({"x": [1, 2, 3]})
result = np.mean(data["x"])
"""

# Execute with runtime import validation
exec(task_code, namespace)
print(f"Result: {namespace['result']}")  # 2.0
</syntaxhighlight>

=== Dynamic Import Interception ===
<syntaxhighlight lang="python">
from src.task_executor import TaskExecutor
from src.security_config import SecurityConfig
from src.exceptions import SecurityViolationError

security_config = SecurityConfig()
security_config.import_allowlist = ["json"]

namespace = {
    "__builtins__": {
        "__import__": TaskExecutor._create_safe_import(security_config)
    }
}

# User code attempting dynamic import
task_code = """
# Even dynamic imports are caught
module_name = "os"
try:
    imported = __import__(module_name)
except Exception as e:
    error_caught = str(e)
"""

exec(task_code, namespace)
assert "not in the allowlist" in namespace["error_caught"]
</syntaxhighlight>

=== Submodule Import Validation ===
<syntaxhighlight lang="python">
from src.task_executor import TaskExecutor
from src.security_config import SecurityConfig

security_config = SecurityConfig()
security_config.import_allowlist = ["pandas"]

safe_import = TaskExecutor._create_safe_import(security_config)

# Submodule imports are allowed if parent is allowed
try:
    # Import pandas.core.frame - validates "pandas" (top-level)
    submodule = safe_import("pandas.core.frame")
    print("Submodule import succeeded")
except SecurityViolationError:
    print("Submodule import blocked")
</syntaxhighlight>

=== FromList Parameter Handling ===
<syntaxhighlight lang="python">
from src.task_executor import TaskExecutor
from src.security_config import SecurityConfig

security_config = SecurityConfig()
security_config.import_allowlist = ["os"]

safe_import = TaskExecutor._create_safe_import(security_config)

# from os import path, environ
# Translates to: __import__("os", fromlist=["path", "environ"])
module = safe_import("os", fromlist=["path", "environ"])

# Returns the os module with requested attributes
assert hasattr(module, "path")
assert hasattr(module, "environ")
</syntaxhighlight>

=== Error Message Inspection ===
<syntaxhighlight lang="python">
from src.task_executor import TaskExecutor
from src.security_config import SecurityConfig
from src.exceptions import SecurityViolationError

security_config = SecurityConfig()
security_config.import_allowlist = ["json"]

safe_import = TaskExecutor._create_safe_import(security_config)

# Attempt unauthorized import
try:
    safe_import("subprocess")
except SecurityViolationError as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Message: {e.message}")
    print(f"Description: {e.description}")

    # Output:
    # Exception type: SecurityViolationError
    # Message: Security violation detected
    # Description: Module 'subprocess' is not in the allowlist
</syntaxhighlight>

=== Multiple Safe Import Instances ===
<syntaxhighlight lang="python">
from src.task_executor import TaskExecutor
from src.security_config import SecurityConfig

# Different configs for different execution contexts
strict_config = SecurityConfig()
strict_config.import_allowlist = ["json"]

permissive_config = SecurityConfig()
permissive_config.import_allowlist = ["json", "os", "sys"]

# Create separate safe import wrappers
strict_import = TaskExecutor._create_safe_import(strict_config)
permissive_import = TaskExecutor._create_safe_import(permissive_config)

# Same import, different results
try:
    strict_import("os")
    print("Strict: os allowed")
except SecurityViolationError:
    print("Strict: os blocked")

try:
    permissive_import("os")
    print("Permissive: os allowed")
except SecurityViolationError:
    print("Permissive: os blocked")

# Output:
# Strict: os blocked
# Permissive: os allowed
</syntaxhighlight>

== Security Rationale ==

=== Why Runtime Validation is Necessary ===
Runtime import validation complements AST analysis by catching:

'''Dynamic Imports:'''
<syntaxhighlight lang="python">
# AST analysis sees string concatenation, not import
module_name = "o" + "s"
__import__(module_name)  # Caught by safe_import at runtime
</syntaxhighlight>

'''Importlib Usage:'''
<syntaxhighlight lang="python">
import importlib  # If allowed
# Uses __import__ internally
importlib.import_module("os")  # Caught by safe_import
</syntaxhighlight>

'''Conditional Imports:'''
<syntaxhighlight lang="python">
# AST sees both branches, but only one executes
if some_condition:
    import safe_module
else:
    import dangerous_module  # Only caught at runtime if condition is False
</syntaxhighlight>

=== Defense in Depth ===
The safe import wrapper is the third security layer:

1. '''AST Analysis''' (`SecurityValidator`): Catches explicit, static imports
2. '''Builtin Filtering''' (`_filter_builtins`): Removes dangerous built-ins
3. '''Runtime Import Validation''' (`_create_safe_import`): Catches dynamic imports

All three layers are necessary for comprehensive protection.

=== Why Preserve Original Import ===
The wrapper delegates to `original_import` rather than reimplementing import logic because:
* Python's import system is complex (PEP 302, import hooks, etc.)
* Reimplementation would miss edge cases
* Original import handles all the machinery (sys.modules cache, package init, etc.)
* Maintains compatibility with standard library expectations

=== Thread Safety ===
The closure-based design is thread-safe:
* Each task execution gets its own `safe_import` instance
* `security_config` is immutable during execution
* `original_import` is a reference to the built-in (thread-safe)
* No shared mutable state

== Performance Characteristics ==

=== Time Complexity ===
* '''Validation''': O(1) for allowlist lookup (set/dict membership)
* '''Import delegation''': Same as standard `__import__` (varies by module)
* '''Overhead per import''': ~10-50 microseconds for validation

=== Space Complexity ===
* '''Closure''': O(1) - captures two references
* '''No memory leak''': Closure is garbage collected with namespace

=== Performance Impact ===
The validation overhead is negligible compared to actual import time:
* '''Module import''': 1-100ms (file I/O, parsing, execution)
* '''Validation''': <0.1ms (set lookup)
* '''Overhead''': <1% of import time

=== Caching Behavior ===
Python's import system caches modules in `sys.modules`:
* First import: Full cost (validation + import)
* Subsequent imports: Validation only (import cached)
* Validation still occurs every time for security

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Runtime_Import_Validation]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Task_Runner]]

=== Related Implementations ===
* [[related::Implementation:n8n-io_n8n_TaskExecutor_filter_builtins]]
* [[related::Implementation:n8n-io_n8n_SecurityValidator_visit_Import]]

=== Used By Workflow ===
* [[used_by::Workflow:n8n-io_n8n_Security_Validation]]

=== Security References ===
* [https://docs.python.org/3/library/functions.html#__import__ Python __import__ Documentation]
* [https://peps.python.org/pep-0302/ PEP 302: New Import Hooks]
