{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Security]], [[domain::Runtime_Protection]], [[domain::Sandbox]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Concrete tool for creating a filtered builtins dictionary that removes dangerous built-in functions and replaces `__import__` with a security-validated version.

=== Description ===

The `_filter_builtins()` static method in the `TaskExecutor` class creates a modified copy of Python's `__builtins__` dictionary with security-sensitive functions removed according to the `security_config.builtins_deny` list. It also replaces the standard `__import__` function with a safe wrapper that enforces import allowlists at runtime.

=== Usage ===

This implementation is invoked during task execution environment setup, before any user code runs. The filtered builtins dictionary is provided as the `__builtins__` in the execution namespace, creating a runtime security layer that complements the static AST analysis. This prevents code from accessing dangerous built-ins even if they weren't explicitly called in the analyzed code.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/task_executor.py
* '''Lines:''' L424-439

=== Signature ===
<syntaxhighlight lang="python">
@staticmethod
def _filter_builtins(security_config: SecurityConfig) -> dict[str, Any]:
    """Get __builtins__ with denied ones removed."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.task_executor import TaskExecutor
from src.security_config import SecurityConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| security_config || SecurityConfig || Yes || Configuration containing builtins_deny list and import allowlists
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| filtered_builtins || dict[str, Any] || Dictionary of built-in functions with dangerous ones removed and __import__ replaced
|}

== Implementation Details ==

=== Complete Implementation ===
<syntaxhighlight lang="python">
@staticmethod
def _filter_builtins(security_config: SecurityConfig):
    """Get __builtins__ with denied ones removed."""
    if len(security_config.builtins_deny) == 0:
        filtered = dict(__builtins__)
    else:
        filtered = {
            k: v
            for k, v in __builtins__.items()
            if k not in security_config.builtins_deny
        }
    filtered["__import__"] = TaskExecutor._create_safe_import(security_config)
    return filtered
</syntaxhighlight>

=== Filtering Logic ===

'''No Deny List:'''
<syntaxhighlight lang="python">
if len(security_config.builtins_deny) == 0:
    filtered = dict(__builtins__)
</syntaxhighlight>
When no builtins are denied, creates a shallow copy of the entire `__builtins__` dictionary. This optimization avoids dictionary comprehension when not needed.

'''With Deny List:'''
<syntaxhighlight lang="python">
filtered = {
    k: v
    for k, v in __builtins__.items()
    if k not in security_config.builtins_deny
}
</syntaxhighlight>
Dictionary comprehension filters out all built-ins whose names appear in `builtins_deny`. This is an O(n) operation where n is the number of built-ins (~150 in CPython 3.x).

=== Safe Import Injection ===
<syntaxhighlight lang="python">
filtered["__import__"] = TaskExecutor._create_safe_import(security_config)
</syntaxhighlight>

After filtering, `__import__` is always replaced with the safe wrapper, regardless of whether it was in the deny list. This ensures that even indirect imports (via `importlib`, `__import__()` calls, etc.) go through validation.

=== Common Denied Builtins ===
Typical `builtins_deny` configurations include:
<syntaxhighlight lang="python">
builtins_deny = [
    "eval",          # Dynamic code evaluation
    "exec",          # Dynamic code execution
    "compile",       # Bytecode compilation
    "open",          # File system access
    "__import__",    # Direct import (replaced anyway)
    "breakpoint",    # Debugger access
    "input",         # User input (can cause hangs)
    "exit",          # Process termination
    "quit",          # Process termination
    "help",          # Documentation system
    "license",       # License text display
    "copyright",     # Copyright text display
]
</syntaxhighlight>

=== Execution Environment Setup ===
The filtered builtins are used when creating the execution namespace:
<syntaxhighlight lang="python">
def _execute_task(self, task_code: str, security_config: SecurityConfig):
    # Create isolated namespace
    namespace = {
        "__builtins__": TaskExecutor._filter_builtins(security_config),
        "__name__": "__main__",
    }

    # Execute user code in controlled environment
    exec(task_code, namespace)
</syntaxhighlight>

=== Static Method Design ===
The method is static because:
* It doesn't access instance state
* It can be called before TaskExecutor instantiation
* It's a pure function: same inputs → same outputs
* It can be tested independently

== Usage Examples ==

=== Basic Filtering ===
<syntaxhighlight lang="python">
from src.task_executor import TaskExecutor
from src.security_config import SecurityConfig

# Configure denied builtins
security_config = SecurityConfig()
security_config.builtins_deny = ["eval", "exec", "open"]

# Get filtered builtins
filtered = TaskExecutor._filter_builtins(security_config)

# Check removals
assert "eval" not in filtered
assert "exec" not in filtered
assert "open" not in filtered

# Safe builtins still present
assert "print" in filtered
assert "len" in filtered
assert "range" in filtered

# __import__ is replaced
assert "__import__" in filtered
assert filtered["__import__"].__name__ == "safe_import"
</syntaxhighlight>

=== Execution with Filtered Builtins ===
<syntaxhighlight lang="python">
from src.task_executor import TaskExecutor
from src.security_config import SecurityConfig

security_config = SecurityConfig()
security_config.builtins_deny = ["eval", "exec"]

# User code attempting to use eval
task_code = """
result = eval("2 + 2")  # This will fail at runtime
"""

# Setup execution environment
namespace = {
    "__builtins__": TaskExecutor._filter_builtins(security_config)
}

try:
    exec(task_code, namespace)
except NameError as e:
    print(f"Runtime error: {e}")
    # Output: Runtime error: name 'eval' is not defined
</syntaxhighlight>

=== No Deny List Optimization ===
<syntaxhighlight lang="python">
from src.task_executor import TaskExecutor
from src.security_config import SecurityConfig

security_config = SecurityConfig()
security_config.builtins_deny = []  # Empty deny list

filtered = TaskExecutor._filter_builtins(security_config)

# All builtins present except __import__ is replaced
original_builtins = set(__builtins__.keys())
filtered_builtins = set(filtered.keys())

# Same keys (though __import__ is replaced)
assert original_builtins == filtered_builtins
</syntaxhighlight>

=== Combining with Safe Import ===
<syntaxhighlight lang="python">
from src.task_executor import TaskExecutor
from src.security_config import SecurityConfig
from src.exceptions import SecurityViolationError

security_config = SecurityConfig()
security_config.builtins_deny = []
security_config.import_allowlist = ["json"]

task_code = """
# Try to import disallowed module at runtime
import os  # Will be caught by safe_import
"""

namespace = {
    "__builtins__": TaskExecutor._filter_builtins(security_config)
}

try:
    exec(task_code, namespace)
except SecurityViolationError as e:
    print(f"Import blocked: {e.description}")
    # Output: Import blocked: Module 'os' is not in the allowlist
</syntaxhighlight>

=== Testing Filtered Namespace ===
<syntaxhighlight lang="python">
from src.task_executor import TaskExecutor
from src.security_config import SecurityConfig

security_config = SecurityConfig()
security_config.builtins_deny = ["print", "len"]

task_code = """
# Safe operations work
x = 1 + 2
y = [1, 2, 3]

# Denied builtins don't work
try:
    print(x)  # NameError
except NameError:
    pass

try:
    size = len(y)  # NameError
except NameError:
    pass

result = "success"
"""

namespace = {
    "__builtins__": TaskExecutor._filter_builtins(security_config)
}

exec(task_code, namespace)
assert namespace["result"] == "success"
</syntaxhighlight>

=== Checking Builtin Count ===
<syntaxhighlight lang="python">
from src.task_executor import TaskExecutor
from src.security_config import SecurityConfig

security_config = SecurityConfig()
security_config.builtins_deny = ["eval", "exec", "compile", "open"]

filtered = TaskExecutor._filter_builtins(security_config)

original_count = len(__builtins__)
filtered_count = len(filtered)

# Should have removed exactly the denied ones
assert filtered_count == original_count - len(security_config.builtins_deny)
</syntaxhighlight>

== Security Rationale ==

=== Defense in Depth ===
Builtin filtering provides runtime protection that complements AST analysis:
* '''AST analysis''': Catches explicit calls to dangerous functions
* '''Builtin filtering''': Prevents indirect access or dynamically constructed calls

Example of indirect access:
<syntaxhighlight lang="python">
# AST analysis might miss this
dangerous_func = getattr(__builtins__, "eval")
dangerous_func("malicious_code")

# But builtin filtering prevents it:
# getattr(__builtins__, "eval") raises AttributeError
</syntaxhighlight>

=== Why Filter vs Override ===
Filtering (removing) is safer than overriding (replacing with dummy functions):
* '''NameError''' is clear: function doesn't exist
* No risk of incomplete dummy implementation
* No way to detect and bypass dummy
* Simpler security model

=== Why Always Replace __import__ ===
Even if `__import__` is in the deny list, it must be replaced (not removed) because:
* Python's import system needs `__import__` internally
* Import statements call `__import__` behind the scenes
* Removing it breaks all imports, including allowed ones

=== Immutability Consideration ===
The filtered dictionary is a new object, not a modification of `__builtins__`:
<syntaxhighlight lang="python">
filtered = dict(__builtins__)  # Create new dict
# Original __builtins__ unchanged
</syntaxhighlight>

This prevents accidental global state modification.

== Performance Characteristics ==

=== Time Complexity ===
* '''No deny list''': O(n) for shallow copy where n = number of builtins (~150)
* '''With deny list''': O(n) for dictionary comprehension + O(d) for deny list lookups where d = deny list size
* '''Overall''': O(n) linear in builtin count

=== Space Complexity ===
* '''O(n)''' for filtered dictionary
* '''Dictionary overhead''': ~240 bytes + 24 bytes per entry in CPython

=== Optimization Notes ===
The optimization for empty deny lists:
<syntaxhighlight lang="python">
if len(security_config.builtins_deny) == 0:
    filtered = dict(__builtins__)
</syntaxhighlight>

Saves the overhead of dictionary comprehension and membership checks when no filtering is needed, though the performance difference is minimal (<1ms on modern hardware).

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Builtin_Filtering]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Task_Runner]]

=== Related Implementations ===
* [[related::Implementation:n8n-io_n8n_TaskExecutor_create_safe_import]]

=== Used By Workflow ===
* [[used_by::Workflow:n8n-io_n8n_Security_Validation]]
