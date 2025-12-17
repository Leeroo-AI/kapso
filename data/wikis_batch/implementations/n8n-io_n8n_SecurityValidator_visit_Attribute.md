{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Security]], [[domain::Code_Analysis]], [[domain::Pattern_Detection]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Concrete tool for detecting dangerous attribute access patterns and blocked name usage in Python code through AST traversal.

=== Description ===

The `SecurityValidator` implements two visitor methods: `visit_Attribute()` and `visit_Name()`. These methods detect security risks by identifying access to blocked attributes (like `__globals__`, `__code__`), name-mangled attributes, and usage of dangerous built-in names (like `eval`, `exec`).

=== Usage ===

This implementation is invoked during AST traversal after code is parsed with `ast.parse()`. It serves as a static analysis layer to detect attempts to bypass security restrictions through reflection, introspection, or direct use of dangerous built-ins. When suspicious patterns are detected, violations are recorded for reporting before task execution begins.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/task_analyzer.py
* '''Lines:''' L52-71

=== Signature ===
<syntaxhighlight lang="python">
def visit_Attribute(self, node: ast.Attribute) -> None:
    """Detect access to unsafe attributes."""

def visit_Name(self, node: ast.Name) -> None:
    """Detect usage of dangerous names."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.task_analyzer import SecurityValidator, BLOCKED_ATTRIBUTES, BLOCKED_NAMES
import ast
</syntaxhighlight>

== I/O Contract ==

=== Inputs (visit_Attribute) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| node || ast.Attribute || Yes || AST node representing attribute access (e.g., obj.attr)
|}

=== Inputs (visit_Name) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| node || ast.Name || Yes || AST node representing a name reference (variable, function, etc.)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| (none) || None || Methods update internal violations list as side effect
|}

=== Side Effects ===
* Appends violation strings to `self.violations` list when dangerous patterns detected
* Calls `self.generic_visit(node)` to continue AST traversal

== Implementation Details ==

=== visit_Attribute Implementation ===
<syntaxhighlight lang="python">
def visit_Attribute(self, node: ast.Attribute) -> None:
    """Detect access to unsafe attributes"""
    if node.attr in BLOCKED_ATTRIBUTES:
        self._add_violation(node.lineno, ERROR_DANGEROUS_ATTRIBUTE.format(attr=node.attr))
    if node.attr.startswith("_") and "__" in node.attr:
        # Name mangling detection
        self._add_violation(node.lineno, ERROR_NAME_MANGLED_ATTRIBUTE)
    self.generic_visit(node)
</syntaxhighlight>

This method handles attribute access like:
<syntaxhighlight lang="python">
obj.__globals__      # Blocked attribute
func.__code__        # Blocked attribute
cls._ClassName__attr # Name mangling
</syntaxhighlight>

Key behaviors:
* '''Direct blocking''': Checks `node.attr` against `BLOCKED_ATTRIBUTES` list
* '''Pattern detection''': Identifies name-mangled attributes via string pattern
* '''Continuation''': Calls `generic_visit()` to traverse nested structures

=== visit_Name Implementation ===
<syntaxhighlight lang="python">
def visit_Name(self, node: ast.Name) -> None:
    if node.id in BLOCKED_NAMES:
        self._add_violation(node.lineno, ERROR_DANGEROUS_NAME.format(name=node.id))
    self.generic_visit(node)
</syntaxhighlight>

This method handles name references like:
<syntaxhighlight lang="python">
eval(code)          # Blocked name
exec(code)          # Blocked name
__import__('os')    # Blocked name
</syntaxhighlight>

Key behaviors:
* '''Name checking''': Compares `node.id` against `BLOCKED_NAMES` list
* '''Context-agnostic''': Blocks usage in any context (call, assignment, etc.)

=== Blocked Attributes ===
The `BLOCKED_ATTRIBUTES` constant typically includes:
<syntaxhighlight lang="python">
BLOCKED_ATTRIBUTES = [
    "__globals__",    # Access to global namespace
    "__code__",       # Access to function bytecode
    "__dict__",       # Direct access to object namespace
    "__class__",      # Type introspection
    "__bases__",      # Class hierarchy access
    "__subclasses__", # Subclass enumeration
    "__builtins__",   # Access to built-in functions
    "func_globals",   # Legacy global access
    "func_code",      # Legacy code access
]
</syntaxhighlight>

=== Blocked Names ===
The `BLOCKED_NAMES` constant typically includes:
<syntaxhighlight lang="python">
BLOCKED_NAMES = [
    "eval",           # Dynamic code evaluation
    "exec",           # Dynamic code execution
    "__import__",     # Direct import function
    "compile",        # Code compilation
    "open",           # File system access
]
</syntaxhighlight>

=== Name Mangling Detection ===
Python name mangling transforms `_ClassName__attr` to make private attributes harder to access. The detection logic:
<syntaxhighlight lang="python">
if node.attr.startswith("_") and "__" in node.attr:
    # Matches patterns like: _MyClass__private_attr
    self._add_violation(...)
</syntaxhighlight>

This prevents bypassing private attribute restrictions through mangling.

=== Error Messages ===
The implementation uses predefined error constants:
* '''ERROR_DANGEROUS_ATTRIBUTE''': "Access to dangerous attribute '{attr}' is not allowed"
* '''ERROR_NAME_MANGLED_ATTRIBUTE''': "Access to name-mangled attributes is not allowed"
* '''ERROR_DANGEROUS_NAME''': "Usage of dangerous name '{name}' is not allowed"

== Usage Examples ==

=== Detecting Dangerous Attributes ===
<syntaxhighlight lang="python">
import ast
from src.task_analyzer import SecurityValidator
from src.security_config import SecurityConfig

code = """
def get_globals(func):
    return func.__globals__  # Attempt to access global namespace

def get_code(func):
    return func.__code__.co_consts  # Attempt to access bytecode
"""

tree = ast.parse(code)
validator = SecurityValidator(SecurityConfig())
validator.visit(tree)

print(f"Violations found: {len(validator.violations)}")  # 2
for violation in validator.violations:
    print(f"  {violation}")
# Output:
# Line 2: Access to dangerous attribute '__globals__' is not allowed
# Line 5: Access to dangerous attribute '__code__' is not allowed
</syntaxhighlight>

=== Detecting Name Mangling ===
<syntaxhighlight lang="python">
import ast
from src.task_analyzer import SecurityValidator

code = """
class MyClass:
    def __init__(self):
        self.__private = 42

obj = MyClass()
# Attempt to bypass private attribute protection
value = obj._MyClass__private
"""

tree = ast.parse(code)
validator = SecurityValidator(SecurityConfig())
validator.visit(tree)

assert len(validator.violations) == 1
assert "name-mangled" in validator.violations[0]
</syntaxhighlight>

=== Detecting Dangerous Names ===
<syntaxhighlight lang="python">
import ast
from src.task_analyzer import SecurityValidator

code = """
# Attempts to execute dynamic code
result = eval("1 + 1")
exec("print('hello')")
imported = __import__('os')
"""

tree = ast.parse(code)
validator = SecurityValidator(SecurityConfig())
validator.visit(tree)

print(f"Violations: {len(validator.violations)}")  # 3
# Each dangerous name usage is caught
</syntaxhighlight>

=== Complex Attribute Access Chain ===
<syntaxhighlight lang="python">
import ast
from src.task_analyzer import SecurityValidator

code = """
# Nested attribute access
globals_dict = some_func.__globals__['__builtins__']
"""

tree = ast.parse(code)
validator = SecurityValidator(SecurityConfig())
validator.visit(tree)

# Both __globals__ and __builtins__ are blocked
assert len(validator.violations) >= 1
</syntaxhighlight>

=== Safe Attribute Access ===
<syntaxhighlight lang="python">
import ast
from src.task_analyzer import SecurityValidator

code = """
# Safe attribute access patterns
class Data:
    def __init__(self):
        self.value = 42
        self.name = "test"
        self._internal = 123  # Single underscore is fine

obj = Data()
print(obj.value)
print(obj._internal)
"""

tree = ast.parse(code)
validator = SecurityValidator(SecurityConfig())
validator.visit(tree)

# No violations - normal attribute access is allowed
assert len(validator.violations) == 0
</syntaxhighlight>

=== Introspection Prevention ===
<syntaxhighlight lang="python">
import ast
from src.task_analyzer import SecurityValidator

code = """
# Attempt to enumerate all subclasses (common exploit vector)
class MyClass:
    pass

all_subclasses = MyClass.__subclasses__()

# Attempt to access class dictionary
MyClass.__dict__['__init__']
"""

tree = ast.parse(code)
validator = SecurityValidator(SecurityConfig())
validator.visit(tree)

print(f"Violations: {len(validator.violations)}")  # 2
# Both __subclasses__ and __dict__ are blocked
</syntaxhighlight>

== Security Rationale ==

=== Why Block These Attributes ===

'''Reflection and Introspection:'''
* `__globals__`: Provides access to the global namespace, bypassing import restrictions
* `__code__`: Allows bytecode inspection and modification
* `__dict__`: Direct namespace manipulation
* `__class__`, `__bases__`: Type system traversal to access restricted classes

'''Common Exploit Vectors:'''
<syntaxhighlight lang="python">
# Without blocking __globals__, attacker could:
().__class__.__bases__[0].__subclasses__()[104].__init__.__globals__['sys'].modules

# This chain can reach any module regardless of import restrictions
</syntaxhighlight>

=== Why Block These Names ===

'''Dynamic Code Execution:'''
* `eval()`: Executes arbitrary Python expressions
* `exec()`: Executes arbitrary Python statements
* `compile()`: Creates code objects that can bypass analysis

'''Import System Bypass:'''
* `__import__()`: Direct import function that bypasses `_create_safe_import()` wrapper

'''File System Access:'''
* `open()`: Direct file access bypassing any I/O restrictions

=== Why Detect Name Mangling ===
Python name mangling transforms `_ClassName__attr` to prevent accidental access to private attributes. Detecting this pattern prevents:
* Intentional bypass of private attribute protection
* Access to implementation details that may expose vulnerabilities
* Circumvention of class-level security boundaries

=== Defense in Depth ===
These AST checks provide **static analysis** protection. They complement:
* `_filter_builtins()`: Runtime removal of dangerous built-ins
* `_create_safe_import()`: Runtime import validation
* Allowlist validation: Module-level access control

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Pattern_Detection]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Task_Runner]]

=== Related Implementations ===
* [[related::Implementation:n8n-io_n8n_ast_parse]]
* [[related::Implementation:n8n-io_n8n_SecurityValidator_visit_Import]]
* [[related::Implementation:n8n-io_n8n_TaskExecutor_filter_builtins]]

=== Used By Workflow ===
* [[used_by::Workflow:n8n-io_n8n_Security_Validation]]

=== Security References ===
* [https://docs.python.org/3/reference/datamodel.html#special-method-names Python Special Attributes]
* [https://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html Why eval() is Dangerous]
