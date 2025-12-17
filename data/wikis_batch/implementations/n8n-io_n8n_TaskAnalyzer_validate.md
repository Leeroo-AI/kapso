{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Automation]], [[domain::Security]], [[domain::Code_Analysis]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Concrete tool for performing static security analysis on Python code before execution, provided by the n8n Python task runner.

=== Description ===

The `validate()` method performs AST (Abstract Syntax Tree) based security validation on user-provided Python code. It uses caching to optimize repeated validations, checks for dangerous imports and attribute access, and raises security violations before any code execution occurs.

=== Usage ===

This implementation is invoked immediately before task execution to ensure user code doesn't violate security policies. It serves as a static analysis firewall that prevents malicious or restricted code from running in the isolated subprocess.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/task_analyzer.py
* '''Lines:''' L172-196

=== Signature ===
<syntaxhighlight lang="python">
def validate(self, code: str) -> None:
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.task_analyzer import TaskAnalyzer
from src.config.security_config import SecurityConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| code || str || Yes || Python source code to validate
|-
| self._security_config || SecurityConfig || Yes || Security policy configuration
|-
| self._cache || OrderedDict || Yes || LRU cache for validation results
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| None || None || Returns silently if validation passes
|-
| SecurityViolationError || Exception || Raised with violation details if validation fails
|}

== Usage Examples ==

=== Basic Validation ===
<syntaxhighlight lang="python">
from src.task_analyzer import TaskAnalyzer
from src.config.security_config import SecurityConfig

# Create analyzer with security policy
security_config = SecurityConfig(
    stdlib_allow=["json", "math", "datetime"],
    external_allow=["numpy", "pandas"],
    builtins_deny=["eval", "exec"],
    runner_env_deny=True
)

analyzer = TaskAnalyzer(security_config)

# Valid code passes silently
safe_code = """
import json
data = json.dumps({"key": "value"})
"""
analyzer.validate(safe_code)  # No exception

# Invalid code raises error
unsafe_code = """
import os
os.system("rm -rf /")
"""
try:
    analyzer.validate(unsafe_code)
except SecurityViolationError as e:
    print(f"Security violation: {e.message}")
    print(f"Details: {e.description}")
</syntaxhighlight>

=== Validation with Caching ===
<syntaxhighlight lang="python">
analyzer = TaskAnalyzer(security_config)

code = "import json\nresult = json.dumps(_items)"

# First validation - parses AST and caches result
analyzer.validate(code)  # AST parsing occurs

# Subsequent validations - uses cache
analyzer.validate(code)  # Cache hit, no AST parsing
analyzer.validate(code)  # Cache hit, no AST parsing
</syntaxhighlight>

=== Allow-All Bypass ===
<syntaxhighlight lang="python">
# When both allowlists use wildcard, validation is skipped
permissive_config = SecurityConfig(
    stdlib_allow=["*"],
    external_allow=["*"],
    builtins_deny=[],
    runner_env_deny=False
)

analyzer = TaskAnalyzer(permissive_config)

# All code passes without AST parsing
analyzer.validate("import os; os.system('echo hello')")  # No validation
</syntaxhighlight>

== Implementation Details ==

=== Early Exit for Permissive Mode ===
<syntaxhighlight lang="python">
def validate(self, code: str) -> None:
    if self._allow_all:
        return  # Skip validation entirely
</syntaxhighlight>

When both `stdlib_allow` and `external_allow` contain "*", validation is bypassed.

=== Cache Key Generation ===
<syntaxhighlight lang="python">
cache_key = self._to_cache_key(code)

def _to_cache_key(self, code: str) -> CacheKey:
    code_hash = hashlib.sha256(code.encode()).hexdigest()
    return (code_hash, self._allowlists)
</syntaxhighlight>

Cache key combines:
* SHA-256 hash of code content
* Sorted tuples of allowlists (stdlib and external)

This ensures cache hits only occur for identical code and security policies.

=== Cache Lookup and Update ===
<syntaxhighlight lang="python">
cached_violations = self._cache.get(cache_key)
cache_hit = cached_violations is not None

if cache_hit:
    self._cache.move_to_end(cache_key)  # LRU update

    if len(cached_violations) == 0:
        return  # Previously validated as safe

    self._raise_security_error(cached_violations)
</syntaxhighlight>

LRU cache behavior:
* Cache hit moves entry to end (most recently used)
* Empty violations list indicates previously validated safe code
* Non-empty violations list replays previous errors

=== AST Parsing and Validation ===
<syntaxhighlight lang="python">
tree = ast.parse(code)

security_validator = SecurityValidator(self._security_config)
security_validator.visit(tree)

self._set_in_cache(cache_key, security_validator.violations)

if security_validator.violations:
    self._raise_security_error(security_validator.violations)
</syntaxhighlight>

The SecurityValidator is an AST visitor that detects:
* Unauthorized imports (stdlib and external modules)
* Dangerous attribute access (__spec__, __globals__, etc.)
* Relative imports
* Dynamic import calls
* Blocked builtins access

=== Cache Size Management ===
<syntaxhighlight lang="python">
def _set_in_cache(self, cache_key: CacheKey, violations: CachedViolations) -> None:
    if len(self._cache) >= MAX_VALIDATION_CACHE_SIZE:
        self._cache.popitem(last=False)  # FIFO eviction

    self._cache[cache_key] = violations.copy()
    self._cache.move_to_end(cache_key)
</syntaxhighlight>

Cache uses OrderedDict with FIFO eviction when max size is reached.

=== Error Reporting ===
<syntaxhighlight lang="python">
def _raise_security_error(self, violations: CachedViolations) -> None:
    raise SecurityViolationError(
        message="Security violations detected",
        description="\n".join(violations)
    )
</syntaxhighlight>

Violations are formatted with line numbers:
```
Line 3: Import not allowed: os
Line 5: Dangerous attribute access: __globals__
```

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Static_Security_Analysis]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Task_Runner]]

=== Related Implementations ===
* [[Implementation:n8n-io_n8n_TaskRunner_execute_task]]
* [[Implementation:n8n-io_n8n_TaskExecutor_create_process]]

=== Security Checks Performed ===
* Import allowlist validation (stdlib and external)
* Blocked attribute access detection
* Relative import prevention
* Dynamic import detection
* Name-mangled attribute blocking
* Dangerous builtin access prevention

=== Performance Optimizations ===
* SHA-256 based cache keying
* LRU cache with configurable size (MAX_VALIDATION_CACHE_SIZE)
* Early exit for permissive mode ("*" allowlists)
* Violation results cached to avoid repeated AST parsing
