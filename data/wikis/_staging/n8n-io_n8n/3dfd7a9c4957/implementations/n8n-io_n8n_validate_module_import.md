# Implementation: validate_module_import

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Security]], [[domain::Runtime_Enforcement]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Concrete function for validating module imports against security configuration, used by both static analysis and runtime import wrapper.

=== Description ===

`validate_module_import()` is the core validation function that:

1. Extracts root module name from full path (`"os.path"` → `"os"`)
2. Classifies module as stdlib or external using `sys.stdlib_module_names`
3. Checks against appropriate allowlist (stdlib_allow or external_allow)
4. Returns tuple of (is_allowed, error_message) for caller to handle

This function is called from:
- `SecurityValidator._validate_import()` - during static AST analysis
- `TaskExecutor._create_safe_import()` - during runtime execution

=== Usage ===

Call this function to check if a module import should be permitted under current security configuration.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/import_validation.py
* '''Lines:''' L7-37

=== Signature ===
<syntaxhighlight lang="python">
def validate_module_import(
    module_path: str,
    security_config: SecurityConfig,
) -> tuple[bool, str | None]:
    """
    Validate that a module import is allowed under security configuration.

    Args:
        module_path: Full module path (e.g., "os.path", "pandas").
        security_config: Security configuration with allowlists.

    Returns:
        Tuple of (is_allowed, error_message):
        - (True, None) if import is allowed
        - (False, "error description") if import is blocked
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.import_validation import validate_module_import
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| module_path || str || Yes || Full module path to validate (e.g., "os.path", "numpy")
|-
| security_config || SecurityConfig || Yes || Security configuration with allowlists
|}

=== Outputs ===
{| class="wikitable"
|-
! Return Value !! Description
|-
| (True, None) || Module import is allowed
|-
| (False, error_msg) || Module blocked, error_msg describes why
|}

=== Error Message Format ===
{| class="wikitable"
|-
! Module Type !! Error Format
|-
| stdlib || `"Import '{module}' is not allowed. Allowed stdlib modules: {list}"`
|-
| external || `"Import '{module}' is not allowed. Allowed external modules: {list}"`
|}

== Usage Examples ==

=== Basic Validation ===
<syntaxhighlight lang="python">
from src.import_validation import validate_module_import
from src.config.security_config import SecurityConfig

config = SecurityConfig(
    stdlib_allow={"json", "datetime"},
    external_allow={"pandas"},
    builtins_deny=set(),
    runner_env_deny=True,
)

# Allowed stdlib import
is_allowed, error = validate_module_import("json", config)
# (True, None)

# Blocked stdlib import
is_allowed, error = validate_module_import("os", config)
# (False, "Import 'os' is not allowed. Allowed stdlib modules: datetime, json")

# Allowed external import
is_allowed, error = validate_module_import("pandas", config)
# (True, None)

# Blocked external import
is_allowed, error = validate_module_import("numpy", config)
# (False, "Import 'numpy' is not allowed. Allowed external modules: pandas")
</syntaxhighlight>

=== Submodule Handling ===
<syntaxhighlight lang="python">
# Full path extracts root module
is_allowed, error = validate_module_import("os.path", config)
# Validates "os" against stdlib_allow

is_allowed, error = validate_module_import("pandas.DataFrame", config)
# Validates "pandas" against external_allow
</syntaxhighlight>

=== Wildcard Support ===
<syntaxhighlight lang="python">
permissive_config = SecurityConfig(
    stdlib_allow={"*"},      # All stdlib allowed
    external_allow={"*"},    # All external allowed
    builtins_deny=set(),
    runner_env_deny=False,
)

# Everything allowed
is_allowed, _ = validate_module_import("os", permissive_config)  # (True, None)
is_allowed, _ = validate_module_import("subprocess", permissive_config)  # (True, None)
is_allowed, _ = validate_module_import("numpy", permissive_config)  # (True, None)
</syntaxhighlight>

=== Used in Safe Import Wrapper ===
<syntaxhighlight lang="python">
# In TaskExecutor._create_safe_import()
def safe_import(name, *args, **kwargs):
    is_allowed, error_msg = validate_module_import(name, security_config)

    if not is_allowed:
        raise SecurityViolationError(
            message="Security violation detected",
            description=error_msg,
        )

    return original_import(name, *args, **kwargs)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Runtime_Import_Validation]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Sandbox_Environment]]
