# Principle: Security Configuration

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Security]], [[domain::Configuration]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Principle for defining security policies as data structures that control module imports, builtin access, and environment isolation in sandboxed code execution.

=== Description ===

Security Configuration encapsulates the security policy as a structured data object:

1. **Stdlib Allowlist**: Set of standard library modules permitted for import
2. **External Allowlist**: Set of third-party packages permitted for import
3. **Builtins Denylist**: Set of builtin functions/names to remove from execution context
4. **Environment Deny Flag**: Boolean controlling whether to clear environment variables

This separation of policy from enforcement enables:
- **Policy Flexibility**: Different security levels for different contexts
- **Environment Configuration**: Policies loaded from environment variables
- **Wildcard Support**: `*` in allowlists permits all modules
- **Testability**: Policies can be constructed directly for testing

=== Usage ===

Apply this principle when:
- Building configurable security systems
- Implementing multi-tenant environments with different security levels
- Creating test fixtures for security testing
- Designing defense-in-depth security architectures

== Theoretical Basis ==

Security configuration follows the **Policy as Data** pattern:

<syntaxhighlight lang="python">
# Pseudo-code for security policy structure

@dataclass
class SecurityConfig:
    # Allowlists (what IS permitted)
    stdlib_allow: set[str]    # e.g., {"json", "datetime", "re"}
    external_allow: set[str]  # e.g., {"pandas", "numpy"}

    # Denylists (what is NOT permitted)
    builtins_deny: set[str]   # e.g., {"eval", "exec", "compile"}

    # Isolation flags
    runner_env_deny: bool     # Clear os.environ if True

# Wildcard semantics
if "*" in stdlib_allow:
    # All stdlib modules permitted
if "*" in external_allow:
    # All external packages permitted
</syntaxhighlight>

Default security policy typically includes:
- Empty `stdlib_allow` (no stdlib by default)
- Empty `external_allow` (no external packages by default)
- Builtins deny: `{"eval", "exec", "compile", "open", ...}`
- Environment deny: `True`

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_SecurityConfig]]
