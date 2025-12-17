# File: `packages/@n8n/task-runner-python/src/config/security_config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 9 |
| Classes | `SecurityConfig` |
| Imports | dataclasses |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines Python code execution security constraints

**Mechanism:** The `SecurityConfig` dataclass contains four security-related sets/flags: `stdlib_allow` (set of allowed standard library modules), `external_allow` (set of allowed external packages), `builtins_deny` (set of denied built-in functions), and `runner_env_deny` (boolean controlling access to runner environment variables). This is a simple data container with no methods - it serves as a typed configuration object passed to security enforcement components.

**Significance:** This is a critical security component that enforces sandbox boundaries for user-supplied Python code execution. By controlling which standard library modules, external packages, and built-in functions are accessible, it prevents malicious or accidental code from accessing sensitive system resources, performing unauthorized operations, or escaping the execution sandbox. The runner environment denial prevents code from reading task runner configuration and credentials.
