# File: `vllm/version.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 39 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Version information and version checking utilities.

**Mechanism:** Attempts to import `__version__` and `__version_tuple__` from a generated `_version` module (created during build). Falls back to "dev" version if import fails. Provides helper functions: `_prev_minor_version_was()` checks if a version string matches the previous minor version (for backward compatibility features), and `_prev_minor_version()` returns the previous minor version string. Version tuple format is `(major, minor, patch/string)`. The checking logic assumes 0.x versioning (major=0).

**Significance:** Essential for version management and backward compatibility. The version string is exposed via vLLM's public API (`vllm.__version__`) for programmatic version checking. The helper functions support features like `--show-hidden-metrics-for-version` that enable deprecated metrics for one version back. The fallback to "dev" ensures development builds work without a build system. This is a simple but critical piece of package infrastructure.
