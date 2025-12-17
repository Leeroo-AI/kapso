# File: `vllm/version.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 39 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Version management and compatibility

**Mechanism:** Imports __version__ and __version_tuple__ from auto-generated _version module with fallback to "dev" on failure. Provides helper functions: _prev_minor_version_was() checks if supplied version matches previous minor version (for --show-hidden-metrics-for-version flag), and _prev_minor_version() returns the previous minor version string. Handles development builds (0.0.x) specially by matching any version.

**Significance:** Central version tracking for the package. Used for API compatibility checks, feature gating, and backward compatibility warnings. The previous version checking supports gradual deprecation of features and metrics. Critical for maintaining stable releases and managing version-dependent behavior across the codebase.
