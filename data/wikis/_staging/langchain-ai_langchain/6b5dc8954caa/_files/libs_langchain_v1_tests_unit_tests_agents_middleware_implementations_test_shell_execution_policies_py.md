# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_shell_execution_policies.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 403 |
| Classes | `_BaseResource`, `DummyProcess`, `DummyProcess`, `DummyProcess`, `DummyProcess`, `DummyProcess`, `Custom`, `DummyProcess`, `DummyProcess`, `DummyProcess`, `_Resource`, `_Resource` |
| Functions | `test_host_policy_validations`, `test_host_policy_requires_resource_for_limits`, `test_host_policy_applies_prlimit`, `test_host_policy_uses_preexec_on_macos`, `test_host_policy_respects_process_group_flag`, `test_host_policy_falls_back_to_rlimit_data`, `test_codex_policy_spawns_codex_cli`, `test_codex_policy_auto_platform_linux`, `... +13 more` |
| Imports | __future__, langchain, os, pathlib, pytest, shutil, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests execution policy implementations for safely running shell commands with resource limits and sandboxing.

**Mechanism:** Uses monkeypatch to mock system modules and subprocess launching, creating fake resource modules with tracking of prlimit/setrlimit calls. Tests three policy types: (1) HostExecutionPolicy - validates parameter checking, verifies prlimit usage on Linux with RLIMIT_CPU/RLIMIT_AS, tests macOS fallback to preexec_fn with setrlimit, checks process group creation control, and validates RLIMIT_DATA fallback when RLIMIT_AS unavailable, (2) CodexSandboxExecutionPolicy - verifies codex CLI command construction with config overrides, platform auto-detection (linux/macos), binary resolution, and config sorting, (3) DockerExecutionPolicy - validates docker run command building with volume mounts, working directory, environment variables, memory/CPU limits, read-only rootfs, user specification, and special handling for temporary workspaces.

**Significance:** Critical security validation ensuring shell command execution respects resource limits (preventing DoS), properly sandboxes execution environments (CodexSandbox/Docker), and handles platform differences correctly for safe agent tool execution.
