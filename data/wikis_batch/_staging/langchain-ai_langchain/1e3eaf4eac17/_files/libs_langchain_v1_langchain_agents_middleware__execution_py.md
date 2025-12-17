# File: `libs/langchain_v1/langchain/agents/middleware/_execution.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 389 |
| Classes | `BaseExecutionPolicy`, `HostExecutionPolicy`, `CodexSandboxExecutionPolicy`, `DockerExecutionPolicy` |
| Imports | __future__, abc, collections, dataclasses, json, os, pathlib, shutil, subprocess, sys, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides secure execution policy abstractions for running shell commands in agent middleware, with three distinct isolation strategies: direct host execution, Codex CLI sandboxing, and Docker containerization.

**Mechanism:** The file defines an abstract `BaseExecutionPolicy` contract that requires subclasses to implement the `spawn()` method for launching shell processes. Three concrete policies implement different security models: (1) `HostExecutionPolicy` runs commands directly on the host with optional CPU and memory limits via the `resource` module (using `prlimit` on Linux or `preexec_fn` on macOS); (2) `CodexSandboxExecutionPolicy` wraps commands with the Codex CLI sandbox providing syscall and filesystem restrictions through Anthropic's Seatbelt/Landlock profiles; (3) `DockerExecutionPolicy` executes commands in isolated Docker containers with configurable network isolation, read-only rootfs, user mapping, and resource constraints. All policies configure timeouts, output limits, and environment variables.

**Significance:** This module is critical for the ShellToolMiddleware security model, enabling safe execution of agent-generated shell commands in production environments. The pluggable policy design lets operators choose appropriate isolation levels based on their threat model: trusted environments can use host execution for performance, while multi-tenant systems can leverage Docker for strong isolation. The policies prevent common security issues like resource exhaustion, path traversal, and unrestricted network access.
