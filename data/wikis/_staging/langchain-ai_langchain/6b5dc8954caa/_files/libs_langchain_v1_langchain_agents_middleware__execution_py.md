# File: `libs/langchain_v1/langchain/agents/middleware/_execution.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 389 |
| Classes | `BaseExecutionPolicy`, `HostExecutionPolicy`, `CodexSandboxExecutionPolicy`, `DockerExecutionPolicy` |
| Imports | __future__, abc, collections, dataclasses, json, os, pathlib, shutil, subprocess, sys, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines execution policies for running persistent shell processes with varying isolation and security guarantees, supporting host-level, sandboxed, and containerized execution environments.

**Mechanism:** Implements abstract BaseExecutionPolicy with three concrete strategies: (1) HostExecutionPolicy spawns shells directly with optional CPU/memory limits via resource.prlimit (Linux) or preexec_fn (macOS), (2) CodexSandboxExecutionPolicy wraps commands with the Codex CLI sandbox tool applying syscall restrictions via Seatbelt/Landlock/seccomp, (3) DockerExecutionPolicy executes shells in isolated containers with configurable network, memory, CPU limits, read-only rootfs, and workspace mounting based on path prefix. All policies configure subprocess.Popen with timeouts, output limits, and environment/cwd management.

**Significance:** Core security infrastructure for the ShellToolMiddleware - enables operators to select appropriate isolation levels based on trust boundaries (trusted local dev vs untrusted multi-tenant), with each policy documenting its security guarantees and platform requirements.
