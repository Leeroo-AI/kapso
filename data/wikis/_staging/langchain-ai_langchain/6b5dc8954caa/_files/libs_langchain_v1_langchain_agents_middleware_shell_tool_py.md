# File: `libs/langchain_v1/langchain/agents/middleware/shell_tool.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 760 |
| Classes | `_SessionResources`, `ShellToolState`, `CommandExecutionResult`, `ShellSession`, `_ShellToolInput`, `ShellToolMiddleware` |
| Imports | __future__, contextlib, dataclasses, langchain, langchain_core, langgraph, logging, os, pathlib, pydantic, ... +10 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides a persistent shell session tool for agents with configurable execution policies and security controls.

**Mechanism:** Implements ShellToolMiddleware that manages ShellSession (subprocess wrapper with stdin/stdout/stderr queues). Creates shell subprocess at agent start, registers "shell" tool that executes commands sequentially in persistent session. Uses marker-based output collection (injects unique markers after commands) and timeout handling. Supports HostExecutionPolicy, CodexSandboxExecutionPolicy, and DockerExecutionPolicy for different security/isolation levels. Handles startup/shutdown commands, output truncation, redaction rules, and resource cleanup via weakref finalizers.

**Significance:** Core infrastructure middleware enabling agents to execute shell commands safely. Essential for code-execution agents, file system operations, and environment setup. Provides production-grade features: persistent sessions, configurable security policies, PII redaction, timeout handling, and proper resource lifecycle management with automatic cleanup.
