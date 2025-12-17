# File: `libs/langchain_v1/langchain/agents/middleware/shell_tool.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 760 |
| Classes | `_SessionResources`, `ShellToolState`, `CommandExecutionResult`, `ShellSession`, `_ShellToolInput`, `ShellToolMiddleware` |
| Imports | __future__, contextlib, dataclasses, langchain, langchain_core, langgraph, logging, os, pathlib, pydantic, ... +10 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements a persistent shell tool that agents can use to execute shell commands in a stateful session, supporting multiple execution policies (host, Docker, Codex sandbox) for different security requirements, with features like output limits, timeouts, PII redaction, and session lifecycle management.

**Mechanism:** The ShellToolMiddleware creates a long-lived shell session (typically bash) at agent start via `before_agent`, manages it through the agent's lifecycle, and exposes it as a tool that agents can call. The ShellSession class spawns a subprocess with stdin/stdout/stderr pipes, uses background threads to queue output, and employs a marker-based protocol to detect command completion. Commands are executed with configurable timeouts, output is limited by lines/bytes, and a cleanup finalizer ensures resources are released. The middleware supports startup/shutdown commands, workspace directories, custom environments, and optional PII redaction on command output.

**Significance:** This is a powerful infrastructure component that enables agents to interact with the filesystem, run build tools, execute tests, and perform system operations. The multiple execution policy options (HostExecutionPolicy for trusted environments, DockerExecutionPolicy for isolation, CodexSandboxExecutionPolicy for additional restrictions) make it adaptable to different security postures. The persistent session design preserves state (working directory, environment variables) across multiple commands, which is essential for complex multi-step operations like software development workflows.
