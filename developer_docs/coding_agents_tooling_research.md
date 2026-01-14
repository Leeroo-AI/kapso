# Coding agent tooling research (MCP / tool calling)
#
# This document summarizes what our current adapters actually do today, and what
# we would need to change to enable tool calling (MCP, CLI tools, etc.) across
# different coding agents.

## Current reality in Praxium (as implemented)

### Claude Code (`claude_code`)

- **Tool runtime exists** because Claude Code is a CLI agent that supports tools.
- Our adapter already supports a tool allowlist via `--allowedTools`.
- Today we allow only built-ins like `Edit`, `Read`, `Write`, `Bash` (see `agents.yaml`).
- **MCP is NOT wired in the Praxium adapter today**:
  - We do not pass an MCP config flag.
  - Even if the user configured MCP globally, our `--allowedTools` would block
    `mcp__...` tools unless we explicitly include them.

Pragmatic short-term approach (what we implemented):
- Provide a **RepoMemory CLI** that Claude Code can call using **Bash**:
  - `python3 tools/repo_memory_cli.py get-section core.architecture`

This gives Claude Code “tool-like” access to RepoMemory without adding MCP coupling.

### Gemini (`gemini`)

- Our adapter is **text-in/text-out** via Google’s SDK, then we parse code blocks.
- Despite comments mentioning tool loops/MCP, the current adapter does **not**
  implement:
  - function/tool calling
  - sandboxed execution
  - MCP client integration

If we want tools for Gemini, we need to:
- implement function calling in the adapter and route calls to engine-defined tools
  (e.g., “get RepoMemory section”, “grep”, “read file”, “run tests”)
- or run Gemini as an external “agent runner” that supports tool loops.

### OpenHands (`openhands`)

- Our adapter is also **text-in/text-out**, plus code-block parsing.
- It does not run the full OpenHands agent runtime (no sandbox/tools loop in this adapter).

If we want OpenHands tool use, we need to:
- integrate the actual OpenHands agent runtime (and its tool API),
- or define an engine-mediated tool protocol similar to ideation ReAct.

### Aider (`aider`)

- Our adapter expects the `aider` python package (`aider.coders`) and calls `coder.run(prompt)`.
- In this environment, Aider is often not installed, so it is unavailable unless installed.
- Tool calling / MCP is not part of our wrapper. Aider’s “tools” are largely internal
  to its CLI UX and would need explicit integration if we want audited tool calls.

## What “real tools” would look like (future work)

### Option A: Engine-mediated tools (portable, deterministic)

Model outputs structured JSON actions like:
- `{"action":"get_repo_memory_section","section_id":"core.architecture"}`
- `{"action":"read_file","path":"src/foo.py"}`
- `{"action":"run_tests","cmd":"pytest -q ..."}`

The engine executes these actions and feeds results back into the conversation.

Pros:
- works with any LLM backend (no native tool calling required)
- easy to log/audit centrally

Cons:
- you need to implement the protocol and loop in each adapter/flow

### Option B: MCP integration (standardized, multi-tool ecosystem)

Expose tools as MCP servers (like our Knowledge Search MCP server under `src/knowledge/wiki_mcps/`),
then configure agents to connect.

Pros:
- standard protocol, works with multiple MCP clients
- tool definitions are external and reusable

Cons:
- each agent must support MCP *and* allow tool names (e.g. `mcp__server__tool`)
- Praxium must pass MCP config through adapters, and manage allowlists

## Recommended next engineering steps

1. **Claude Code MCP pass-through**
   - Extend `ClaudeCodeCodingAgent` config to optionally pass an MCP config file path.
   - Expand `allowed_tools` to include selected `mcp__...` tools when enabled.
   - Keep default “safe mode” (no MCP) to avoid unexpected tool access.

2. **Gemini function calling**
   - Implement function calling in `GeminiCodingAgent` with a small set of tools:
     - get RepoMemory section
     - grep/read files
     - (optionally) run tests

3. **OpenHands full runtime integration**
   - Replace the simplified litellm-based adapter with the actual OpenHands agent runtime
     if we want sandbox + tools.

4. **Aider availability + capability clarification**
   - Make Aider adapter optional and improve error messages / install path.
   - Decide whether we want “tool use” semantics for Aider or keep it as pure editing.

