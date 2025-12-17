# File: `libs/langchain_v1/langchain/agents/middleware/human_in_the_loop.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 357 |
| Classes | `Action`, `ActionRequest`, `ReviewConfig`, `HITLRequest`, `ApproveDecision`, `EditDecision`, `RejectDecision`, `HITLResponse`, `_DescriptionFactory`, `InterruptOnConfig`, `HumanInTheLoopMiddleware` |
| Imports | langchain, langchain_core, langgraph, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements human-in-the-loop (HITL) approval flows for agent tool calls, allowing human reviewers to approve, edit, or reject actions before execution.

**Mechanism:** The middleware hooks into the `after_model()` lifecycle point, inspecting the last `AIMessage` for tool calls. For each tool in the `interrupt_on` configuration, it constructs an `ActionRequest` with tool name, arguments, and description (static string or dynamic callable). All pending requests are batched into a single `HITLRequest` and sent via LangGraph's `interrupt()` function, pausing execution until a human responds with a `HITLResponse` containing decisions. The middleware processes decisions based on allowed actions: `approve` passes through unchanged, `edit` creates a new `ToolCall` with modified name/args, `reject` injects an artificial `ToolMessage` with error status and human-provided message. The revised tool calls replace the original `AIMessage.tool_calls`, and any rejection messages are appended to the conversation.

**Significance:** This middleware is essential for production agent deployments where autonomous tool execution carries risk (e.g., financial transactions, code deployment, data deletion). It provides a flexible approval workflow that can be configured per-tool, supporting different safety models: critical tools require approval, semi-trusted tools allow editing for correction, low-risk tools auto-approve. The design integrates with LangGraph Studio's UI for human review, but the interrupt mechanism is extensible to custom approval systems. The batching approach minimizes human interruptions while maintaining safety.
