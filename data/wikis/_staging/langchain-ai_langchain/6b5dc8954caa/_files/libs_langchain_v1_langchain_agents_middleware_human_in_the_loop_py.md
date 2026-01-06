# File: `libs/langchain_v1/langchain/agents/middleware/human_in_the_loop.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 357 |
| Classes | `Action`, `ActionRequest`, `ReviewConfig`, `HITLRequest`, `ApproveDecision`, `EditDecision`, `RejectDecision`, `HITLResponse`, `_DescriptionFactory`, `InterruptOnConfig`, `HumanInTheLoopMiddleware` |
| Imports | langchain, langchain_core, langgraph, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Intercepts agent tool executions to request human approval, editing, or rejection before execution, enabling human oversight of sensitive or high-risk agent actions.

**Mechanism:** HumanInTheLoopMiddleware's after_model hook examines the last AIMessage's tool_calls, filters for tools in interrupt_on configuration (mapping tool name to InterruptOnConfig with allowed_decisions=['approve','edit','reject'] and optional description/args_schema), creates HITLRequest with ActionRequest list containing tool names/args/descriptions, calls langgraph.types.interrupt() to pause execution and receive HITLResponse with Decision list, then processes decisions to produce revised tool_calls (approve keeps original, edit creates new ToolCall with edited_action, reject injects error ToolMessage). Auto-approves tools not in interrupt_on.

**Significance:** Core safety and control mechanism for autonomous agents - enables human-in-the-loop patterns where sensitive operations (data deletion, external API calls, financial transactions) require explicit human approval, with flexible decision types supporting both simple yes/no approval and more sophisticated parameter editing workflows.
