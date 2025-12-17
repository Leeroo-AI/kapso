{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai/langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Agent Middleware]], [[domain::Rate Limiting]], [[domain::Resource Control]], [[domain::Safety]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==
ToolCallLimitMiddleware tracks and enforces limits on tool call counts during agent execution, supporting both thread-level (persistent across runs) and run-level (per invocation) limits with configurable exit behaviors.

=== Description ===
This middleware provides fine-grained control over tool usage by counting and limiting tool calls at two scopes:
* '''Thread-level limits''': Persistent across multiple agent runs in the same thread
* '''Run-level limits''': Per-agent invocation (resets each run)

When limits are exceeded, the middleware supports three exit behaviors:
* '''Continue''': Block exceeded tools with error messages, allow execution to continue
* '''Error''': Raise `ToolCallLimitExceededError` exception
* '''End''': Immediately stop execution with injected error messages

The middleware can apply globally to all tools or target specific tools by name, enabling sophisticated resource management strategies like:
* Preventing runaway tool loops
* Enforcing rate limits on expensive operations
* Budget control for paid API tools
* Safety limits for dangerous operations

=== Usage ===
Use this middleware when you need to:
* Prevent infinite loops or excessive tool usage
* Control costs for agents using paid APIs
* Enforce safety limits on risky operations (file deletion, system commands)
* Implement rate limiting for specific tools
* Track and limit tool usage across conversation history
* Gracefully degrade when resource budgets are exhausted

== Code Reference ==
'''Source location:''' `/tmp/praxium_repo_wjjl6pl8/libs/langchain_v1/langchain/agents/middleware/tool_call_limit.py`

'''Signature:'''
<syntaxhighlight lang="python">
class ToolCallLimitMiddleware(
    AgentMiddleware[ToolCallLimitState[ResponseT], ContextT],
    Generic[ResponseT, ContextT],
):
    def __init__(
        self,
        *,
        tool_name: str | None = None,
        thread_limit: int | None = None,
        run_limit: int | None = None,
        exit_behavior: ExitBehavior = "continue",
    ) -> None
</syntaxhighlight>

'''Import statement:'''
<syntaxhighlight lang="python">
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.agents.middleware.tool_call_limit import (
    ToolCallLimitExceededError,
    ExitBehavior
)
</syntaxhighlight>

== I/O Contract ==

=== Initialization Parameters ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| tool_name || str/None || None || Name of specific tool to limit. If None, limits apply to all tools
|-
| thread_limit || int/None || None || Maximum calls allowed per thread (persistent). None = no limit
|-
| run_limit || int/None || None || Maximum calls allowed per run (resets each invocation). None = no limit
|-
| exit_behavior || ExitBehavior || "continue" || How to handle exceeded limits: "continue", "error", or "end"
|}

=== Exit Behaviors ===
{| class="wikitable"
! Behavior !! Description !! Use Case
|-
| "continue" || Block exceeded tools with error ToolMessages, let other tools continue || Graceful degradation, let model decide when to end
|-
| "error" || Raise ToolCallLimitExceededError exception || Strict enforcement, immediate failure
|-
| "end" || Stop immediately with ToolMessage + final AI message || Clean termination with user-facing explanation
|}

=== State Schema Extension ===
{| class="wikitable"
! Field !! Type !! Persistence !! Description
|-
| thread_tool_call_count || dict[str, int] || Thread-level || Counts per tool across runs (key: tool_name or "__all__")
|-
| run_tool_call_count || dict[str, int] || Run-level (UntrackedValue) || Counts per tool in current run (resets each invocation)
|}

=== Hook Methods ===
{| class="wikitable"
! Method !! Execution Point !! Purpose
|-
| after_model() || After model invocation || Check tool calls in last AIMessage, increment counts, enforce limits
|}

=== Exception ===
{| class="wikitable"
! Exception !! Attributes !! When Raised
|-
| ToolCallLimitExceededError || thread_count, run_count, thread_limit, run_limit, tool_name || When exit_behavior="error" and limits exceeded
|}

== Usage Examples ==

=== Example 1: Global Tool Call Limit (Continue Behavior) ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware

# Limit total tool calls across all tools
agent = create_agent(
    "openai:gpt-4o",
    tools=[search_tool, calculator_tool, database_tool],
    middleware=[
        ToolCallLimitMiddleware(
            thread_limit=20,        # Max 20 calls per thread
            run_limit=10,           # Max 10 calls per run
            exit_behavior="continue"  # Default: block but continue
        )
    ],
)

# After 10 calls in a run or 20 calls total:
# - Tool calls blocked with error ToolMessages
# - Model continues processing (can finish without tools)
</syntaxhighlight>

=== Example 2: Specific Tool Limit (Error Behavior) ===
<syntaxhighlight lang="python">
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError

# Limit expensive search tool to 5 calls per thread
agent = create_agent(
    "openai:gpt-4o",
    tools=[expensive_search_tool, calculator_tool],
    middleware=[
        ToolCallLimitMiddleware(
            tool_name="expensive_search",
            thread_limit=5,
            exit_behavior="error"
        )
    ],
)

try:
    result = agent.invoke({"messages": [{"role": "user", "content": "Research topic X"}]})
except ToolCallLimitExceededError as e:
    print(f"Search limit exceeded: {e.thread_count}/{e.thread_limit} calls")
    # Handle appropriately (notify user, log, etc.)
</syntaxhighlight>

=== Example 3: Graceful Termination (End Behavior) ===
<syntaxhighlight lang="python">
# Stop immediately when limit exceeded, with clean exit message
agent = create_agent(
    "openai:gpt-4o",
    tools=[calculator_tool],
    middleware=[
        ToolCallLimitMiddleware(
            run_limit=5,
            exit_behavior="end"
        )
    ],
)

result = agent.invoke({"messages": [{"role": "user", "content": "Complex calculation"}]})

# When 6th call attempted:
# - ToolMessage with error injected
# - AIMessage with explanation injected
# - Execution jumps to "end" (bypasses remaining graph nodes)
# - Result returned to user with clear explanation
</syntaxhighlight>

=== Example 4: Multiple Tool Limits ===
<syntaxhighlight lang="python">
# Apply different limits to different tools
agent = create_agent(
    "openai:gpt-4o",
    tools=[search_tool, database_tool, file_reader_tool],
    middleware=[
        ToolCallLimitMiddleware(
            tool_name="search",
            thread_limit=10,
            run_limit=5,
        ),
        ToolCallLimitMiddleware(
            tool_name="database",
            thread_limit=20,
            run_limit=10,
        ),
        ToolCallLimitMiddleware(
            tool_name="file_reader",
            thread_limit=50,
            run_limit=25,
        ),
    ],
)

# Each tool tracked independently with different limits
</syntaxhighlight>

=== Example 5: Run-Only Limit (No Thread Persistence) ===
<syntaxhighlight lang="python">
# Limit per run only (resets each invocation)
agent = create_agent(
    "openai:gpt-4o",
    tools=[calculator_tool],
    middleware=[
        ToolCallLimitMiddleware(
            run_limit=15,
            # No thread_limit specified
        )
    ],
)

# Each run gets fresh budget of 15 calls
# No accumulation across runs
</syntaxhighlight>

=== Example 6: Thread-Only Limit (Persistent Budget) ===
<syntaxhighlight lang="python">
# Total budget across all runs in thread
agent = create_agent(
    "openai:gpt-4o",
    tools=[paid_api_tool],
    middleware=[
        ToolCallLimitMiddleware(
            tool_name="paid_api",
            thread_limit=100,
            # No run_limit specified
            exit_behavior="error"
        )
    ],
)

# Total of 100 calls allowed across ALL runs
# Useful for budget control on paid APIs
</syntaxhighlight>

=== Example 7: Safety Limit on Dangerous Tool ===
<syntaxhighlight lang="python">
# Prevent excessive use of potentially dangerous tool
agent = create_agent(
    "openai:gpt-4o",
    tools=[file_deletion_tool, file_reader_tool],
    middleware=[
        ToolCallLimitMiddleware(
            tool_name="delete_file",
            thread_limit=3,     # Max 3 deletions per thread
            run_limit=1,        # Max 1 deletion per run
            exit_behavior="error"
        )
    ],
)

# Prevents runaway deletion loops
</syntaxhighlight>

=== Example 8: Parallel Tool Call Handling ===
<syntaxhighlight lang="python">
# Middleware handles parallel tool calls intelligently
agent = create_agent(
    "openai:gpt-4o",
    tools=[search_tool, calculator_tool, translator_tool],
    middleware=[
        ToolCallLimitMiddleware(
            run_limit=8,
            exit_behavior="continue"
        )
    ],
)

# If model makes 5 parallel tool calls and limit is 8:
# - Calls 1-3 allowed (total: 3)
# - Calls 4-6 allowed (total: 6)
# - Calls 7-8 allowed (total: 8)
# - Call 9+ blocked with error ToolMessages
# Order: tool calls processed in list order
</syntaxhighlight>

=== Example 9: Combining with Other Middleware ===
<syntaxhighlight lang="python">
from langchain.agents.middleware import ToolRetryMiddleware, SummarizationMiddleware

# Compose with retry and summarization
agent = create_agent(
    "openai:gpt-4o",
    tools=[unreliable_api_tool, calculator_tool],
    middleware=[
        ToolRetryMiddleware(max_retries=3),           # Retry failed calls
        ToolCallLimitMiddleware(                      # But still enforce limits
            tool_name="unreliable_api",
            thread_limit=15,
            exit_behavior="continue"
        ),
        SummarizationMiddleware(                      # Manage long conversations
            model="openai:gpt-4o-mini",
            trigger=("tokens", 4000)
        ),
    ],
)

# Middleware execution order matters:
# 1. Retry attempts tool up to 3 times
# 2. Each attempt counts toward limit (retries count as separate calls)
# 3. Limit enforced after retries exhausted
</syntaxhighlight>

== Implementation Details ==

=== Count Tracking ===
Two separate dictionaries maintain counts:
* `thread_tool_call_count`: Persisted in checkpoints (thread-level)
* `run_tool_call_count`: Marked as `UntrackedValue` (run-level only)

Dictionary keys:
* Specific tool: `tool_name` (e.g., "search")
* All tools: `"__all__"`

Multiple middleware instances can coexist with different keys.

=== Tool Call Separation Logic ===
The `_separate_tool_calls` method processes tool calls sequentially:
```python
for tool_call in tool_calls:
    if not matches_tool_filter(tool_call):
        continue  # Skip if not tracking this tool

    if would_exceed_limit(temp_thread_count, temp_run_count):
        blocked_calls.append(tool_call)
    else:
        allowed_calls.append(tool_call)
        temp_thread_count += 1
        temp_run_count += 1
```

This ensures fair ordering: first N calls allowed, remainder blocked.

=== Count Update Strategy ===
'''Allowed calls:'''
* Increment both thread_count and run_count
* Added to permanent state

'''Blocked calls:'''
* Increment only run_count (attempted in this run)
* Do NOT increment thread_count (not actually executed)

Rationale: Blocked calls didn't consume thread budget, but were attempted in the run.

=== Exit Behavior Implementation ===

'''Continue:'''
* Inject error ToolMessages for blocked calls
* Model receives errors and continues processing
* Can complete without blocked tools or request stop

'''Error:'''
* Raise `ToolCallLimitExceededError` immediately
* Exception includes counts and limits for debugging
* Agent execution stops

'''End:'''
* Check for other pending tool calls
* If none (or only blocked tool), inject ToolMessage + AIMessage
* Set `jump_to: "end"` in state update
* Execution bypasses remaining graph nodes
* Returns cleanly to user

=== Parallel Tool Call Constraints ===
With `exit_behavior="end"` and `tool_name` specified:
* If model calls limited tool AND other tools in parallel
* Middleware raises `NotImplementedError`
* Reason: Can't cleanly end while other tools are executing
* Solution: Use "continue" or "error" behavior instead

=== Message Content Generation ===

'''ToolMessage (sent to model):'''
```
Tool call limit exceeded. Do not call '{tool_name}' again.
```
No mention of thread/run concepts (model doesn't understand these).

'''AIMessage (displayed to user):'''
```
'{tool_name}' tool call limit reached: thread limit exceeded (21/20 calls) and run limit exceeded (11/10 calls).
```
Detailed information for user/developer debugging.

=== Hook Configuration ===
The `after_model` hook can jump to end:
```python
@hook_config(can_jump_to=["end"])
def after_model(self, state, runtime):
    # ...
    return {"jump_to": "end", ...}
```

This enables immediate termination when `exit_behavior="end"`.

=== State Update Return ===
Returns dictionary with:
* `thread_tool_call_count`: Updated thread-level counts
* `run_tool_call_count`: Updated run-level counts
* `messages`: (if limits exceeded) Injected error messages
* `jump_to`: (if exit_behavior="end") "end" destination

=== Middleware Naming ===
Instance name includes tool name if specified:
```python
@property
def name(self) -> str:
    if self.tool_name:
        return f"ToolCallLimitMiddleware[{self.tool_name}]"
    return "ToolCallLimitMiddleware"
```

Enables multiple instances with clear identification in logs.

=== Validation ===
Constructor validates:
* At least one limit specified (thread or run)
* Exit behavior is valid ("continue", "error", "end")
* Run limit doesn't exceed thread limit (prevents impossible scenarios)

=== Async Support ===
Async hook delegates to sync implementation:
```python
async def aafter_model(self, state, runtime):
    return self.after_model(state, runtime)
```

No actual async work (just count checking and message building).

=== Performance Characteristics ===
* O(n) where n = number of tool calls in last AIMessage
* Typically small n (1-10 parallel calls)
* Dictionary lookups: O(1)
* No network calls or I/O
* Negligible overhead

== Related Pages ==
* [[langchain-ai_langchain_AgentMiddleware|AgentMiddleware]] - Base middleware class
* [[langchain-ai_langchain_ToolRetryMiddleware|ToolRetryMiddleware]] - Retry failed tool calls
* [[Resource Management Patterns]] - Controlling agent resource usage
* [[Agent Safety Mechanisms]] - Preventing runaway execution
* [[LangGraph State Management]] - Understanding thread vs run state
