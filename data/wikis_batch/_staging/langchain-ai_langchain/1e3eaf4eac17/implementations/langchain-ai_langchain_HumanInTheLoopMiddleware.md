{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai/langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Agent Middleware]], [[domain::Human Oversight]], [[domain::Tool Approval]], [[domain::LangGraph]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==
Agent middleware that intercepts tool calls from LLMs and requires human approval, editing, or rejection before execution.

=== Description ===
HumanInTheLoopMiddleware (HITL) provides granular control over which agent tool calls require human oversight. It hooks into the agent execution flow after the model generates tool calls but before they execute, creating interrupts via LangGraph's interrupt() function. When configured tools are invoked, the middleware packages the tool calls into ActionRequest objects with descriptions and sends them for human review.

The middleware supports three decision types: "approve" (execute as-is), "edit" (modify tool name/args), and "reject" (skip execution with error message). Each tool in the interrupt_on configuration can specify which decisions are allowed. The description field supports both static strings and callable functions that dynamically generate descriptions based on the tool call, agent state, and runtime context.

When decisions are returned, the middleware processes them in order, updating the AIMessage's tool_calls list and optionally injecting artificial ToolMessage objects for rejections. This allows the agent to continue execution with modified or rejected tool calls reflected in the message history.

=== Usage ===
Use HumanInTheLoopMiddleware when building agents that perform high-stakes actions (file deletion, API calls, financial transactions) or when operating in environments where human oversight is required for compliance or safety. Configure specific tools for approval rather than all tools to balance automation with control. Use dynamic description callables to provide context-specific information to human reviewers. Combine with other middleware for layered safety measures.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai/langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain_v1/langchain/agents/middleware/human_in_the_loop.py libs/langchain_v1/langchain/agents/middleware/human_in_the_loop.py]

=== Signature ===
<syntaxhighlight lang="python">
DecisionType = Literal["approve", "edit", "reject"]


class InterruptOnConfig(TypedDict):
    allowed_decisions: list[DecisionType]
    description: NotRequired[str | _DescriptionFactory]
    args_schema: NotRequired[dict[str, Any]]


class HumanInTheLoopMiddleware(AgentMiddleware[StateT, ContextT]):
    def __init__(
        self,
        interrupt_on: dict[str, bool | InterruptOnConfig],
        *,
        description_prefix: str = "Tool execution requires approval",
    ) -> None: ...

    def after_model(
        self, state: AgentState, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None: ...

    async def aafter_model(
        self, state: AgentState, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None: ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain.agents.middleware.human_in_the_loop import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    HITLRequest,
    HITLResponse,
    Decision,
)
</syntaxhighlight>

== I/O Contract ==

=== Initialization Parameters ===
{| class="wikitable"
|-
! Parameter
! Type
! Default
! Description
|-
| interrupt_on
| dict[str, bool or InterruptOnConfig]
| Required
| Mapping of tool names to approval configuration
|-
| description_prefix
| str
| "Tool execution requires approval"
| Default prefix for auto-generated descriptions
|}

=== InterruptOnConfig Fields ===
{| class="wikitable"
|-
! Field
! Type
! Required
! Description
|-
| allowed_decisions
| list[DecisionType]
| Yes
| Which decisions are allowed: "approve", "edit", "reject"
|-
| description
| str or Callable
| No
| Static string or function to generate approval description
|-
| args_schema
| dict[str, Any]
| No
| JSON schema for tool args (if editing is allowed)
|}

=== HITLRequest Structure ===
{| class="wikitable"
|-
! Field
! Type
! Description
|-
| action_requests
| list[ActionRequest]
| List of tool calls awaiting human review
|-
| review_configs
| list[ReviewConfig]
| Configuration for each tool's allowed decisions
|}

=== HITLResponse Structure ===
{| class="wikitable"
|-
! Field
! Type
! Description
|-
| decisions
| list[Decision]
| Human decisions for each action request (in order)
|}

=== Decision Types ===
{| class="wikitable"
|-
! Type
! Fields
! Description
|-
| ApproveDecision
| type: "approve"
| Execute tool call as-is
|-
| EditDecision
| type: "edit", edited_action: Action
| Execute with modified name/args
|-
| RejectDecision
| type: "reject", message: str (optional)
| Skip execution, inject error ToolMessage
|}

=== Methods ===
{| class="wikitable"
|-
! Method
! Returns
! Description
|-
| after_model(state, runtime)
| dict[str, Any] or None
| Process tool calls after model generation (sync)
|-
| aafter_model(state, runtime)
| dict[str, Any] or None
| Process tool calls after model generation (async)
|}

== Usage Examples ==

=== Basic Approval Required ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware
from langchain_openai import ChatOpenAI

# Require approval for delete operations
middleware = HumanInTheLoopMiddleware(
    interrupt_on={
        "delete_file": True,  # All decisions allowed (approve, edit, reject)
        "delete_database": True,
    }
)

agent = create_agent(
    model=ChatOpenAI(),
    tools=[delete_file_tool, delete_database_tool, read_file_tool],
    middleware=[middleware]
)

# When agent tries to call delete_file, execution pauses for human input
</syntaxhighlight>

=== Selective Approval Configuration ===
<syntaxhighlight lang="python">
from langchain.agents.middleware.human_in_the_loop import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
)

middleware = HumanInTheLoopMiddleware(
    interrupt_on={
        # Can only approve or reject (no editing)
        "send_email": InterruptOnConfig(
            allowed_decisions=["approve", "reject"]
        ),
        # Can approve, edit, or reject
        "create_invoice": InterruptOnConfig(
            allowed_decisions=["approve", "edit", "reject"]
        ),
        # Can only approve (must explicitly approve, no auto-pass)
        "deploy_code": InterruptOnConfig(
            allowed_decisions=["approve"]
        ),
        # Auto-approved (no interrupt)
        "read_data": False,
    }
)
</syntaxhighlight>

=== Static Description ===
<syntaxhighlight lang="python">
from langchain.agents.middleware.human_in_the_loop import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
)

middleware = HumanInTheLoopMiddleware(
    interrupt_on={
        "charge_credit_card": InterruptOnConfig(
            allowed_decisions=["approve", "reject"],
            description="This tool will charge the customer's credit card. Please verify the amount before approving."
        ),
    }
)
</syntaxhighlight>

=== Dynamic Description Function ===
<syntaxhighlight lang="python">
import json
from langchain_core.messages import ToolCall
from langgraph.runtime import Runtime
from langchain.agents.middleware.types import AgentState
from langchain.agents.middleware.human_in_the_loop import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
)


def format_file_deletion(
    tool_call: ToolCall,
    state: AgentState,
    runtime: Runtime
) -> str:
    file_path = tool_call["args"].get("path", "unknown")
    # Access state for additional context
    user = state.get("user_id", "unknown")

    return f"""
FILE DELETION REQUEST

User: {user}
File Path: {file_path}
Tool Args: {json.dumps(tool_call["args"], indent=2)}

WARNING: This action cannot be undone. Please verify the path is correct.
"""


middleware = HumanInTheLoopMiddleware(
    interrupt_on={
        "delete_file": InterruptOnConfig(
            allowed_decisions=["approve", "reject"],
            description=format_file_deletion
        ),
    }
)
</syntaxhighlight>

=== Processing Human Decisions ===
<syntaxhighlight lang="python">
from langchain.agents.middleware.human_in_the_loop import (
    ApproveDecision,
    EditDecision,
    RejectDecision,
    Action,
)

# In your application's interrupt handler:

def handle_hitl_request(request):
    """Process the HITL request and return decisions."""
    decisions = []

    for action_request in request["action_requests"]:
        tool_name = action_request["name"]
        tool_args = action_request["args"]
        description = action_request.get("description", "")

        # Present to human reviewer (CLI, UI, etc.)
        print(f"\n{description}")
        print(f"Tool: {tool_name}")
        print(f"Args: {tool_args}")

        choice = input("Decision (approve/edit/reject): ").lower()

        if choice == "approve":
            decisions.append(ApproveDecision(type="approve"))

        elif choice == "edit":
            # Get edited values from user
            new_args = {}  # Collect from user input
            decisions.append(EditDecision(
                type="edit",
                edited_action=Action(
                    name=tool_name,
                    args=new_args
                )
            ))

        elif choice == "reject":
            reason = input("Rejection reason: ")
            decisions.append(RejectDecision(
                type="reject",
                message=reason
            ))

    return {"decisions": decisions}
</syntaxhighlight>

=== Editing Tool Arguments ===
<syntaxhighlight lang="python">
from langchain.agents.middleware.human_in_the_loop import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    EditDecision,
    Action,
)

middleware = HumanInTheLoopMiddleware(
    interrupt_on={
        "send_email": InterruptOnConfig(
            allowed_decisions=["approve", "edit", "reject"],
            # Optional: Provide schema for validation
            args_schema={
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"}
                }
            }
        ),
    }
)

# Human reviewer can modify arguments:
# EditDecision(
#     type="edit",
#     edited_action=Action(
#         name="send_email",
#         args={
#             "to": "corrected@example.com",
#             "subject": "Updated subject",
#             "body": "Corrected body"
#         }
#     )
# )
</syntaxhighlight>

=== Rejection with Feedback ===
<syntaxhighlight lang="python">
from langchain.agents.middleware.human_in_the_loop import RejectDecision

# When rejecting, provide clear feedback to the agent:
decision = RejectDecision(
    type="reject",
    message="Cannot delete system files. Please specify a user data directory instead."
)

# The middleware injects a ToolMessage with this content, allowing the agent
# to see the rejection reason and potentially try a different approach
</syntaxhighlight>

=== Multiple Tool Approvals ===
<syntaxhighlight lang="python">
from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware

# Agent may call multiple tools in one turn
middleware = HumanInTheLoopMiddleware(
    interrupt_on={
        "tool_a": True,
        "tool_b": True,
        "tool_c": False,  # Auto-approved
    }
)

# If agent generates: [tool_a, tool_b, tool_c]
# Human reviews: tool_a (approve), tool_b (reject)
# Execution continues with: tool_a (executed), tool_c (executed)
# tool_b gets ToolMessage with rejection reason
</syntaxhighlight>

=== Integration with LangGraph ===
<syntaxhighlight lang="python">
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware

# HITL middleware integrates with LangGraph's interrupt system
middleware = HumanInTheLoopMiddleware(
    interrupt_on={"dangerous_tool": True}
)

agent = create_react_agent(
    model=llm,
    tools=tools,
    middleware=[middleware]
)

# When tool requires approval, graph execution pauses
# Resume with: agent.invoke(state, config={"resume_value": hitl_response})
</syntaxhighlight>

=== Context-Aware Descriptions ===
<syntaxhighlight lang="python">
from langchain.agents.middleware.human_in_the_loop import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
)


def describe_api_call(tool_call, state, runtime):
    """Generate description with context from state."""
    endpoint = tool_call["args"].get("endpoint")
    method = tool_call["args"].get("method", "GET")

    # Check state for sensitive operations
    if state.get("environment") == "production":
        warning = "⚠️ PRODUCTION ENVIRONMENT"
    else:
        warning = ""

    return f"""
{warning}
API Call: {method} {endpoint}
Data: {tool_call["args"].get("data", "None")}

Request Count: {state.get("request_count", 0)}
"""


middleware = HumanInTheLoopMiddleware(
    interrupt_on={
        "api_call": InterruptOnConfig(
            allowed_decisions=["approve", "edit", "reject"],
            description=describe_api_call
        ),
    }
)
</syntaxhighlight>

=== Combining with Other Middleware ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware
from langchain.agents.middleware.model_retry import ModelRetryMiddleware

# Layer multiple middleware for comprehensive control
retry_middleware = ModelRetryMiddleware(max_retries=2)
hitl_middleware = HumanInTheLoopMiddleware(
    interrupt_on={"delete_file": True}
)

agent = create_agent(
    model=llm,
    tools=tools,
    middleware=[retry_middleware, hitl_middleware]
)

# Execution flow:
# 1. Model retry handles failures
# 2. HITL blocks sensitive operations
# Both work together for robust agent behavior
</syntaxhighlight>

== Related Pages ==
* [[langchain-ai_langchain_AgentMiddleware]] - Base class for all middleware
* [[langchain-ai_langchain_ModelRetryMiddleware]] - Automatic retry middleware
* [[principle::Human Oversight in AI Systems]]
* [[principle::Graceful Degradation]]
* [[environment::Production AI Agents]]
* [[environment::Regulated Industries]]
