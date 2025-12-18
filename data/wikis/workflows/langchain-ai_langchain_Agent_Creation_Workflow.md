{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Agents|https://docs.langchain.com/oss/python/langchain/agents]]
|-
! Domains
| [[domain::LLMs]], [[domain::Agents]], [[domain::LangGraph]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
End-to-end process for creating and executing AI agents with tool calling, middleware composition, and structured output capabilities using LangChain's `create_agent` factory.

=== Description ===
This workflow outlines the complete process for building autonomous AI agents that can reason and act through tool calling loops. The `create_agent` factory function constructs a LangGraph StateGraph that orchestrates model invocations, tool executions, and middleware hooks. The agent iterates in a loop: calling the model, executing any tool calls, and repeating until no more tools are requested or a structured response is generated.

Key capabilities:
* Multi-provider model support via unified initialization
* Middleware composition for cross-cutting concerns (retry, logging, rate limiting)
* Structured output via tool-based or provider-native strategies
* Checkpointing for conversation persistence
* Human-in-the-loop approval workflows

=== Usage ===
Execute this workflow when you need to build an autonomous agent that can:
* Call external tools/APIs based on user requests
* Maintain conversation history across turns
* Return structured, validated responses
* Apply cross-cutting middleware (retry, logging, security)

Typical use cases: chatbots with tool access, automated research agents, coding assistants, customer support bots.

== Execution Steps ==

=== Step 1: Model Initialization ===
[[step::Principle:langchain-ai_langchain_Chat_Model_Initialization]]

Initialize the language model using either a string identifier or direct model instance. The `init_chat_model` factory provides a unified interface across 20+ providers, automatically inferring the provider from model name prefixes (e.g., "gpt-" → OpenAI, "claude" → Anthropic).

'''Key considerations:'''
* Provider packages must be installed (e.g., `langchain-openai`, `langchain-anthropic`)
* Model names can use `provider:model` syntax (e.g., `"anthropic:claude-sonnet-4-5-20250929"`)
* Configurable models support runtime model switching

=== Step 2: Tool Definition ===
[[step::Principle:langchain-ai_langchain_Tool_Definition]]

Define the tools available to the agent. Tools can be Python functions, LangChain `BaseTool` instances, or provider-native tools (dict format). Each tool must have a clear docstring describing its purpose - this becomes the tool's description for the model.

'''What happens:'''
* Functions are converted to `StructuredTool` instances
* Tool schemas are extracted for model binding
* Middleware tools are merged with user-provided tools

=== Step 3: Middleware Composition ===
[[step::Principle:langchain-ai_langchain_Middleware_Composition]]

Configure middleware instances to intercept and modify agent behavior at various lifecycle stages. Middleware provides hooks for before/after model calls, tool call interception, and agent-level events.

'''Available hook points:'''
* `before_agent` / `after_agent` - Run once at start/end
* `before_model` / `after_model` - Run each model iteration
* `wrap_model_call` - Intercept model execution (retry, caching)
* `wrap_tool_call` - Intercept tool execution

=== Step 4: Response Format Configuration ===
[[step::Principle:langchain-ai_langchain_Structured_Output_Strategy]]

Configure structured output strategy if validated responses are required. Three strategies are available: `ToolStrategy` (adds schema as a tool), `ProviderStrategy` (uses native provider JSON mode), or `AutoStrategy` (auto-selects based on model capabilities).

'''Strategy selection:'''
* AutoStrategy: Let the system choose based on model profile
* ToolStrategy: Works with all tool-calling models, supports retry on validation errors
* ProviderStrategy: More efficient for models with native structured output support

=== Step 5: Graph Construction ===
[[step::Principle:langchain-ai_langchain_StateGraph_Assembly]]

Build the LangGraph StateGraph that orchestrates the agent loop. The factory creates nodes for model invocation, tool execution, and middleware hooks, then connects them with conditional edges based on tool calls and jump directives.

'''Graph structure:'''
* START → before_agent (optional) → before_model (optional) → model
* model → after_model (optional) → tools (if tool calls) → model (loop)
* model → after_agent (optional) → END (if no tool calls)

=== Step 6: Agent Execution ===
[[step::Principle:langchain-ai_langchain_Agent_Loop_Execution]]

Execute the compiled graph with user input messages. The agent loops through model invocations and tool executions until reaching a terminal condition: no more tool calls, a structured response is generated, or a middleware jumps to end.

'''Execution modes:'''
* `invoke()` / `ainvoke()` - Run to completion, return final state
* `stream()` / `astream()` - Yield incremental updates for each node
* `batch()` / `abatch()` - Process multiple inputs in parallel

== Execution Diagram ==
{{#mermaid:graph TD
    A[Model Initialization] --> B[Tool Definition]
    B --> C[Middleware Composition]
    C --> D[Response Format Configuration]
    D --> E[Graph Construction]
    E --> F[Agent Execution]
    F --> G{Has Tool Calls?}
    G -->|Yes| H[Execute Tools]
    H --> F
    G -->|No| I[Return Response]
}}

== Related Pages ==
* [[step::Principle:langchain-ai_langchain_Chat_Model_Initialization]]
* [[step::Principle:langchain-ai_langchain_Tool_Definition]]
* [[step::Principle:langchain-ai_langchain_Middleware_Composition]]
* [[step::Principle:langchain-ai_langchain_Structured_Output_Strategy]]
* [[step::Principle:langchain-ai_langchain_StateGraph_Assembly]]
* [[step::Principle:langchain-ai_langchain_Agent_Loop_Execution]]
