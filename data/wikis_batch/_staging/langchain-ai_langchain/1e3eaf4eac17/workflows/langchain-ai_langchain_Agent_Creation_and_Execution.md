# Workflow: Agent Creation and Execution

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Agents|https://docs.langchain.com/oss/python/langchain/agents]]
|-
! Domains
| [[domain::LLMs]], [[domain::Agents]], [[domain::Tool_Calling]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
End-to-end process for creating and executing LLM agents with tool-calling capabilities using the LangChain framework.

=== Description ===
This workflow outlines the standard procedure for building conversational agents that can use tools to accomplish tasks. The `create_agent` factory function constructs a state graph that implements an agent loop: the model is called, if it produces tool calls they are executed, and the results are fed back to the model until no more tool calls are made. The workflow supports structured output, middleware injection, and provider-agnostic model initialization.

=== Usage ===
Execute this workflow when you need to build an AI agent that can:
* Use tools to interact with external systems (APIs, databases, file systems)
* Engage in multi-turn conversations with memory
* Produce structured outputs conforming to a schema
* Apply custom logic before/after model calls via middleware

== Execution Steps ==

=== Step 1: Initialize Chat Model ===
[[step::Principle:langchain-ai_langchain_Chat_Model_Initialization]]

Configure the language model that will power the agent. The model can be specified as a string identifier (e.g., `"openai:gpt-4"`) or as a direct `BaseChatModel` instance. The factory supports 20+ providers with automatic provider inference from model names.

'''Key considerations:'''
* Model string format: `{provider}:{model_name}` or just `{model_name}` for auto-inference
* Provider packages must be installed (e.g., `langchain-openai`)
* Model settings like temperature and max_tokens can be passed as kwargs

=== Step 2: Define Tools ===
[[step::Principle:langchain-ai_langchain_Tool_Definition]]

Specify the tools the agent can use to accomplish tasks. Tools can be Python functions, `BaseTool` instances, or dictionaries for provider-specific built-in tools. Each tool should have a clear docstring describing its purpose, as this is used by the LLM to decide when to use it.

'''Key considerations:'''
* Functions are automatically converted to `BaseTool` instances
* Tools with `return_direct=True` end the agent loop immediately
* Middleware can inject additional tools dynamically

=== Step 3: Configure Middleware (Optional) ===
[[step::Principle:langchain-ai_langchain_Middleware_Configuration]]

Add middleware to customize agent behavior at various lifecycle points. Middleware can intercept model calls for retry logic, add human approval workflows, enforce tool call limits, manage context window size, and more.

'''Key considerations:'''
* Middleware is composable and executes in order (first = outermost)
* Use `before_model`, `after_model`, `wrap_model_call`, and `wrap_tool_call` hooks
* Middleware can extend the agent state schema with custom fields

=== Step 4: Configure Response Format (Optional) ===
[[step::Principle:langchain-ai_langchain_Structured_Output_Configuration]]

Optionally configure structured output to ensure the agent returns responses conforming to a Pydantic schema. The framework supports three strategies: ToolStrategy (tool call for output), ProviderStrategy (native structured output), and AutoStrategy (automatic selection based on model capabilities).

'''Key considerations:'''
* ToolStrategy works with any model that supports tool calling
* ProviderStrategy requires model support (e.g., OpenAI JSON mode)
* Error handling can be configured for retry on validation failures

=== Step 5: Create Agent Graph ===
[[step::Principle:langchain-ai_langchain_Agent_Graph_Construction]]

Call `create_agent()` to construct the agent as a compiled `StateGraph`. This builds the model node, tool node, and all edges including middleware nodes. The graph implements the agent loop with conditional routing based on tool calls and structured output.

'''Key considerations:'''
* The graph is compiled with configurable checkpointer for persistence
* Interrupt points can be set before/after specific nodes
* Debug mode enables verbose logging of execution flow

=== Step 6: Execute Agent ===
[[step::Principle:langchain-ai_langchain_Agent_Execution]]

Invoke or stream the agent with input messages. The agent maintains state across turns, executes tools as needed, and returns the final response. Execution can be synchronous (`invoke`, `stream`) or asynchronous (`ainvoke`, `astream`).

'''What happens:'''
* Input messages are added to state
* Model is called with system prompt and available tools
* Tool calls are executed in parallel
* Loop continues until no tool calls or structured output is produced
* Final state includes all messages and optional structured response

== Execution Diagram ==
{{#mermaid:graph TD
    A[Initialize Chat Model] --> B[Define Tools]
    B --> C[Configure Middleware]
    C --> D[Configure Response Format]
    D --> E[Create Agent Graph]
    E --> F[Execute Agent]
}}

== Related Pages ==
* [[step::Principle:langchain-ai_langchain_Chat_Model_Initialization]]
* [[step::Principle:langchain-ai_langchain_Tool_Definition]]
* [[step::Principle:langchain-ai_langchain_Middleware_Configuration]]
* [[step::Principle:langchain-ai_langchain_Structured_Output_Configuration]]
* [[step::Principle:langchain-ai_langchain_Agent_Graph_Construction]]
* [[step::Principle:langchain-ai_langchain_Agent_Execution]]
