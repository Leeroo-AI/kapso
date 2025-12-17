# Tool Definition

**Sources:**
- `langchain_core.tools` module (external to this monorepo)
- LangChain Documentation: Tools and Toolkits
- Usage examples across integration tests

**Domains:** Agent Architecture, Function Calling, LLM Tool Use

**Last Updated:** 2025-12-17

---

## Overview

Tool Definition is the principle of converting Python functions and capabilities into structured, LLM-invocable interfaces that language models can discover, reason about, and execute. This principle enables agents to extend their capabilities beyond text generation by interacting with external systems, APIs, databases, and computational resources.

## Description

The principle addresses the fundamental challenge of bridging between natural language reasoning (what LLMs do) and programmatic actions (what applications need). Tool Definition establishes a contract between:

1. **Tool declarations** - Metadata describing what a tool does, its parameters, and constraints
2. **Tool invocation** - The mechanism for LLMs to request tool execution with specific arguments
3. **Tool execution** - The actual implementation that performs the action and returns results
4. **Tool response handling** - Converting execution results back to natural language context

This principle is foundational to agentic systems because it determines:
- What capabilities an agent has available
- How the LLM understands and reasons about those capabilities
- The interface boundary between LLM reasoning and deterministic code
- Error handling and validation of tool calls

### Key Architectural Decisions

**Declarative Tool Interface**
Tools are defined declaratively with explicit schemas rather than through dynamic introspection, providing clear contracts that LLMs can understand and reason about consistently.

**Function-First Design**
The `@tool` decorator approach allows developers to define tools as regular Python functions, with the framework automatically generating the necessary metadata and validation logic. This reduces friction and keeps tool code maintainable.

**Structured Arguments via JSON Schema**
Tool parameters are described using JSON Schema (often generated from Pydantic models), creating a universal format that works across different LLM providers and their tool-calling implementations.

**Return Type Flexibility**
Tools can return various types (strings, objects, errors) which are normalized into `ToolMessage` objects that fit into the message-based conversation paradigm.

## Theoretical Basis

This principle draws from several computer science concepts:

**Interface Segregation Principle (ISP)**
From SOLID design principles - tools expose minimal, focused interfaces rather than monolithic capabilities. Each tool does one thing well with a clear contract.

**Adapter Pattern**
Tools act as adapters between the LLM's text-based interface and arbitrary Python code, translating between natural language requests and programmatic execution.

**Remote Procedure Call (RPC) Model**
Tool calling mirrors RPC systems where:
- The LLM acts as a client making procedure calls
- The tool implementation is the server executing procedures
- Tool messages carry serialized arguments and return values

**Capability-Based Security**
By explicitly declaring and providing tools to an agent, you implement capability-based access control - the agent can only interact with systems you've explicitly granted it access to.

## Usage

### When to Apply This Principle

Apply Tool Definition when:

- Building agents that need to interact with external systems (APIs, databases, file systems)
- Enabling LLMs to perform computations they cannot do natively (math, data analysis, code execution)
- Creating reusable capabilities that multiple agents can share
- Implementing deterministic behaviors that should not vary with LLM behavior

### When to Use Alternative Approaches

Consider alternatives when:

- **Simple Q&A**: If the LLM only needs to answer questions from its knowledge, tools add unnecessary complexity
- **Fully autonomous systems**: For fully automated systems without LLM involvement, use direct function calls
- **Real-time constraints**: Tool calling adds latency from LLM reasoning; consider rule-based systems for time-critical paths
- **Complex orchestration**: For complex multi-step workflows, consider workflow engines instead of tool chains

### Anti-Patterns to Avoid

1. **Overly granular tools**: Creating separate tools for every tiny operation leads to excessive LLM reasoning overhead
2. **Undocumented tools**: Tools without clear descriptions confuse LLMs and lead to misuse
3. **Stateful tools**: Tools that depend on hidden state across calls create unpredictable behavior
4. **Tools without error handling**: Unhandled exceptions from tools crash agent execution
5. **Security bypass tools**: Tools that give unconstrained access to sensitive systems without validation
6. **Monolithic tools**: A single tool that does many unrelated things is hard for LLMs to use correctly

### Best Practices

**Write Clear Tool Descriptions**
The tool's docstring is critical - LLMs use it to decide when and how to call the tool. Be specific about what the tool does, when to use it, and any constraints.

**Validate Tool Arguments**
Use Pydantic models or type hints with validation to catch malformed arguments before execution.

**Return Informative Results**
Tool responses should be clear and actionable for the LLM to incorporate into its reasoning.

**Keep Tools Focused**
Each tool should do one thing well. Compose complex behaviors from multiple focused tools.

## Related Pages

**Implementation:**
- [[implemented_by::Implementation:langchain-ai_langchain_BaseTool_creation]] - Implementation mechanisms for creating tools

**Related Principles:**
- [[langchain-ai_langchain_Middleware_Configuration]] - How tools integrate into agent middleware
- [[langchain-ai_langchain_Agent_Graph_Construction]] - How tools are bound to agent graphs

**Used In Workflows:**
- [[langchain-ai_langchain_Agent_Creation_and_Execution]] - Tool definition is Step 2
- [[langchain-ai_langchain_Middleware_Composition]] - Tools can be registered via middleware

**Related Implementations:**
- [[langchain-ai_langchain_ToolNode]] - Node that executes tool calls in agent graphs
- [[langchain-ai_langchain_wrap_tool_call]] - Middleware hook for intercepting tool execution

**Environment:**
- [[langchain-ai_langchain_Python]] - Python runtime for tool execution
