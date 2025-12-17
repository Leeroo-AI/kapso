# Agent Execution

**Sources:**
- LangGraph Documentation (external framework)
- LangGraph API Reference: CompiledStateGraph
- Usage patterns across LangChain agents

**Domains:** Graph Execution, State Management, Async Programming

**Last Updated:** 2025-12-17

---

## Overview

Agent Execution is the principle of running compiled agent graphs through their reasoning-action loops, managing state transitions, handling interrupts, and streaming intermediate results. This principle bridges between the static graph definition and dynamic runtime execution.

## Description

The principle addresses the challenge of executing complex stateful computations while providing control, observability, and flexibility. Agent Execution establishes patterns for:

1. **Invocation modes** - Synchronous (`invoke`), asynchronous (`ainvoke`), streaming (`stream`, `astream`)
2. **State management** - Input validation, state updates, persistence via checkpointers
3. **Execution control** - Interrupts for human-in-the-loop, resumption after pauses
4. **Configuration** - Runtime parameters (thread_id, recursion limits, callbacks)
5. **Result retrieval** - Final state extraction, intermediate step access
6. **Error handling** - Graceful failures, retry mechanisms, exception propagation

The execution model follows a message-passing paradigm where:
- Initial state is provided as input
- Graph nodes process state and produce updates
- State updates are merged using reducer functions
- Execution continues until reaching an END node or interrupt
- Final state is returned containing complete conversation history

### Key Architectural Decisions

**Message-Based State**
State centers around a `messages` list that accumulates the conversation history. This makes agents naturally conversational and enables context preservation across turns.

**Streaming Support**
Agents can stream execution in multiple modes:
- `"values"`: Full state after each node
- `"updates"`: Only state changes from each node
- `"messages"`: Only message updates

**Configuration Objects**
Runtime behavior is controlled via `RunnableConfig` passed to invocation methods, enabling thread-specific settings without changing the graph.

**Checkpoint-Based Persistence**
State is persisted to checkpointers keyed by `thread_id`, enabling:
- Multi-turn conversations
- Long-running agent sessions
- Resume after interrupts
- Time travel debugging

**Interrupt Mechanisms**
Execution can pause before/after specific nodes, yielding control back to the caller for approval, modification, or inspection.

## Theoretical Basis

This principle draws from several computing paradigms:

**Coroutines and Generators**
Streaming execution mirrors coroutine/generator patterns where computation yields intermediate results before completion.

**Event-Driven Architecture**
Each node execution is an event that triggers state transitions, with the graph acting as an event dispatcher.

**Transactional Systems**
Checkpointing implements transaction-like semantics where state is durably persisted at consistent points.

**Process Calculus**
The graph execution model resembles process algebras like CSP or π-calculus, where concurrent processes communicate via message passing.

**Continuation-Passing Style (CPS)**
Interrupts and resumption implement continuation-based control flow where execution can be suspended and later resumed.

## Usage

### When to Apply This Principle

Apply Agent Execution patterns when:

- Building interactive conversational agents that maintain context
- Implementing multi-turn workflows with tool use
- Creating systems requiring human approval for certain actions
- Building agents that need to be debugged or traced
- Implementing long-running agent sessions with persistence

### When to Use Alternative Approaches

Consider alternatives when:

- **Batch processing**: Non-interactive workflows may not need the agent abstraction
- **Simple chains**: Linear workflows without branching can use simpler execution models
- **Real-time constraints**: Very low-latency requirements might need custom execution
- **Stateless operations**: Operations without context can use direct function calls

### Anti-Patterns to Avoid

1. **Ignoring configuration**: Not setting `thread_id` when using checkpointers loses conversation context
2. **Blocking in async**: Using synchronous blocking operations in async execution
3. **Unhandled interrupts**: Not resuming after interrupts leaves agents in incomplete states
4. **Excessive streaming**: Streaming every execution when only final results are needed adds overhead
5. **Missing error handling**: Not catching and handling exceptions during execution
6. **State mutation**: Directly mutating state objects instead of returning updates

### Best Practices

**Use Appropriate Invocation Mode**
- Use `invoke()` for single-turn interactions
- Use `stream()` when you need to display intermediate steps
- Use async variants (`ainvoke()`, `astream()`) for I/O-bound operations

**Set Thread IDs for Persistence**
Always provide `thread_id` in config when using checkpointers to maintain conversation continuity.

**Handle Interrupts Gracefully**
When using interrupts, ensure UI/code handles the pause and provides resumption paths.

**Stream for User Experience**
Streaming provides better UX for long-running operations, showing progress rather than waiting.

**Configure Recursion Limits**
Set appropriate `recursion_limit` to prevent infinite loops while allowing sufficient iterations.

## Related Pages

**Implementation:**
- [[implemented_by::Implementation:langchain-ai_langchain_CompiledStateGraph_invoke]] - Implementation details

**Related Principles:**
- [[langchain-ai_langchain_Agent_Graph_Construction]] - Creates the executable graph
- [[langchain-ai_langchain_Chat_Model_Initialization]] - Model used during execution
- [[langchain-ai_langchain_Tool_Definition]] - Tools executed during agent runs
- [[langchain-ai_langchain_Middleware_Configuration]] - Middleware hooks during execution

**Used In Workflows:**
- [[langchain-ai_langchain_Agent_Creation_and_Execution]] - Execution is Step 6

**Related Implementations:**
- [[langchain-ai_langchain_create_agent]] - Creates the graph to execute
- [[langchain-ai_langchain_Checkpointer]] - Persistence mechanism
- [[langchain-ai_langchain_RunnableConfig]] - Configuration object
- [[langchain-ai_langchain_StreamMode]] - Streaming configuration

**Environment:**
- [[langchain-ai_langchain_Python]] - Python runtime
- [[langchain-ai_langchain_LangGraph]] - Graph execution framework
