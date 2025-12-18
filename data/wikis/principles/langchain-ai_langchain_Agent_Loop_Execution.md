{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|ReAct|https://arxiv.org/abs/2210.03629]]
* [[source::Doc|LangGraph|https://langchain-ai.github.io/langgraph/]]
* [[source::Doc|LangChain Agents|https://docs.langchain.com/oss/python/langchain/overview]]
|-
! Domains
| [[domain::LLM]], [[domain::Agents]], [[domain::Reasoning]], [[domain::Execution]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Execution pattern that iteratively cycles between model reasoning and tool execution until a termination condition is reached.

=== Description ===

Agent Loop Execution is the runtime pattern that drives tool-calling agents. The agent iteratively:
1. Calls the model with current context
2. Checks if model requested tool calls
3. If yes: executes tools and loops back to model
4. If no: terminates and returns result

This implements the ReAct (Reasoning + Acting) paradigm where LLMs interleave reasoning with external actions. The loop continues until:
* Model produces a final answer (no tool calls)
* Structured output is successfully parsed
* Maximum iterations reached
* Explicit termination signal

=== Usage ===

Use Agent Loop Execution when:
* Building autonomous agents that can use tools
* Creating systems that reason over multiple steps
* Implementing conversational agents with actions
* Developing research or task-completion agents

Key considerations:
* Set maximum iterations to prevent infinite loops
* Use middleware for call limits and monitoring
* Consider streaming for responsive UIs
* Enable checkpointing for long-running tasks

== Theoretical Basis ==

Agent Loop Execution implements the **ReAct** (Reasoning and Acting) paradigm.

'''1. The Core Loop'''

<syntaxhighlight lang="python">
# Pseudo-code for agent loop
def agent_loop(messages, model, tools, max_iterations=10):
    for iteration in range(max_iterations):
        # Step 1: Model reasoning
        response = model.invoke(messages, tools=tools)
        messages.append(response)

        # Step 2: Check for tool calls
        if not response.tool_calls:
            # No tools = final answer
            return messages

        # Step 3: Execute tools
        for tool_call in response.tool_calls:
            tool = find_tool(tools, tool_call.name)
            result = tool.invoke(tool_call.arguments)
            messages.append(ToolMessage(
                content=result,
                tool_call_id=tool_call.id
            ))

        # Loop back to model with tool results

    raise MaxIterationsError("Agent did not terminate")
</syntaxhighlight>

'''2. Message Flow'''

The message history grows with each iteration:

<syntaxhighlight lang="text">
Iteration 1:
  [User] "What's the weather in NYC and LA?"
  [AI]   tool_calls: [get_weather("NYC"), get_weather("LA")]
  [Tool] "NYC: 72°F, sunny"
  [Tool] "LA: 85°F, clear"

Iteration 2:
  [AI]   "The weather in NYC is 72°F and sunny. LA is 85°F and clear."
  (No tool calls = DONE)
</syntaxhighlight>

'''3. Termination Conditions'''

<syntaxhighlight lang="python">
def should_terminate(state, response, response_format):
    # 1. No tool calls = natural termination
    if not response.tool_calls:
        return True

    # 2. Structured output parsed successfully
    if response_format and state.get("structured_response"):
        return True

    # 3. Jump directive from middleware
    if state.get("jump_to") == "end":
        return True

    # 4. Iteration limit (handled externally)
    return False
</syntaxhighlight>

'''4. Tool Call Processing'''

<syntaxhighlight lang="python">
# Pseudo-code for tool execution
def process_tool_calls(tool_calls, tools, middleware):
    results = []
    for call in tool_calls:
        # Find matching tool
        tool = tools[call.name]

        # Wrap with middleware
        handler = tool.invoke
        for m in reversed(middleware):
            handler = lambda req, h=handler, m=m: m.wrap_tool_call(req, h)

        # Execute
        request = ToolCallRequest(tool=tool, arguments=call.arguments)
        result = handler(request)

        # Create tool message
        results.append(ToolMessage(
            content=str(result),
            tool_call_id=call.id,
            name=call.name
        ))

    return results
</syntaxhighlight>

'''5. Streaming Execution'''

<syntaxhighlight lang="python">
# Pseudo-code for streaming
def stream_agent_loop(messages, model, tools):
    while True:
        # Stream model response
        for chunk in model.stream(messages, tools=tools):
            yield {"type": "model_chunk", "content": chunk}

        response = aggregate_chunks(chunks)
        messages.append(response)

        if not response.tool_calls:
            yield {"type": "done", "messages": messages}
            return

        # Execute tools (can also be streamed)
        for tool_call in response.tool_calls:
            yield {"type": "tool_start", "call": tool_call}
            result = execute_tool(tool_call)
            yield {"type": "tool_end", "result": result}
            messages.append(ToolMessage(content=result, ...))
</syntaxhighlight>

'''6. Error Recovery'''

<syntaxhighlight lang="python">
def agent_loop_with_recovery(messages, model, tools, response_format):
    while True:
        try:
            response = model.invoke(messages, tools=tools)
            messages.append(response)

            # Parse structured output if configured
            if response_format and is_structured_output_call(response):
                try:
                    parsed = parse_structured_output(response, response_format)
                    return {"messages": messages, "structured_response": parsed}
                except ValidationError as e:
                    # Add error message and retry
                    if response_format.handle_errors:
                        messages.append(ToolMessage(
                            content=f"Error: {e}. Please try again.",
                            tool_call_id=response.tool_calls[-1].id
                        ))
                        continue

            if not response.tool_calls:
                return {"messages": messages}

            # Execute tools...

        except Exception as e:
            # Middleware can handle retries
            raise
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:langchain-ai_langchain_CompiledStateGraph_invocation]]

=== Used By Workflows ===
* Agent_Creation_Workflow (Step 6)
