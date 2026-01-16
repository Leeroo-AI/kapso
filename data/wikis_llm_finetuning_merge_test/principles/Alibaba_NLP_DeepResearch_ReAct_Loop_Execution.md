# Principle: ReAct_Loop_Execution

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|ReAct: Synergizing Reasoning and Acting in Language Models|https://arxiv.org/abs/2210.03629]]
* [[source::Paper|Toolformer: Language Models Can Teach Themselves to Use Tools|https://arxiv.org/abs/2302.04761]]
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Agent_Systems]], [[domain::NLP]], [[domain::Autonomous_Research]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:30 GMT]]
|}

== Overview ==

The iterative reasoning and action loop that drives autonomous web research. The agent generates thoughts, selects tools, receives observations, and continues until an answer is produced or limits are reached.

=== Description ===

The ReAct loop execution is the core runtime mechanism of autonomous research agents. It implements a continuous cycle of reasoning and acting that allows the agent to progressively gather information and synthesize answers to complex questions.

The execution loop encompasses several critical mechanisms:

1. **Thought Generation** - The LLM reasons about the current state and decides next steps within `<think>` tags
2. **Tool Selection** - Based on reasoning, the agent selects and parameterizes tool calls using `<tool_call>` XML tags
3. **Observation Processing** - Tool results are wrapped in `<tool_response>` tags and added to context
4. **Answer Detection** - The loop terminates when `<answer>` tags are detected in the response
5. **Safety Limits** - Maximum LLM calls (100) and time limits (150 minutes) prevent infinite loops

The loop maintains a conversation history as a list of messages, alternating between assistant (thought + action) and user (observation) roles.

=== Usage ===

Use ReAct Loop Execution when:
- Running autonomous research on complex, multi-step questions
- Implementing agentic workflows requiring iterative information gathering
- Building systems that need to reason about when to stop researching

Loop termination conditions:
| Condition | Result |
|-----------|--------|
| `<answer>` tag found | Normal completion with extracted answer |
| MAX_LLM_CALL_PER_RUN exceeded | Force answer generation or error |
| Token limit exceeded (110K) | Force final answer generation |
| Time limit exceeded (150 min) | Return "No answer found after 2h30mins" |

== Theoretical Basis ==

The ReAct execution follows an iterative refinement process modeled as:

<math>
\text{State}_{t+1} = f(\text{State}_t, \text{Action}_t, \text{Observation}_t)
</math>

Where each iteration produces:
<math>
\text{Thought}_t = \text{LLM}(\text{Context}_t)
</math>
<math>
\text{Action}_t = \text{Parse}(\text{Thought}_t)
</math>
<math>
\text{Observation}_t = \text{Tool}(\text{Action}_t)
</math>

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# ReAct Loop Execution Pattern
def react_loop(question: str, max_calls: int = 100, max_time: int = 9000):
    start_time = time.time()

    # Initialize conversation
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + today_date()},
        {"role": "user", "content": question}
    ]

    num_calls_remaining = max_calls
    round_number = 0

    while num_calls_remaining > 0:
        # Check time limit (150 minutes)
        if time.time() - start_time > max_time:
            return {"termination": "timeout", "prediction": "No answer found"}

        round_number += 1
        num_calls_remaining -= 1

        # Generate thought + action
        response = llm_call(messages)

        # Clean up any leaked tool_response tags
        if '<tool_response>' in response:
            response = response[:response.find('<tool_response>')]

        messages.append({"role": "assistant", "content": response})

        # Check for tool call
        if '<tool_call>' in response and '</tool_call>' in response:
            tool_result = execute_tool_call(response)
            observation = f"<tool_response>\n{tool_result}\n</tool_response>"
            messages.append({"role": "user", "content": observation})

        # Check for answer
        if '<answer>' in response and '</answer>' in response:
            answer = extract_between_tags(response, 'answer')
            return {"termination": "answer", "prediction": answer}

        # Check token limit
        token_count = count_tokens(messages)
        if token_count > MAX_TOKENS:
            # Force final answer generation
            messages[-1]['content'] = FORCE_ANSWER_PROMPT
            final_response = llm_call(messages)
            return {"termination": "token_limit", "prediction": extract_answer(final_response)}

    return {"termination": "max_calls_exceeded", "prediction": "No answer found"}
</syntaxhighlight>

Key execution principles:
- **Stateful Conversation**: All messages are preserved to maintain reasoning context
- **Graceful Degradation**: Multiple termination paths ensure the agent always produces output
- **Progress Logging**: Round numbers and token counts are logged for debugging
- **Tool Isolation**: Each tool call is executed synchronously and results are sanitized

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_MultiTurnReactAgent__run]]

=== Related Principles ===
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Agent_Initialization]]
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Context_Management]]
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Answer_Extraction]]
