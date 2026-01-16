# Principle: Multi_Turn_Agent_Loop

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::File|agent_eval.py|WebAgent/WebWatcher/infer/scripts_eval/agent_eval.py]]
* [[source::Paper|ReAct: Synergizing Reasoning and Acting|https://arxiv.org/abs/2210.03629]]
|-
! Domains
| [[domain::Multimodal]], [[domain::Agent_Systems]], [[domain::Vision_Language]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

Multimodal agent loop for vision-language question answering. Orchestrates tool calls (VLSearch, web_search, visit, code) until answer is produced.

=== Description ===

The Multi-Turn Agent Loop is a ReAct-style orchestration pattern that enables vision-language models to iteratively reason and act to answer complex questions about images. The loop continues until the model produces a final answer or reaches a step limit.

Key components of the loop:

1. **Initial Setup** - Process input image, construct system prompt with tool definitions, create initial user message with image and question
2. **Iterative Reasoning** - Model produces thinking (in `<think>` tags), tool calls (in `<tool_call>` tags), or answers (in `<answer>` tags)
3. **Tool Execution** - Parse and route tool calls to appropriate handlers, append results to conversation
4. **Termination** - Loop ends when model outputs `<answer>` or step limit (12 steps) is reached

Available tools in the multimodal pipeline:
- '''VLSearchImage''' - Reverse image search for visual similarity
- '''web_search''' - Text-based web search via Serper API
- '''visit''' - Webpage content retrieval with LLM summarization
- '''code_interpreter''' - Python code execution in sandbox

=== Usage ===

Use the multi-turn agent loop when:
- Answering complex visual questions requiring external knowledge
- Performing multimodal research tasks that need web information
- Running inference on vision-language benchmarks (HLE, GAIA, MMSearch)

The loop provides systematic exploration of information sources while maintaining conversation context.

== Theoretical Basis ==

The agent loop follows the ReAct paradigm:

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# Multi-turn agent loop
def run_main(sample) -> (status, messages, answer):
    # Step 1: Setup
    image, data_url = process_image(sample["file_path"])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": sample["prompt"]}
        ]}
    ]

    # Step 2: Agent loop
    for step in range(max_steps):
        # Get model response
        response = llm.chat(messages)
        messages.append({"role": "assistant", "content": response})

        # Parse response
        if "<answer>" in response:
            answer = extract_answer(response)
            return "success", messages, answer

        if "<tool_call>" in response:
            tool_name, tool_args = parse_tool_call(response)

            # Route to tool handler
            if tool_name == "VLSearchImage":
                result = vl_search.call(tool_args)
            elif tool_name == "web_search":
                result = web_search.call(tool_args)
            elif tool_name == "visit":
                result = visit.call(tool_args)
            elif tool_name == "code":
                result = run_code_in_sandbox(tool_args)

            messages.append({"role": "user", "content": f"<tool_response>{result}</tool_response>"})

    # Step 3: Force final answer if step limit reached
    messages.append({"role": "user", "content": "Please provide your final answer now."})
    response = llm.chat(messages)
    answer = extract_answer(response)

    return "max_steps", messages, answer
</syntaxhighlight>

The tag-based parsing enables clean separation of:
- '''`<think>`''' - Chain-of-thought reasoning (optional)
- '''`<tool_call>`''' - Structured tool invocation
- '''`<answer>`''' - Final response to user

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_OmniSearch_run_main]]
