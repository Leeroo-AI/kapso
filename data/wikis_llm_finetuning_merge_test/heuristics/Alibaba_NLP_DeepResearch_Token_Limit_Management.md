# Heuristic: Token_Limit_Management

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::Experience|Internal|Code analysis of react_agent.py]]
|-
! Domains
| [[domain::LLM_Agents]], [[domain::Context_Management]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==
Context management strategy using a 110K token limit to force answer generation before exceeding model context window.

=== Description ===
The ReAct agent accumulates context across multiple reasoning and tool call cycles. To prevent context overflow and ensure the model can still generate a coherent response, a hard token limit (110K tokens by default) triggers forced answer generation. When this limit is reached, the agent injects a prompt instructing the model to stop making tool calls and provide its best answer based on gathered information.

=== Usage ===
Use this heuristic when the agent has been running for many turns and accumulated significant context. The forced answer generation prevents OOM errors and ensures the user always receives a response, even if the research is incomplete.

== The Insight (Rule of Thumb) ==
* **Action:** Monitor token count after each turn using `count_tokens()`. When count exceeds `max_tokens`, inject forced answer prompt.
* **Value:** Default `max_tokens = 110 * 1024` (110K tokens). Configurable via environment.
* **Trade-off:** May produce incomplete answers if research is cut short, but guarantees a response.
* **Trigger Message:** "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer."

== Reasoning ==
LLMs have finite context windows. Qwen models used by DeepResearch typically support 128K tokens, but leaving buffer for output generation is critical. The 110K threshold:

1. **Prevents overflow:** Leaves ~18K tokens for response generation
2. **Ensures response:** Forces answer generation rather than silent failure
3. **Preserves quality:** Model can still reference all gathered context
4. **Handles gracefully:** Includes format error fallback if answer tags missing

Empirically, most research tasks complete well under this limit, but complex multi-hop queries can accumulate significant context through web page visits.

== Code Evidence ==

Token limit check and forced answer from `react_agent.py:186-209`:
<syntaxhighlight lang="python">
max_tokens = 110 * 1024
token_count = self.count_tokens(messages)
print(f"round: {round}, token count: {token_count}")

if token_count > max_tokens:
    print(f"Token quantity exceeds the limit: {token_count} > {max_tokens}")

    messages[-1]['content'] = "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:<think>your final thinking</think>\n<answer>your answer</answer>"
    content = self.call_server(messages, planning_port)
    messages.append({"role": "assistant", "content": content.strip()})
    if '<answer>' in content and '</answer>' in content:
        prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
        termination = 'generate an answer as token limit reached'
    else:
        prediction = messages[-1]['content']
        termination = 'format error: generate an answer as token limit reached'
</syntaxhighlight>

Token counting implementation from `react_agent.py:112-118`:
<syntaxhighlight lang="python">
def count_tokens(self, messages):
    tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path)
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    tokens = tokenizer(full_prompt, return_tensors="pt")
    token_count = len(tokens["input_ids"][0])

    return token_count
</syntaxhighlight>

Alternative limit for WebSailor from `WebSailor/src/react_agent.py:16`:
<syntaxhighlight lang="python">
MAX_TOKEN_LENGTH = int(os.getenv('MAX_LENGTH', 31 * 1024 - 500))
</syntaxhighlight>

== Related Pages ==
* [[used_by::Implementation:Alibaba_NLP_DeepResearch_MultiTurnReactAgent__run]]
* [[used_by::Implementation:Alibaba_NLP_DeepResearch_count_tokens]]
* [[used_by::Principle:Alibaba_NLP_DeepResearch_Context_Management]]
