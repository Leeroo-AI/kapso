# Implementation: count_tokens

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Context_Length]], [[domain::Token_Management]], [[domain::Agent_Systems]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:30 GMT]]
|}

== Overview ==

Method to count tokens in a conversation message list using the HuggingFace transformers AutoTokenizer.

=== Description ===

The `count_tokens()` method accurately counts tokens in the agent's conversation history. It uses the model's actual tokenizer and chat template to ensure the count matches what the model will see.

Key implementation details:
- Loads tokenizer from the configured model path
- Applies the model's chat template to messages
- Returns the token count of the formatted prompt

This method is called after each round in the ReAct loop to check if the context limit has been reached.

=== Usage ===

Use `count_tokens()` when:
- Monitoring context length during agent execution
- Deciding whether to force answer generation
- Debugging context overflow issues

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba-NLP/DeepResearch]
* '''File:''' inference/react_agent.py
* '''Lines:''' 112-118

=== Signature ===
<syntaxhighlight lang="python">
def count_tokens(self, messages: List[Dict]) -> int:
    """
    Count tokens in a message list using the model's tokenizer.

    Args:
        messages: List[Dict] - Conversation history with format:
            [{"role": "system"|"user"|"assistant", "content": str}, ...]

    Returns:
        int: Total token count after applying chat template

    Note:
        - Uses AutoTokenizer from the model path stored in self.llm_local_path
        - Applies chat template for accurate counting
        - Creates a new tokenizer instance on each call (could be optimized)
    """
    tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path)
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    tokens = tokenizer(full_prompt, return_tensors="pt")
    token_count = len(tokens["input_ids"][0])

    return token_count
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer
from react_agent import MultiTurnReactAgent
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| messages || List[Dict] || Yes || Conversation history list
|-
| messages[i]["role"] || str || Yes || Role: "system", "user", or "assistant"
|-
| messages[i]["content"] || str || Yes || Message content
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| token_count || int || Total number of tokens in the formatted prompt
|}

== Usage Examples ==

=== Basic Token Counting ===
<syntaxhighlight lang="python">
from react_agent import MultiTurnReactAgent

# Initialize agent
llm_config = {
    "model": "/models/Qwen2.5-72B-Instruct",
    "generate_cfg": {"temperature": 0.6}
}
agent = MultiTurnReactAgent(llm=llm_config)

# Sample messages
messages = [
    {"role": "system", "content": "You are a helpful research assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
]

token_count = agent.count_tokens(messages)
print(f"Token count: {token_count}")
</syntaxhighlight>

=== Monitoring in ReAct Loop ===
<syntaxhighlight lang="python">
from react_agent import MultiTurnReactAgent

agent = MultiTurnReactAgent(llm=llm_config)

MAX_TOKENS = 110 * 1024  # 110K tokens

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": question}
]

for round_num in range(100):
    # Generate response
    response = agent.call_server(messages, planning_port)
    messages.append({"role": "assistant", "content": response})

    # Execute tool call and add observation
    if has_tool_call(response):
        result = execute_tool(response)
        messages.append({"role": "user", "content": f"<tool_response>{result}</tool_response>"})

    # Check token limit
    token_count = agent.count_tokens(messages)
    print(f"Round {round_num}: {token_count} tokens")

    if token_count > MAX_TOKENS:
        print("Token limit reached, forcing answer...")
        messages[-1]['content'] = FORCE_ANSWER_PROMPT
        break

    if has_answer(response):
        break
</syntaxhighlight>

=== Comparing Different Message Lengths ===
<syntaxhighlight lang="python">
from react_agent import MultiTurnReactAgent

agent = MultiTurnReactAgent(llm=llm_config)

# Short conversation
short_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"}
]

# Long conversation with tool results
long_messages = short_messages + [
    {"role": "user", "content": "<tool_response>" + "A" * 10000 + "</tool_response>"},
    {"role": "assistant", "content": "I found the information."},
    {"role": "user", "content": "<tool_response>" + "B" * 10000 + "</tool_response>"},
]

short_count = agent.count_tokens(short_messages)
long_count = agent.count_tokens(long_messages)

print(f"Short conversation: {short_count} tokens")
print(f"Long conversation: {long_count} tokens")
print(f"Difference: {long_count - short_count} tokens")
</syntaxhighlight>

=== Estimating Remaining Capacity ===
<syntaxhighlight lang="python">
from react_agent import MultiTurnReactAgent

agent = MultiTurnReactAgent(llm=llm_config)

MAX_TOKENS = 110 * 1024

def get_remaining_capacity(messages):
    """Calculate how many more tokens we can use."""
    current = agent.count_tokens(messages)
    remaining = MAX_TOKENS - current
    return remaining, current / MAX_TOKENS * 100

messages = [...]  # Current conversation

remaining, usage_pct = get_remaining_capacity(messages)
print(f"Remaining capacity: {remaining:,} tokens")
print(f"Current usage: {usage_pct:.1f}%")

if usage_pct > 80:
    print("Warning: Approaching context limit")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Alibaba_NLP_DeepResearch_Context_Management]]

=== Related Implementations ===
* [[related_to::Implementation:Alibaba_NLP_DeepResearch_MultiTurnReactAgent__run]]
* [[related_to::Implementation:Alibaba_NLP_DeepResearch_MultiTurnReactAgent__init__]]

=== Requires Environment ===
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_Python_Dependencies]]

=== Uses Heuristic ===
* [[uses_heuristic::Heuristic:Alibaba_NLP_DeepResearch_Token_Limit_Management]]
