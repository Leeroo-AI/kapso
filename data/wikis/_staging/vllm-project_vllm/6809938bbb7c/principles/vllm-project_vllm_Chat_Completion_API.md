{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|OpenAI Chat API Reference|https://platform.openai.com/docs/api-reference/chat]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Chat_API]], [[domain::API_Design]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

The standardized HTTP API interface for sending conversational messages to an LLM and receiving generated responses.

=== Description ===

The Chat Completion API is the primary interface for interacting with instruction-tuned and chat-optimized language models. Unlike the legacy completion API that accepts raw text, the chat API uses structured message objects with roles:

- **system:** Defines the assistant's behavior and personality
- **user:** Contains user inputs and queries
- **assistant:** Contains previous model responses (for context)
- **tool:** Contains results from tool/function calls

This structured format enables proper prompt formatting across different models and supports advanced features like tool calling and multi-turn conversations.

=== Usage ===

Use the Chat Completion API when:
- Building conversational applications (chatbots)
- Using instruction-tuned models (e.g., Llama-Instruct, ChatGPT)
- Implementing multi-turn dialogue systems
- Creating tool-using AI agents
- Requiring streaming responses for UX

Prefer this over the legacy completion API for all instruction-following tasks.

== Theoretical Basis ==

'''Message Structure:'''

<syntaxhighlight lang="python">
# Standard message format
Message = {
    "role": str,      # "system" | "user" | "assistant" | "tool"
    "content": str,   # The message text
    "name": str,      # Optional: for multi-user scenarios
    "tool_calls": [],  # Optional: function calls by assistant
    "tool_call_id": str,  # Optional: for tool responses
}
</syntaxhighlight>

'''Chat Template Application:'''

Models expect specific formatting. The chat template converts messages:

<syntaxhighlight lang="python">
# Conceptual chat template (Llama-style)
def apply_template(messages):
    text = "<|begin_of_text|>"
    for msg in messages:
        text += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
        text += msg["content"]
        text += "<|eot_id|>"
    # Add generation prompt
    text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return text
</syntaxhighlight>

'''Request/Response Flow:'''

<syntaxhighlight lang="text">
Client                    vLLM Server
  |                           |
  |-- POST /v1/chat/completions -->|
  |   {                       |
  |     "model": "...",       |
  |     "messages": [...],    |    1. Apply chat template
  |     "temperature": 0.7    |    2. Tokenize prompt
  |   }                       |    3. Generate tokens
  |                           |    4. Detokenize output
  |<-- ChatCompletion --------|    5. Format response
  |   {                       |
  |     "choices": [{         |
  |       "message": {...}    |
  |     }],                   |
  |     "usage": {...}        |
  |   }                       |
</syntaxhighlight>

'''Streaming Protocol:'''

For streaming, response chunks arrive as Server-Sent Events:

<syntaxhighlight lang="text">
data: {"choices": [{"delta": {"role": "assistant"}}]}

data: {"choices": [{"delta": {"content": "The"}}]}

data: {"choices": [{"delta": {"content": " capital"}}]}

data: {"choices": [{"delta": {"content": " is"}}]}

data: {"choices": [{"delta": {"content": " Paris"}}]}

data: {"choices": [{"finish_reason": "stop"}]}

data: [DONE]
</syntaxhighlight>

'''Tool Calling Flow:'''

<syntaxhighlight lang="python">
# Multi-step tool calling flow
# 1. User asks question
messages = [{"role": "user", "content": "What's the weather?"}]

# 2. Model responds with tool call
response = api.chat.completions.create(messages=messages, tools=tools)
# response.choices[0].message.tool_calls = [...]

# 3. Execute tool and add result
messages.append(response.choices[0].message)
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": '{"temperature": 72}',
})

# 4. Model gives final answer
response = api.chat.completions.create(messages=messages, tools=tools)
# "The weather is 72 degrees."
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_chat_completions_create]]
