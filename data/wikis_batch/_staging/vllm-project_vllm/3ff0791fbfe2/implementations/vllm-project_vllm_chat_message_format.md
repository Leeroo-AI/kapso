# Chat Message Format (Pattern)

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|OpenAI Chat API|https://platform.openai.com/docs/api-reference/chat]]
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Serving]], [[domain::Input_Processing]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:30 GMT]]
|}

== Overview ==

Interface specification for structuring chat messages in OpenAI-compatible format for vLLM's chat completions API.

=== Description ===

This is a **Pattern Doc** documenting the user-defined message structure expected by vLLM's `/v1/chat/completions` endpoint. Messages are dictionaries following the OpenAI Chat API specification.

=== Usage ===

Structure messages when:
* Calling the chat completions API
* Building conversation history
* Implementing system prompts
* Creating multi-turn interactions

== Interface Specification ==

=== Required Signature ===
<syntaxhighlight lang="python">
# Message structure (TypedDict-style)
message = {
    "role": str,      # Required: "system", "user", "assistant", or "tool"
    "content": str,   # Required: The message text
}

# Optional fields
message = {
    "role": str,
    "content": str | list[dict],  # Can be multimodal content parts
    "name": str,                   # Optional: Speaker identifier
    "tool_calls": list[dict],      # Optional: Function calls (assistant only)
    "tool_call_id": str,           # Optional: For tool responses
}
</syntaxhighlight>

=== Constraints ===
* `role` must be one of: "system", "user", "assistant", "tool"
* `content` is required (can be empty string)
* Messages should be ordered chronologically
* System message (if any) should be first
* Conversation should alternate between user/assistant

== Usage Examples ==

=== Simple Single Turn ===
<syntaxhighlight lang="python">
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

messages = [
    {"role": "user", "content": "What is 2 + 2?"}
]

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=messages,
)
</syntaxhighlight>

=== With System Prompt ===
<syntaxhighlight lang="python">
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that responds concisely."
    },
    {
        "role": "user",
        "content": "Explain quantum computing."
    }
]

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=messages,
)
</syntaxhighlight>

=== Multi-Turn Conversation ===
<syntaxhighlight lang="python">
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is calculus?"},
    {"role": "assistant", "content": "Calculus is a branch of mathematics..."},
    {"role": "user", "content": "Can you give me an example?"},
]

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=messages,
)

# Append response for continued conversation
messages.append({
    "role": "assistant",
    "content": response.choices[0].message.content
})
</syntaxhighlight>

=== With Multimodal Content (Vision) ===
<syntaxhighlight lang="python">
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"}
            }
        ]
    }
]

response = client.chat.completions.create(
    model="llava-hf/llava-1.5-7b-hf",
    messages=messages,
)
</syntaxhighlight>

=== Function Calling Pattern ===
<syntaxhighlight lang="python">
messages = [
    {"role": "user", "content": "What's the weather in Paris?"},
]

# First call - model decides to call function
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=messages,
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {...}}
        }
    }]
)

# If model called a function, add tool response
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    messages.append(response.choices[0].message)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": '{"temperature": 22, "condition": "sunny"}'
    })

    # Continue conversation with tool result
    final_response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=messages,
    )
</syntaxhighlight>

=== Conversation Manager Pattern ===
<syntaxhighlight lang="python">
class ChatSession:
    def __init__(self, client, model, system_prompt=None):
        self.client = client
        self.model = model
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def send(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )

        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message

# Usage
session = ChatSession(client, "meta-llama/Llama-3.1-8B-Instruct", "You are helpful.")
print(session.send("Hello!"))
print(session.send("What did I just say?"))  # Has context
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Chat_Formatting]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_CUDA_Environment]]
