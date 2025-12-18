{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|OpenAI Chat API Reference|https://platform.openai.com/docs/api-reference/chat]]
|-
! Domains
| [[domain::NLP]], [[domain::Chat_API]], [[domain::API_Integration]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Wrapper documentation for sending chat completion requests to a vLLM server using the OpenAI Python client.

=== Description ===

The `chat.completions.create()` method sends a conversation history to the vLLM server and receives a model-generated response. It supports:

- **Multi-turn conversations:** Pass message history for context
- **System prompts:** Define assistant behavior
- **Streaming responses:** Receive tokens as they're generated
- **Tool/Function calling:** Let the model call external functions
- **Structured output:** Constrain responses to JSON schemas

This is the recommended API for chat and instruction-following models.

=== Usage ===

Use `chat.completions.create()` when:
- Building chatbots and conversational AI
- Sending instructions to instruction-tuned models
- Implementing tool-using agents
- Streaming responses for better UX
- Getting structured (JSON) outputs

== Code Reference ==

=== Source Location ===
* '''Library:''' [https://github.com/openai/openai-python openai (PyPI)]
* '''vLLM Server Handler:''' vllm/entrypoints/openai/serving_chat.py

=== Signature ===
<syntaxhighlight lang="python">
def create(
    self,
    model: str,
    messages: list[dict],
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    n: int = 1,
    stream: bool = False,
    stop: str | list[str] | None = None,
    max_tokens: int | None = None,
    presence_penalty: float = 0,
    frequency_penalty: float = 0,
    logit_bias: dict[str, float] | None = None,
    user: str | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    response_format: dict | None = None,
) -> ChatCompletion | Stream[ChatCompletionChunk]:
    """
    Creates a chat completion.

    Args:
        model: Model ID to use.
        messages: Conversation history as list of message dicts.
        temperature: Sampling temperature (0-2).
        top_p: Nucleus sampling parameter.
        n: Number of completions to generate.
        stream: Enable streaming response.
        stop: Stop sequences.
        max_tokens: Maximum tokens to generate.
        presence_penalty: Penalty for token presence (-2 to 2).
        frequency_penalty: Penalty for token frequency (-2 to 2).
        tools: Tool definitions for function calling.
        response_format: Output format specification.

    Returns:
        ChatCompletion object or async generator for streaming.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
# Use: client.chat.completions.create(...)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || str || Yes || Model ID (from /v1/models)
|-
| messages || list[dict] || Yes || Conversation with role/content pairs
|-
| temperature || float || No || Sampling randomness (default: 1.0)
|-
| max_tokens || int || No || Maximum response length
|-
| stream || bool || No || Enable streaming (default: False)
|-
| tools || list[dict] || No || Function definitions for tool calling
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| ChatCompletion || object || Response with choices, usage, etc.
|-
| Stream || generator || Streaming chunks (if stream=True)
|}

== Usage Examples ==

=== Basic Chat Completion ===
<syntaxhighlight lang="python">
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    temperature=0.7,
    max_tokens=100,
)

print(response.choices[0].message.content)
# Output: "The capital of France is Paris."
</syntaxhighlight>

=== Multi-Turn Conversation ===
<syntaxhighlight lang="python">
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# Maintain conversation history
messages = [
    {"role": "system", "content": "You are a math tutor."},
]

# First turn
messages.append({"role": "user", "content": "What is 2 + 2?"})
response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    messages=messages,
)
assistant_msg = response.choices[0].message.content
messages.append({"role": "assistant", "content": assistant_msg})
print(f"Assistant: {assistant_msg}")

# Second turn (with context)
messages.append({"role": "user", "content": "And what is that times 3?"})
response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    messages=messages,
)
print(f"Assistant: {response.choices[0].message.content}")
# Output: "12" (model has context of previous answer being 4)
</syntaxhighlight>

=== Streaming Response ===
<syntaxhighlight lang="python">
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

stream = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    messages=[{"role": "user", "content": "Write a short story."}],
    stream=True,
    max_tokens=500,
)

# Print tokens as they arrive
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # Newline at end
</syntaxhighlight>

=== Tool/Function Calling ===
<syntaxhighlight lang="python">
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# Define available tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto",
)

# Check if model wants to call a tool
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    print(f"Tool: {tool_call.function.name}")
    print(f"Args: {tool_call.function.arguments}")
</syntaxhighlight>

=== JSON Mode ===
<syntaxhighlight lang="python">
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    messages=[
        {"role": "system", "content": "Output JSON only."},
        {"role": "user", "content": "List 3 programming languages with their year of creation."},
    ],
    response_format={"type": "json_object"},
    max_tokens=200,
)

data = json.loads(response.choices[0].message.content)
print(json.dumps(data, indent=2))
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Chat_Completion_API]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_Client_Environment]]
