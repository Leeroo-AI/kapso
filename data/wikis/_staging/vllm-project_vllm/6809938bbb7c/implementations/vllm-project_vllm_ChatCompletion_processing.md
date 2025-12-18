{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|OpenAI API Reference|https://platform.openai.com/docs/api-reference/chat]]
|-
! Domains
| [[domain::NLP]], [[domain::Response_Processing]], [[domain::API_Integration]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Pattern documentation for extracting generated text, tool calls, usage statistics, and metadata from ChatCompletion responses.

=== Description ===

`ChatCompletion` is the response object returned by the chat completions API. It contains:

- **choices:** List of generated completions (usually 1 unless n > 1)
- **usage:** Token count statistics (prompt, completion, total)
- **model:** The model that generated the response
- **created:** Unix timestamp of response creation
- **id:** Unique response identifier

This pattern covers extracting useful data from both regular and streaming responses.

=== Usage ===

Process ChatCompletion responses to:
- Extract generated text for display or downstream processing
- Handle tool calls for agent workflows
- Track token usage for cost estimation
- Log request/response pairs for debugging
- Handle streaming chunks for real-time UX

== Code Reference ==

=== Source Location ===
* '''Library:''' [https://github.com/openai/openai-python openai (PyPI)]
* '''Types:''' openai.types.chat.ChatCompletion, ChatCompletionChunk

=== Interface Specification ===
<syntaxhighlight lang="python">
# ChatCompletion structure
class ChatCompletion:
    id: str                              # Unique response ID
    object: str                          # "chat.completion"
    created: int                         # Unix timestamp
    model: str                           # Model used
    choices: list[Choice]                # Generated completions
    usage: Usage                         # Token statistics

class Choice:
    index: int                           # Choice index (for n > 1)
    message: ChatCompletionMessage       # Generated message
    finish_reason: str                   # "stop", "length", "tool_calls"

class ChatCompletionMessage:
    role: str                            # "assistant"
    content: str | None                  # Generated text
    tool_calls: list[ToolCall] | None    # Function calls

class Usage:
    prompt_tokens: int                   # Input token count
    completion_tokens: int               # Output token count
    total_tokens: int                    # Sum of above

# Streaming: ChatCompletionChunk
class ChatCompletionChunk:
    choices: list[ChunkChoice]

class ChunkChoice:
    delta: ChoiceDelta                   # Incremental content
    finish_reason: str | None            # Set on final chunk

class ChoiceDelta:
    role: str | None                     # Set on first chunk
    content: str | None                  # Token(s) in this chunk
    tool_calls: list[ToolCallChunk] | None
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| response || ChatCompletion || Yes || Response from chat.completions.create()
|-
| stream || Stream[ChatCompletionChunk] || Alt || Streaming response generator
|}

=== Outputs (Accessible Fields) ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| choices[0].message.content || str || Generated text
|-
| choices[0].message.tool_calls || list || Function calls to execute
|-
| choices[0].finish_reason || str || "stop", "length", "tool_calls"
|-
| usage.total_tokens || int || Total tokens consumed
|}

== Usage Examples ==

=== Basic Text Extraction ===
<syntaxhighlight lang="python">
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Extract generated text
text = response.choices[0].message.content
print(f"Response: {text}")

# Check finish reason
finish_reason = response.choices[0].finish_reason
if finish_reason == "stop":
    print("Completed normally")
elif finish_reason == "length":
    print("Truncated due to max_tokens")
</syntaxhighlight>

=== Token Usage Tracking ===
<syntaxhighlight lang="python">
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    messages=[{"role": "user", "content": "Write a paragraph about AI."}],
    max_tokens=100,
)

# Track token usage
usage = response.usage
print(f"Prompt tokens: {usage.prompt_tokens}")
print(f"Completion tokens: {usage.completion_tokens}")
print(f"Total tokens: {usage.total_tokens}")

# Estimate cost (example rates)
cost_per_1k_input = 0.0015
cost_per_1k_output = 0.002
cost = (usage.prompt_tokens / 1000 * cost_per_1k_input +
        usage.completion_tokens / 1000 * cost_per_1k_output)
print(f"Estimated cost: ${cost:.6f}")
</syntaxhighlight>

=== Handling Multiple Choices (n > 1) ===
<syntaxhighlight lang="python">
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    messages=[{"role": "user", "content": "Give me a creative name for a cat."}],
    n=3,  # Generate 3 different completions
    temperature=0.9,
)

print("Generated names:")
for i, choice in enumerate(response.choices):
    print(f"  {i + 1}. {choice.message.content}")
</syntaxhighlight>

=== Processing Tool Calls ===
<syntaxhighlight lang="python">
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

tools = [{"type": "function", "function": {...}}]

response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
)

message = response.choices[0].message

if message.tool_calls:
    for tool_call in message.tool_calls:
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)
        print(f"Call function: {func_name}")
        print(f"  Arguments: {func_args}")
        print(f"  Tool call ID: {tool_call.id}")
else:
    print(f"Response: {message.content}")
</syntaxhighlight>

=== Streaming Response Processing ===
<syntaxhighlight lang="python">
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

stream = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    messages=[{"role": "user", "content": "Write a story."}],
    stream=True,
    max_tokens=200,
)

full_response = ""
for chunk in stream:
    if chunk.choices[0].delta.content:
        token = chunk.choices[0].delta.content
        full_response += token
        print(token, end="", flush=True)

    # Check if stream is complete
    if chunk.choices[0].finish_reason:
        print(f"\n\n[Finished: {chunk.choices[0].finish_reason}]")

print(f"\nTotal length: {len(full_response)} chars")
</syntaxhighlight>

=== Error Response Handling ===
<syntaxhighlight lang="python">
from openai import OpenAI, APIError, RateLimitError

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

try:
    response = client.chat.completions.create(
        model="nonexistent-model",
        messages=[{"role": "user", "content": "Hello"}],
    )
except RateLimitError as e:
    print(f"Rate limited. Retry after: {e.response.headers.get('Retry-After')}")
except APIError as e:
    print(f"API Error {e.status_code}: {e.message}")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Response_Handling]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_Client_Environment]]
