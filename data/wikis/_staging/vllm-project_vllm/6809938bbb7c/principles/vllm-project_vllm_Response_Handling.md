{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|OpenAI API Reference|https://platform.openai.com/docs/api-reference/chat]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Response_Processing]], [[domain::API_Design]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

The practice of parsing, validating, and utilizing responses from chat completion APIs including text extraction, tool call handling, and error management.

=== Description ===

Response Handling encompasses all the steps needed to work with API responses in client applications:

1. **Text Extraction:** Getting the generated content from response objects
2. **Tool Call Processing:** Detecting and executing function calls
3. **Streaming Handling:** Accumulating and displaying incremental tokens
4. **Error Management:** Handling API errors, rate limits, and timeouts
5. **Usage Tracking:** Monitoring token consumption for cost management
6. **Logging/Debugging:** Recording request/response pairs for analysis

Robust response handling is essential for building reliable LLM-powered applications.

=== Usage ===

Implement response handling when:
- Building production applications with LLMs
- Creating agent workflows with tool calling
- Implementing real-time streaming UX
- Tracking usage and costs
- Debugging prompt/response quality issues

== Theoretical Basis ==

'''Response Structure:'''

<syntaxhighlight lang="python">
# ChatCompletion response hierarchy
ChatCompletion
├── id: str           # "chatcmpl-abc123"
├── object: str       # "chat.completion"
├── created: int      # Unix timestamp
├── model: str        # Model that responded
├── choices: [
│   └── Choice
│       ├── index: int
│       ├── finish_reason: str
│       └── message: Message
│           ├── role: str
│           ├── content: str | None
│           └── tool_calls: [...] | None
│   ]
└── usage: Usage
    ├── prompt_tokens: int
    ├── completion_tokens: int
    └── total_tokens: int
</syntaxhighlight>

'''Finish Reasons:'''

| Reason | Meaning | Action |
|--------|---------|--------|
| "stop" | Natural completion or stop sequence hit | Normal handling |
| "length" | Hit max_tokens limit | May need to continue |
| "tool_calls" | Model wants to call tools | Execute tools, continue |
| "content_filter" | Content filtered | Handle appropriately |

'''Streaming Accumulation:'''

<syntaxhighlight lang="python">
# Accumulate streaming response
def process_stream(stream):
    full_content = ""
    tool_calls = []

    for chunk in stream:
        delta = chunk.choices[0].delta

        # Accumulate content
        if delta.content:
            full_content += delta.content
            yield delta.content  # For real-time display

        # Accumulate tool calls
        if delta.tool_calls:
            for tc in delta.tool_calls:
                # Tool calls come in pieces
                accumulate_tool_call(tool_calls, tc)

        # Check completion
        if chunk.choices[0].finish_reason:
            return full_content, tool_calls
</syntaxhighlight>

'''Error Handling Strategy:'''

<syntaxhighlight lang="python">
import time
from openai import APIError, RateLimitError, APITimeoutError

def robust_completion(client, **kwargs):
    max_retries = 3
    backoff = 1

    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)

        except RateLimitError as e:
            # Exponential backoff for rate limits
            wait = backoff * (2 ** attempt)
            time.sleep(wait)

        except APITimeoutError:
            # Retry timeouts with same backoff
            wait = backoff * (2 ** attempt)
            time.sleep(wait)

        except APIError as e:
            if e.status_code >= 500:
                # Retry server errors
                time.sleep(backoff)
            else:
                # Don't retry client errors
                raise

    raise Exception("Max retries exceeded")
</syntaxhighlight>

'''Usage Tracking Pattern:'''

<syntaxhighlight lang="python">
# Track cumulative usage across requests
class UsageTracker:
    def __init__(self):
        self.total_prompt = 0
        self.total_completion = 0
        self.request_count = 0

    def record(self, response):
        if response.usage:
            self.total_prompt += response.usage.prompt_tokens
            self.total_completion += response.usage.completion_tokens
            self.request_count += 1

    def get_cost(self, input_rate, output_rate):
        return (self.total_prompt / 1000 * input_rate +
                self.total_completion / 1000 * output_rate)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_ChatCompletion_processing]]
