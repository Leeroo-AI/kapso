# Implementation: QwenChatAtDS

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::Doc|DashScope|https://dashscope.aliyun.com]]
|-
! Domains
| [[domain::LLM_Backend]], [[domain::DashScope]], [[domain::Qwen]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:00 GMT]]
|}

== Overview ==
DashScope API backend for Qwen models with streaming support, reasoning content extraction, and partial response continuation.

=== Description ===
The `QwenChatAtDS` class provides a DashScope-specific LLM backend registered as `'qwen_dashscope'` type. It enables:

- Direct access to Alibaba Cloud's Qwen models via DashScope API
- Streaming (delta and full) response modes
- Reasoning content extraction for chain-of-thought models
- Partial response continuation for assistant messages
- Automatic API key initialization from environment

The class extends `BaseFnCallModel` for Qwen-Agent compatibility.

=== Usage ===
Use as the LLM backend when accessing Qwen models through DashScope API, particularly for production deployments.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebDancer/demos/llm/qwen_dashscope.py WebAgent/WebDancer/demos/llm/qwen_dashscope.py]
* '''Lines:''' 1-140

=== Signature ===
<syntaxhighlight lang="python">
@register_llm('qwen_dashscope')
class QwenChatAtDS(BaseFnCallModel):
    """DashScope API backend for Qwen models."""

    def __init__(self, cfg: Optional[Dict] = None):
        """
        Initialize DashScope backend.

        Args:
            cfg: Configuration dict with keys:
                - model: Model name (default 'qwen-max')
                - api_key: DashScope API key
                - base_http_api_url: Custom HTTP endpoint
                - base_websocket_api_url: Custom WebSocket endpoint
        """
        ...

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict
    ) -> Iterator[List[Message]]:
        """Stream chat completion with delta or full mode."""
        ...

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict
    ) -> List[Message]:
        """Non-streaming chat completion."""
        ...

    def _continue_assistant_response(
        self,
        messages: List[Message],
        generate_cfg: dict,
        stream: bool
    ) -> Iterator[List[Message]]:
        """Continue partial assistant response."""
        ...

def initialize_dashscope(cfg: Optional[Dict] = None) -> None:
    """Initialize DashScope SDK with API key and endpoints."""
    ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from WebAgent.WebDancer.demos.llm.qwen_dashscope import QwenChatAtDS, initialize_dashscope
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || str || No || Model name (default qwen-max)
|-
| api_key || str || No || DashScope key (or DASHSCOPE_API_KEY env)
|-
| base_http_api_url || str || No || Custom HTTP endpoint
|-
| messages || List[Message] || Yes || Conversation history
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| response || List[Message] || Model response messages
|-
| content || str || Response text content
|-
| reasoning_content || str || Chain-of-thought content
|-
| extra || Dict || Model service info
|}

== Usage Examples ==

=== Basic DashScope Usage ===
<syntaxhighlight lang="python">
import os
from WebAgent.WebDancer.demos.llm.qwen_dashscope import QwenChatAtDS

# Set API key
os.environ['DASHSCOPE_API_KEY'] = 'sk-...'

# Initialize backend
llm = QwenChatAtDS({
    'model': 'qwen-max',
    'generate_cfg': {
        'temperature': 0.7,
        'max_tokens': 4096
    }
})

# Use in agent
from qwen_agent.agents import Assistant
agent = Assistant(llm=llm, function_list=['search'])
</syntaxhighlight>

=== Streaming Response ===
<syntaxhighlight lang="python">
from qwen_agent.llm.schema import Message

messages = [
    Message(role='user', content='Explain quantum computing')
]

# Delta streaming (token by token)
for chunk in llm._chat_stream(messages, delta_stream=True, generate_cfg={}):
    print(chunk[0].content, end='', flush=True)

# Full streaming (accumulated)
for chunk in llm._chat_stream(messages, delta_stream=False, generate_cfg={}):
    print(f"\rLength: {len(chunk[0].content)}", end='')
</syntaxhighlight>

=== With Reasoning Content ===
<syntaxhighlight lang="python">
# For models that support reasoning (e.g., QwQ)
llm = QwenChatAtDS({'model': 'qwq-32b-preview'})

messages = [Message(role='user', content='Solve: 25 * 48')]
response = llm._chat_no_stream(messages, {})

print(f"Reasoning: {response[0].reasoning_content}")
print(f"Answer: {response[0].content}")
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_LLM_Backend]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_API_Keys_Configuration]]
