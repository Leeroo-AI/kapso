# Implementation: TextChatAtOAI

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::Doc|OpenAI API|https://platform.openai.com/docs/api-reference]]
|-
! Domains
| [[domain::LLM_Backend]], [[domain::OpenAI]], [[domain::Function_Calling]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:00 GMT]]
|}

== Overview ==
OpenAI-compatible LLM backend for Qwen-Agent that supports streaming, function calling, and reasoning content extraction.

=== Description ===
The `TextChatAtOAI` class provides an OpenAI API-compatible LLM backend registered as `'oai'` type. It supports:

- Both OpenAI v0.x and v1.x API versions
- Streaming and non-streaming chat completions
- Function calling with tool schemas
- Reasoning content extraction (for models like o1/QwQ)
- Custom API base URLs for vLLM/local servers
- Extra parameters like `top_k` and `repetition_penalty`

The class extends `BaseFnCallModel` from Qwen-Agent for seamless integration.

=== Usage ===
Use as the LLM backend when agents need OpenAI-compatible API access, including local vLLM servers.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebDancer/demos/llm/oai.py WebAgent/WebDancer/demos/llm/oai.py]
* '''Lines:''' 1-218

=== Signature ===
<syntaxhighlight lang="python">
@register_llm('oai')
class TextChatAtOAI(BaseFnCallModel):
    """OpenAI-compatible LLM backend."""

    def __init__(self, cfg: Optional[Dict] = None):
        """
        Initialize OpenAI backend.

        Args:
            cfg: Configuration dict with keys:
                - model: Model name (default 'gpt-4o-mini')
                - api_key: OpenAI API key
                - api_base/base_url/model_server: API endpoint
                - generate_cfg: Generation parameters
        """
        ...

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict
    ) -> Iterator[List[Message]]:
        """Stream chat completion."""
        ...

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict
    ) -> List[Message]:
        """Non-streaming chat completion."""
        ...

    def _chat_with_functions(
        self,
        messages: List[Message],
        functions: List[Dict],
        stream: bool,
        delta_stream: bool,
        generate_cfg: dict,
        lang: Literal['en', 'zh']
    ) -> Union[List[Message], Iterator[List[Message]]]:
        """Chat with function calling support."""
        ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from WebAgent.WebDancer.demos.llm.oai import TextChatAtOAI
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || str || No || Model name (default gpt-4o-mini)
|-
| api_key || str || Yes || OpenAI API key
|-
| api_base || str || No || Custom API endpoint
|-
| generate_cfg || Dict || No || Generation parameters
|-
| messages || List[Message] || Yes || Conversation history
|-
| functions || List[Dict] || No || Tool schemas for function calling
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
| reasoning_content || str || Chain-of-thought (if supported)
|-
| function_call || Dict || Tool call if requested
|}

== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
from WebAgent.WebDancer.demos.llm.oai import TextChatAtOAI

# Initialize with OpenAI
llm = TextChatAtOAI({
    'model': 'gpt-4o',
    'api_key': 'sk-...',
    'generate_cfg': {
        'temperature': 0.6,
        'max_tokens': 4096
    }
})

# Use in agent
from qwen_agent.agents import Assistant
agent = Assistant(llm=llm, function_list=['search'])
</syntaxhighlight>

=== With vLLM Server ===
<syntaxhighlight lang="python">
llm = TextChatAtOAI({
    'model': 'Qwen/QwQ-32B',
    'model_server': 'http://localhost:8000/v1',
    'api_key': 'EMPTY',
    'generate_cfg': {
        'temperature': 0.6,
        'top_p': 0.95,
        'top_k': 40,
        'repetition_penalty': 1.1
    }
})
</syntaxhighlight>

=== Direct Chat Call ===
<syntaxhighlight lang="python">
from qwen_agent.llm.schema import Message

messages = [
    Message(role='user', content='What is 2+2?')
]

# Non-streaming
response = llm._chat_no_stream(messages, {})
print(response[0].content)

# Streaming
for chunk in llm._chat_stream(messages, delta_stream=True, generate_cfg={}):
    print(chunk[0].content, end='')
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_LLM_Backend]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_Python_Dependencies]]
