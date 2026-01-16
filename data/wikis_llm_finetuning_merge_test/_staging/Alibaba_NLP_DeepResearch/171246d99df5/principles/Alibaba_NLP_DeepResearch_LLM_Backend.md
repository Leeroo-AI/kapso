# Principle: LLM_Backend

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|OpenAI API|https://platform.openai.com/docs/api-reference]]
* [[source::Doc|DashScope API|https://help.aliyun.com/document_detail/2712195.html]]
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::LLM]], [[domain::API_Integration]], [[domain::Backend]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:30 GMT]]
|}

== Overview ==

Abstraction layer for LLM API integration supporting multiple providers (OpenAI, DashScope) with unified interface for chat completions and tool calling.

=== Description ===

LLM Backend provides a consistent interface for interacting with different LLM providers:

1. **Provider abstraction** - Same API regardless of backend (OpenAI, DashScope, vLLM)
2. **Configuration** - Model name, API keys, generation parameters
3. **Streaming** - Support for token-by-token response streaming
4. **Tool calling** - Function calling with tool definitions
5. **Error handling** - Retries, timeouts, rate limiting

The WebDancer demos implement OpenAI-style (TextChatAtOAI) and DashScope (QwenChatAtDS) backends.

=== Usage ===

Use LLM Backend when:
- Need to support multiple LLM providers
- Building provider-agnostic agent systems
- Want configurable model selection
- Need streaming chat completions

== Theoretical Basis ==

LLM backend abstraction:

'''LLM Backend Pattern:'''
<syntaxhighlight lang="python">
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """Abstract base class for LLM backends."""

    def __init__(self, config: dict):
        self.model = config['model']
        self.api_key = config.get('api_key')
        self.generate_cfg = config.get('generate_cfg', {})

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        stream: bool = False
    ) -> str:
        """Generate chat completion."""
        pass

class TextChatAtOAI(BaseLLM):
    """OpenAI-compatible LLM backend."""

    def __init__(self, config: dict):
        super().__init__(config)
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def chat(self, messages, tools=None, stream=False):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            stream=stream,
            **self.generate_cfg
        )
        if stream:
            async for chunk in response:
                yield chunk.choices[0].delta.content
        else:
            return response.choices[0].message.content

class QwenChatAtDS(BaseLLM):
    """Alibaba DashScope LLM backend."""

    async def chat(self, messages, tools=None, stream=False):
        # DashScope-specific implementation
        ...
</syntaxhighlight>

Backend features:
- **Async support**: All methods are async
- **Streaming**: Generator for token streaming
- **Tool calling**: Pass tool schemas, parse tool calls
- **Error handling**: Retry logic, rate limit handling

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_TextChatAtOAI]]
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_QwenChatAtDS]]

=== Related Principles ===
* [[related_to::Principle:Alibaba_NLP_DeepResearch_ReAct_Loop_Execution]]
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Search_Agent_Architecture]]
