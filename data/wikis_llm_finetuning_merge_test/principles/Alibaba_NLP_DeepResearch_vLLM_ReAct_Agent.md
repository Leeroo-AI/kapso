# Principle: vLLM_ReAct_Agent

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|vLLM Documentation|https://vllm.readthedocs.io/]]
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Agent_Systems]], [[domain::vLLM]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:30 GMT]]
|}

== Overview ==

ReAct agent implementation using vLLM as the inference backend for high-throughput, low-latency tool calling with OpenAI-compatible API.

=== Description ===

vLLM ReAct Agent adapts the standard ReAct pattern to use vLLM's high-performance inference:

1. **vLLM backend** - OpenAI-compatible API with batched inference
2. **Continuous batching** - Multiple requests processed efficiently
3. **PagedAttention** - Efficient KV cache management
4. **Tool calling** - Function calling via guided generation
5. **Streaming** - Token-by-token response streaming

The WebSailor implementation uses vLLM for faster inference.

=== Usage ===

Use vLLM ReAct Agent when:
- Need high-throughput agent inference
- Running batch evaluations
- Want lower latency than API calls
- Have GPU resources for local inference

== Theoretical Basis ==

vLLM integration pattern:

'''vLLM ReAct Agent Pattern:'''
<syntaxhighlight lang="python">
from openai import OpenAI

class MultiTurnReactAgent:
    """ReAct agent with vLLM backend."""

    def __init__(
        self,
        vllm_url: str,
        model_name: str,
        tools: list[dict]
    ):
        # vLLM exposes OpenAI-compatible API
        self.client = OpenAI(
            base_url=f"{vllm_url}/v1",
            api_key="dummy"  # vLLM doesn't require real key
        )
        self.model = model_name
        self.tools = tools

    async def run(self, messages: list[dict]) -> str:
        """Execute ReAct loop with vLLM."""
        while True:
            # Generate with function calling
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.6
            )

            message = response.choices[0].message

            # Check for tool calls
            if message.tool_calls:
                for tc in message.tool_calls:
                    result = self.execute_tool(tc)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result
                    })
            elif "<answer>" in message.content:
                return self.extract_answer(message.content)

            messages.append(message)
</syntaxhighlight>

vLLM advantages:
- **Throughput**: 10-100x higher than sequential API calls
- **Latency**: Sub-second generation for short responses
- **Cost**: No per-token API charges
- **Control**: Full model access, custom stopping criteria

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_WebSailor_ReActAgent]]

=== Related Principles ===
* [[related_to::Principle:Alibaba_NLP_DeepResearch_ReAct_Loop_Execution]]
* [[related_to::Principle:Alibaba_NLP_DeepResearch_LLM_Backend]]
