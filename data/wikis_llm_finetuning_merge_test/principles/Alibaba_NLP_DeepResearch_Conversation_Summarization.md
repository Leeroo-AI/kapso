# Principle: Conversation_Summarization

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|WebResummer Paper|https://arxiv.org/abs/2502.11543]]
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Summarization]], [[domain::NLP]], [[domain::Memory]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:30 GMT]]
|}

== Overview ==

Utility functions for compressing agent conversation history using a dedicated summarization model (ReSum-Tool) to maintain context within token limits.

=== Description ===

Conversation Summarization provides the core compression utilities for ReSum:

1. **Message formatting** - Convert chat history to summarization input
2. **Server communication** - Call ReSum-Tool model server
3. **Response parsing** - Extract summary from model output
4. **Error handling** - Retry logic for API failures
5. **Tag wrapping** - Format output with `<summary>` tags

The `summary_utils.py` module implements these utilities.

=== Usage ===

Use Conversation Summarization when:
- Implementing context compression
- Building agent memory systems
- Need dedicated summarization model
- Creating incremental summaries

== Theoretical Basis ==

Summarization utility pattern:

'''Conversation Summarization Pattern:'''
<syntaxhighlight lang="python">
import os
import requests

RESUM_TOOL_URL = os.environ.get('RESUM_TOOL_URL')
RESUM_TOOL_NAME = os.environ.get('RESUM_TOOL_NAME')

def format_messages(messages: list[dict]) -> str:
    """Format messages for summarization prompt."""
    formatted = []
    for msg in messages:
        role = msg['role']
        content = msg['content']
        formatted.append(f"{role.upper()}: {content}")
    return "\n\n".join(formatted)

def call_resum_server(query: str, max_retries: int = 10) -> str:
    """
    Call ReSum-Tool model server.

    Args:
        query: Summarization prompt
        max_retries: Max retry attempts

    Returns:
        Summary response
    """
    for attempt in range(max_retries):
        try:
            response = requests.post(
                RESUM_TOOL_URL,
                json={
                    "model": RESUM_TOOL_NAME,
                    "messages": [{"role": "user", "content": query}]
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff

def summarize_conversation(
    question: str,
    recent_history: list[dict],
    last_summary: str | None,
    max_retries: int = 10
) -> str:
    """Main summarization function."""
    # Build prompt with question context
    history_text = format_messages(recent_history)

    if last_summary:
        prompt = f"Previous: {last_summary}\n\nNew: {history_text}"
    else:
        prompt = history_text

    response = call_resum_server(prompt, max_retries)
    return f"<summary>{response}</summary>"
</syntaxhighlight>

Design considerations:
- **Dedicated model**: Use specialized summarization model
- **Incremental**: Support building on previous summaries
- **Resilient**: Retry on transient failures
- **Tagged output**: Machine-readable summary format

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_Summary_Utils]]

=== Related Principles ===
* [[related_to::Principle:Alibaba_NLP_DeepResearch_ReSum_Algorithm]]
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Context_Management]]
