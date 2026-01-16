# Implementation: Summary_Utils

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Summarization]], [[domain::Context_Compression]], [[domain::ReSum]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:00 GMT]]
|}

== Overview ==
Conversation summarization utilities for ReSum that compress agent trajectories using a dedicated summarization model.

=== Description ===
The `summary_utils.py` module provides the core summarization functions for WebResummer's ReSum capability:

- `summarize_conversation`: Main function that takes conversation history and produces compressed summary
- `call_resum_server`: Calls the ReSum-Tool model server for summarization
- Support for incremental summarization with `last_summary` parameter
- Prompt templates for initial and continuation summarization

The summarization preserves key information while dramatically reducing token count.

=== Usage ===
Called by MultiTurnReactAgent when context exceeds threshold. Can also be used standalone for conversation compression.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebResummer/src/summary_utils.py WebAgent/WebResummer/src/summary_utils.py]
* '''Lines:''' 1-66

=== Signature ===
<syntaxhighlight lang="python">
def summarize_conversation(
    question: str,
    recent_history_messages: List[Dict],
    last_summary: Optional[str],
    max_retries: int = 10
) -> str:
    """
    Summarize conversation history.

    Args:
        question: Original user question
        recent_history_messages: Recent conversation turns
        last_summary: Previous summary (for incremental mode)
        max_retries: Max API retry attempts

    Returns:
        Summary string wrapped in <summary> tags
    """
    ...

def call_resum_server(
    query: str,
    max_retries: int = 10
) -> str:
    """
    Call ReSum-Tool model server.

    Args:
        query: Formatted summarization prompt
        max_retries: Max retry attempts

    Returns:
        Summary response with <summary> tags
    """
    ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from WebAgent.WebResummer.src.summary_utils import (
    summarize_conversation,
    call_resum_server
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| question || str || Yes || Original question for context
|-
| recent_history_messages || List[Dict] || Yes || Messages to summarize
|-
| last_summary || str || No || Previous summary for incremental
|-
| max_retries || int || No || API retries (default 10)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| summary || str || Compressed summary with tags
|}

== Usage Examples ==

=== Basic Summarization ===
<syntaxhighlight lang="python">
import os
from WebAgent.WebResummer.src.summary_utils import summarize_conversation

os.environ['RESUM_TOOL_NAME'] = 'resum-tool-30b'
os.environ['RESUM_TOOL_URL'] = 'http://localhost:8001/v1/chat/completions'

# Conversation history
messages = [
    {"role": "assistant", "content": "<think>Let me search...</think><tool_call>..."},
    {"role": "user", "content": "<tool_response>Search results...</tool_response>"},
    {"role": "assistant", "content": "<think>Found info about deadline...</think>"},
]

# First summarization
summary = summarize_conversation(
    question="When is ACL 2025 deadline?",
    recent_history_messages=messages,
    last_summary=None
)
print(summary)
# <summary>Searched for ACL 2025 deadline, found February 15...</summary>
</syntaxhighlight>

=== Incremental Summarization ===
<syntaxhighlight lang="python">
# Subsequent summarization with previous context
new_messages = [
    {"role": "assistant", "content": "<think>Verifying...</think>"},
    {"role": "user", "content": "<tool_response>Confirmed deadline...</tool_response>"},
]

updated_summary = summarize_conversation(
    question="When is ACL 2025 deadline?",
    recent_history_messages=new_messages,
    last_summary=summary  # Pass previous summary
)
</syntaxhighlight>

=== Direct Server Call ===
<syntaxhighlight lang="python">
from WebAgent.WebResummer.src.summary_utils import call_resum_server

# Direct query to ReSum server
query = """Summarize: User asked about deadlines.
Agent searched and found February 15, 2025."""

response = call_resum_server(query)
print(response)
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_Context_Management]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_API_Keys_Configuration]]
