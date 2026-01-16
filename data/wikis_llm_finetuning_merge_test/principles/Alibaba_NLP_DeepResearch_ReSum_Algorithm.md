# Principle: ReSum_Algorithm

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|WebResummer Paper|https://arxiv.org/abs/2502.11543]]
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Summarization]], [[domain::Context_Compression]], [[domain::Agent_Memory]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:30 GMT]]
|}

== Overview ==

ReSum (Recursive Summarization) algorithm that compresses agent conversation history using a dedicated summarization model to enable longer reasoning trajectories.

=== Description ===

ReSum addresses the context length limitation in long-running agents by periodically summarizing conversation history:

1. **Threshold monitoring** - Check token count after each turn
2. **History segmentation** - Identify recent history to summarize
3. **Summarization call** - Use dedicated ReSum-Tool model
4. **Context replacement** - Replace verbose history with summary
5. **Incremental updates** - Build on previous summaries

Unlike simple truncation, ReSum preserves key information through intelligent compression.

=== Usage ===

Use ReSum Algorithm when:
- Agent trajectories exceed context limits
- Need to preserve reasoning history
- Running multi-hour agent sessions
- Building memory-efficient agents

The WebResummer agent integrates ReSum as a core capability.

== Theoretical Basis ==

ReSum compression preserves information:

<math>
S_t = \text{Summarize}(H_{t-k:t}, S_{t-1})
</math>

Where S_t is summary at time t, H is history, and k is window size.

'''ReSum Pattern:'''
<syntaxhighlight lang="python">
def summarize_conversation(
    question: str,
    recent_history: list[dict],
    last_summary: str | None,
    max_retries: int = 10
) -> str:
    """
    Apply ReSum to compress conversation history.

    Args:
        question: Original user question
        recent_history: Recent conversation turns
        last_summary: Previous summary (for incremental)
        max_retries: API retry attempts

    Returns:
        Compressed summary wrapped in <summary> tags
    """
    # Build summarization prompt
    if last_summary:
        prompt = f"""Previous summary: {last_summary}

New conversation to summarize:
{format_messages(recent_history)}

Question being answered: {question}

Provide an updated summary preserving key information."""
    else:
        prompt = f"""Summarize this conversation:
{format_messages(recent_history)}

Question being answered: {question}

Extract key findings and reasoning."""

    # Call ReSum model
    response = call_resum_server(prompt, max_retries)
    return f"<summary>{response}</summary>"

class MultiTurnReactAgent:
    def run(self, messages):
        while not done:
            # Check context length
            if count_tokens(messages) > threshold:
                # Apply ReSum
                summary = summarize_conversation(
                    self.question,
                    messages[-window:],
                    self.last_summary
                )
                # Replace history with summary
                messages = [self.system_prompt, summary] + messages[-recent:]
                self.last_summary = summary

            # Continue reasoning
            ...
</syntaxhighlight>

Key properties:
- **Lossless key facts**: Important information preserved
- **Incremental**: Builds on previous summaries
- **Adaptive**: Triggered by context pressure

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_WebResummer_ReActAgent]]

=== Related Principles ===
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Context_Management]]
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Conversation_Summarization]]
