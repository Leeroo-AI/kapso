# Token Counting Strategy Heuristic

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain Summarization Middleware|https://github.com/langchain-ai/langchain/blob/main/libs/langchain_v1/langchain/agents/middleware/summarization.py]]
* [[source::Doc|Claude Token Counting|https://platform.claude.com/docs/en/build-with-claude/token-counting]]
|-
! Domains
| [[domain::LLMs]], [[domain::Optimization]], [[domain::Cost_Optimization]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==
Guidance for choosing between approximate and exact token counting methods, with model-specific tuning parameters for accurate context management.

=== Description ===
Token counting is critical for context window management, summarization triggers, and cost estimation. LangChain provides both approximate (fast) and model-specific (accurate) token counting methods. The approximate method uses a default of ~4 characters per token, but this varies significantly by model (e.g., Claude uses ~3.3 characters per token).

=== Usage ===
Apply this heuristic when:
- Implementing context window management
- Setting up `SummarizationMiddleware` triggers
- Optimizing token usage for cost reduction
- Deciding between speed and accuracy in token counting

== The Insight (Rule of Thumb) ==

* **Action:** Use approximate token counting for speed; use model-specific counting for accuracy
* **Values:**
  - Default approximation: ~4 characters per token (general models)
  - Anthropic models (Claude): ~3.3 characters per token
  - OpenAI models: Use `tiktoken` for exact counts
* **Trade-off:** Approximate counting is ~100x faster but may be off by 10-20%; exact counting adds latency per API call
* **Default trigger thresholds:**
  - `_DEFAULT_TRIM_TOKEN_LIMIT = 4000` tokens
  - `_DEFAULT_MESSAGES_TO_KEEP = 20` messages
  - `_DEFAULT_FALLBACK_MESSAGE_COUNT = 15` messages

=== When to Use Each Method ===

| Method | Speed | Accuracy | Use When |
|--------|-------|----------|----------|
| `count_tokens_approximately` | Fast | ~80-90% | Real-time applications, initial prototyping |
| `tiktoken.encode()` | Medium | ~99% | OpenAI models, cost-sensitive applications |
| Model API counting | Slow | 100% | Precise billing estimates, critical context limits |

== Reasoning ==

The 3.3 characters-per-token ratio for Anthropic models was derived from empirical testing against Claude's token counting API. This calibration ensures summarization triggers fire at appropriate times without prematurely truncating context.

Code Evidence from `summarization.py:122-128`:
<syntaxhighlight lang="python">
def _get_approximate_token_counter(model: BaseChatModel) -> TokenCounter:
    """Tune parameters of approximate token counter based on model type."""
    if model._llm_type == "anthropic-chat":
        # 3.3 was estimated in an offline experiment, comparing with Claude's token-counting
        # API: https://platform.claude.com/docs/en/build-with-claude/token-counting
        return partial(count_tokens_approximately, chars_per_token=3.3)
    return count_tokens_approximately
</syntaxhighlight>

Default constants from `summarization.py:56-58`:
<syntaxhighlight lang="python">
_DEFAULT_MESSAGES_TO_KEEP = 20
_DEFAULT_TRIM_TOKEN_LIMIT = 4000
_DEFAULT_FALLBACK_MESSAGE_COUNT = 15
</syntaxhighlight>

Token-based trigger configuration from `summarization.py:143-147`:
<syntaxhighlight lang="python">
trigger: ContextSize | list[ContextSize] | None = None,
keep: ContextSize = ("messages", _DEFAULT_MESSAGES_TO_KEEP),
token_counter: TokenCounter = count_tokens_approximately,
summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
trim_tokens_to_summarize: int | None = _DEFAULT_TRIM_TOKEN_LIMIT,
</syntaxhighlight>

== Related Pages ==
* [[uses_heuristic::Implementation:langchain-ai_langchain_TextSplitter_length_functions]]
* [[uses_heuristic::Workflow:langchain-ai_langchain_Agent_Creation_Workflow]]
* [[uses_heuristic::Principle:langchain-ai_langchain_Length_Function_Setup]]
