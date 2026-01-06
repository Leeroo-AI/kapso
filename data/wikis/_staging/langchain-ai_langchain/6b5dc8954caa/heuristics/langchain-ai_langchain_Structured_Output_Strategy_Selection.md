# Structured Output Strategy Selection Heuristic

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain Structured Output|https://github.com/langchain-ai/langchain/blob/main/libs/langchain_v1/langchain/agents/structured_output.py]]
|-
! Domains
| [[domain::LLMs]], [[domain::Agents]], [[domain::Data_Extraction]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==
Decision framework for choosing between `ToolStrategy`, `ProviderStrategy`, and `AutoStrategy` when configuring structured output extraction from language models.

=== Description ===
LangChain provides three strategies for extracting structured data from LLM responses. Each strategy has different tradeoffs in terms of reliability, provider compatibility, and flexibility. Choosing the right strategy depends on your model provider, schema complexity, and error handling requirements.

=== Usage ===
Apply this heuristic when:
- Configuring `response_format` in `create_agent()`
- Deciding how to extract typed data from LLM responses
- Debugging structured output failures
- Optimizing for reliability vs. simplicity

== The Insight (Rule of Thumb) ==

* **Action:** Choose strategy based on provider support and reliability requirements
* **Values:**

| Strategy | Use When | Pros | Cons |
|----------|----------|------|------|
| `ToolStrategy` | Default choice; works with any model supporting tool calling | Universal compatibility; built-in retry via `handle_errors` | Requires tool calling support |
| `ProviderStrategy` | OpenAI models with JSON mode; high reliability needed | Native JSON schema validation; `strict=True` for guaranteed compliance | OpenAI-only; limited provider support |
| `AutoStrategy` | Simple cases; let LangChain decide | Zero configuration | Less predictable behavior |

* **Trade-off:** `ToolStrategy` is more universal but adds parsing overhead; `ProviderStrategy` is more reliable but OpenAI-specific
* **Default error handling:** `handle_errors=True` in `ToolStrategy` enables automatic retry on validation failures

=== Strategy Decision Tree ===

```
┌─────────────────────────────────────┐
│ Need structured output?             │
└─────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│ Using OpenAI with strict JSON mode? │
├──────────── Yes ────────────────────┤
│      Use ProviderStrategy           │
│      (strict=True for reliability)  │
└─────────────────────────────────────┘
                  │ No
                  ▼
┌─────────────────────────────────────┐
│ Model supports tool calling?        │
├──────────── Yes ────────────────────┤
│      Use ToolStrategy               │
│      (handle_errors=True)           │
└─────────────────────────────────────┘
                  │ No
                  ▼
┌─────────────────────────────────────┐
│ Simple schema, one-shot extraction? │
├──────────── Yes ────────────────────┤
│      Use AutoStrategy               │
└─────────────────────────────────────┘
```

== Reasoning ==

The three strategies address different model capabilities and reliability requirements:

1. **ToolStrategy** wraps the schema as a "tool" that the model must call, extracting arguments as structured data. This works with any model supporting function/tool calling.

2. **ProviderStrategy** uses native provider features (like OpenAI's JSON mode with `response_format={"type": "json_schema"}`), which provides stronger guarantees but limited provider support.

3. **AutoStrategy** automatically selects between Tool and Provider strategies based on detected model capabilities.

Code Evidence from `structured_output.py:181-243` (ToolStrategy):
<syntaxhighlight lang="python">
class ToolStrategy(ResponseFormat[SchemaT]):
    """Use tool calling to extract structured output.

    Works with any chat model that supports tool binding. The schema is converted
    to a tool definition, and the model's tool call arguments are parsed as the
    structured output.
    """

    def __init__(
        self,
        schema: type[SchemaT],
        *,
        tool_message_content: str | None = None,
        handle_errors: bool | str | type[Exception] | Callable[..., str] = True,
    ) -> None:
        # handle_errors enables retry on validation failures
</syntaxhighlight>

Code Evidence from `structured_output.py:246-286` (ProviderStrategy):
<syntaxhighlight lang="python">
class ProviderStrategy(ResponseFormat[SchemaT]):
    """Use provider-native structured output (e.g., OpenAI JSON mode).

    Requires provider support for response_format with json_schema.
    More reliable than ToolStrategy when available.
    """

    def __init__(
        self,
        schema: type[SchemaT],
        *,
        strict: bool | None = None,
    ) -> None:
        # strict=True for OpenAI's strict JSON mode
</syntaxhighlight>

== Related Pages ==
* [[uses_heuristic::Implementation:langchain-ai_langchain_ResponseFormat_strategies]]
* [[uses_heuristic::Implementation:langchain-ai_langchain_ResponseFormat_type_union]]
* [[uses_heuristic::Workflow:langchain-ai_langchain_Structured_Output_Workflow]]
* [[uses_heuristic::Principle:langchain-ai_langchain_Strategy_Selection]]
* [[uses_heuristic::Principle:langchain-ai_langchain_Structured_Output_Strategy]]
