{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|Migration Guide|https://python.langchain.com/docs/versions/migrating_chains/]]
|-
! Domains
| [[domain::Deprecation]], [[domain::Migration]], [[domain::Chains]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Warning: The `langchain_classic` package contains deprecated classes and functions that are scheduled for removal in LangChain 1.0.

=== Description ===

The `langchain_classic` module (formerly the main `langchain` package) contains legacy Chain-based abstractions that have been superseded by LangChain Expression Language (LCEL). These classes include `LLMChain`, `SequentialChain`, `MapReduceChain`, memory classes like `ConversationBufferMemory`, and document processing chains. While they remain functional for backwards compatibility, they lack support for modern features like streaming, async, and LangGraph integration.

=== Usage ===

Apply this warning when encountering code that imports from `langchain_classic` or uses deprecated Chain classes. Migration paths exist for all deprecated classes, primarily through LCEL's composable `RunnableSequence` pattern (`prompt | llm | parser`) and LangGraph for complex orchestration.

== The Insight (Rule of Thumb) ==

* **Action:** Migrate legacy Chain code to LCEL or LangGraph patterns
* **Timeline:** All `langchain_classic` classes are deprecated since various versions (0.1.x-0.2.x) and will be removed in 1.0
* **Trade-off:** LCEL provides better streaming, async support, and composability at the cost of different API patterns

=== Common Migrations ===

| Legacy Class | Modern Replacement |
|--------------|-------------------|
| `LLMChain` | `prompt \| llm \| parser` (LCEL) |
| `SequentialChain` | `chain1 \| chain2` (LCEL pipe) |
| `MapReduceChain` | LangGraph map-reduce pattern |
| `ConversationBufferMemory` | LangGraph state persistence |
| `load_summarize_chain` | Custom LCEL or LangGraph workflow |

== Reasoning ==

The Chain abstraction was LangChain's original composition primitive but had limitations:
1. **No streaming by default** - Chains buffered entire outputs
2. **Sync-first design** - Async was bolted on, not native
3. **Rigid interfaces** - Hard to customize intermediate steps
4. **Memory coupling** - Memory was tightly bound to chains

LCEL addresses these by making every component a `Runnable` with native streaming, batching, async, and retry support. LangGraph provides stateful orchestration for complex agent workflows.

== Related Pages ==

* [[uses_heuristic::Implementation:langchain-ai_langchain_LLMChain]]
* [[uses_heuristic::Implementation:langchain-ai_langchain_SequentialChain]]
* [[uses_heuristic::Implementation:langchain-ai_langchain_MapReduceChain]]
* [[uses_heuristic::Implementation:langchain-ai_langchain_BaseMemory]]
* [[uses_heuristic::Implementation:langchain-ai_langchain_ConversationBufferMemory]]
* [[uses_heuristic::Implementation:langchain-ai_langchain_chain_loading]]
* [[uses_heuristic::Implementation:langchain-ai_langchain_langchain_classic_init]]

