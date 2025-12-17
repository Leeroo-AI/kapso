{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai/langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Agent Middleware]], [[domain::Context Management]], [[domain::Memory]], [[domain::Token Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==
SummarizationMiddleware automatically summarizes conversation history when token limits are approached, preserving recent messages while condensing older context to maintain continuity within model token constraints.

=== Description ===
This middleware monitors message token counts before each model invocation and triggers summarization when configurable thresholds are exceeded. It summarizes older messages while preserving recent conversation context, enabling agents to operate indefinitely within model token limits.

The middleware supports three types of thresholds:
* '''Fraction''': Percentage of model's maximum input tokens (e.g., 0.8 = 80%)
* '''Tokens''': Absolute token count (e.g., 3000 tokens)
* '''Messages''': Absolute message count (e.g., 50 messages)

Multiple trigger conditions can be specified (summarization occurs when any threshold is met).

Key features:
* Automatic trigger detection based on token/message counts
* Preserves recent context while summarizing older messages
* Respects AI/Tool message pair boundaries (never splits tool call responses)
* Configurable retention policies for preserved messages
* Customizable summarization prompts
* Binary search optimization for token-based cutoff determination
* Fallback handling for summarization failures

=== Usage ===
Use this middleware when:
* Running long-running agent conversations that exceed token limits
* Building multi-turn assistants with extended context needs
* Implementing agents that perform iterative tasks with many steps
* Managing memory-constrained deployments
* Preserving conversation continuity across extended sessions

== Code Reference ==
'''Source location:''' `/tmp/praxium_repo_wjjl6pl8/libs/langchain_v1/langchain/agents/middleware/summarization.py`

'''Signature:'''
<syntaxhighlight lang="python">
class SummarizationMiddleware(AgentMiddleware):
    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        trigger: ContextSize | list[ContextSize] | None = None,
        keep: ContextSize = ("messages", 20),
        token_counter: TokenCounter = count_tokens_approximately,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        trim_tokens_to_summarize: int | None = 4000,
    ) -> None
</syntaxhighlight>

'''Import statement:'''
<syntaxhighlight lang="python">
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware.summarization import (
    ContextSize,
    ContextFraction,
    ContextTokens,
    ContextMessages
)
</syntaxhighlight>

== I/O Contract ==

=== Initialization Parameters ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| model || str/BaseChatModel || (required) || Language model for generating summaries
|-
| trigger || ContextSize/list/None || None || Threshold(s) that trigger summarization. Can be single tuple or list of tuples
|-
| keep || ContextSize || ("messages", 20) || Context retention policy after summarization
|-
| token_counter || TokenCounter || count_tokens_approximately || Function to count tokens in messages
|-
| summary_prompt || str || DEFAULT_SUMMARY_PROMPT || Prompt template for generating summaries
|-
| trim_tokens_to_summarize || int/None || 4000 || Max tokens in messages sent for summarization (None = no trimming)
|}

=== ContextSize Types ===
{| class="wikitable"
! Type !! Format !! Example !! Description
|-
| ContextFraction || ("fraction", float) || ("fraction", 0.8) || Fraction of model's max input tokens (0 < value <= 1)
|-
| ContextTokens || ("tokens", int) || ("tokens", 3000) || Absolute token count (value > 0)
|-
| ContextMessages || ("messages", int) || ("messages", 50) || Absolute message count (value > 0)
|}

=== Hook Methods ===
{| class="wikitable"
! Method !! Execution Point !! Input !! Output
|-
| before_model() || Before model invocation || AgentState, Runtime || dict[str, Any] or None (updated state with summary replacing old messages)
|}

=== State Updates ===
When summarization occurs:
* Removes all old messages (via `RemoveMessage(id=REMOVE_ALL_MESSAGES)`)
* Adds HumanMessage with summary
* Preserves recent messages (according to `keep` policy)

== Usage Examples ==

=== Example 1: Basic Token-Based Summarization ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

# Summarize when conversation exceeds 3000 tokens
agent = create_agent(
    "openai:gpt-4o",
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",  # Use faster/cheaper model for summaries
            trigger=("tokens", 3000),
            keep=("messages", 20)
        )
    ],
)

# When messages exceed 3000 tokens:
# - Older messages summarized
# - Summary inserted as HumanMessage
# - Last 20 messages preserved
</syntaxhighlight>

=== Example 2: Fractional Trigger (Model Profile Required) ===
<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model

# Summarize at 80% of model's max input tokens
model = init_chat_model(
    "openai:gpt-4o",
    profile={"max_input_tokens": 128000}  # Profile required for fractional triggers
)

agent = create_agent(
    model,
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("fraction", 0.8),  # Trigger at 102,400 tokens
            keep=("fraction", 0.3)      # Keep last 30% (38,400 tokens)
        )
    ],
)
</syntaxhighlight>

=== Example 3: Multiple Trigger Conditions ===
<syntaxhighlight lang="python">
# Summarize when ANY threshold is met
agent = create_agent(
    "openai:gpt-4o",
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            trigger=[
                ("tokens", 5000),    # Trigger at 5000 tokens OR
                ("messages", 100),   # Trigger at 100 messages
            ],
            keep=("messages", 25)
        )
    ],
)

# Whichever limit is reached first triggers summarization
</syntaxhighlight>

=== Example 4: Message-Based Trigger and Retention ===
<syntaxhighlight lang="python">
# Simple message count approach (no token counting overhead)
agent = create_agent(
    "openai:gpt-4o",
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            trigger=("messages", 60),
            keep=("messages", 15)
        )
    ],
)

# Summarize oldest 45 messages when 60 messages reached
# Preserve most recent 15 messages
</syntaxhighlight>

=== Example 5: Custom Summarization Prompt ===
<syntaxhighlight lang="python">
CUSTOM_PROMPT = """You are a context extraction assistant. Review the conversation history below and extract ONLY the key facts, decisions, and unfinished tasks that are critical for continuing the conversation.

Focus on:
- Specific user requirements and constraints
- Decisions made and rationale
- Current progress and blockers
- Next steps or pending actions

Conversation history:
{messages}

Extracted context (be concise):"""

agent = create_agent(
    "openai:gpt-4o",
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20),
            summary_prompt=CUSTOM_PROMPT
        )
    ],
)
</syntaxhighlight>

=== Example 6: Custom Token Counter ===
<syntaxhighlight lang="python">
from tiktoken import encoding_for_model

# Use exact token counter (slower but accurate)
def count_tokens_exact(messages) -> int:
    encoding = encoding_for_model("gpt-4o")
    total = 0
    for msg in messages:
        total += len(encoding.encode(str(msg.content)))
    return total

agent = create_agent(
    "openai:gpt-4o",
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            trigger=("tokens", 4000),
            token_counter=count_tokens_exact
        )
    ],
)
</syntaxhighlight>

=== Example 7: Aggressive Summarization (Low Token Budget) ===
<syntaxhighlight lang="python">
# For severely token-constrained scenarios
agent = create_agent(
    "openai:gpt-4o",
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            trigger=("tokens", 2000),      # Trigger early
            keep=("messages", 10),         # Keep very few messages
            trim_tokens_to_summarize=2000  # Limit summarization input
        )
    ],
)

# Aggressive summarization to minimize token usage
</syntaxhighlight>

=== Example 8: No Automatic Trigger (Manual Control) ===
<syntaxhighlight lang="python">
# Disable automatic summarization (trigger=None)
# Could be triggered manually or via custom logic
agent = create_agent(
    "openai:gpt-4o",
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            trigger=None,  # No automatic triggers
            keep=("messages", 15)
        )
    ],
)

# Summarization only occurs if manually triggered via other mechanisms
</syntaxhighlight>

=== Example 9: Disable Trimming for Summarization ===
<syntaxhighlight lang="python">
# Send ALL messages to summarization model (no pre-trimming)
agent = create_agent(
    "openai:gpt-4o",
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            trigger=("tokens", 5000),
            keep=("messages", 20),
            trim_tokens_to_summarize=None  # No trimming
        )
    ],
)

# Useful when summarization model has large context window
# Or when you want comprehensive summaries
</syntaxhighlight>

== Implementation Details ==

=== Summarization Process ===
# Check if any trigger condition is met
# Determine cutoff index (where to split messages)
# Partition messages into: messages_to_summarize and preserved_messages
# Generate summary from messages_to_summarize
# Build new message list: [RemoveMessage(all), summary_as_HumanMessage, ...preserved_messages]
# Return updated state

=== Cutoff Determination ===

'''Message-based (`keep=("messages", N)`):'''
* Calculate cutoff as `len(messages) - N`
* Advance past any ToolMessages at cutoff point (preserve AI/Tool pairs)

'''Token-based (`keep=("tokens", N)` or `keep=("fraction", F)`):'''
* Use binary search to find earliest message index where suffix fits in budget
* Iterate at most `log2(message_count)` times for efficiency
* Advance past ToolMessages at found cutoff
* Fallback to message-based if token counting fails

=== AI/Tool Message Pair Preservation ===
Cutoff point always advanced past ToolMessages:
```python
while cutoff_index < len(messages) and isinstance(messages[cutoff_index], ToolMessage):
    cutoff_index += 1
```

This prevents splitting parallel tool call responses from their originating AIMessage.

=== Token Counting ===
Default token counter uses character-based approximation:
* General models: ~4 characters per token
* Anthropic models: ~3.3 characters per token (detected via `_llm_type`)

Custom counters can be provided for exact counting (e.g., tiktoken).

=== Summarization Model Invocation ===
Messages formatted into prompt:
```python
summary_prompt.format(messages=trimmed_messages)
```

If trimming enabled (`trim_tokens_to_summarize` not None):
* Use `trim_messages()` to reduce context sent to summarization model
* Strategy: Keep last messages, allow partial truncation
* Fallback: Last 15 messages if trimming fails

=== Error Handling ===
If summary generation fails:
* Catches all exceptions
* Returns error message as summary: `"Error generating summary: {exc}"`
* Agent continues with error summary (prevents pipeline breakage)

=== Message ID Management ===
All messages ensured to have IDs before processing:
```python
for msg in messages:
    if msg.id is None:
        msg.id = str(uuid.uuid4())
```

Required for `add_messages` reducer to properly handle `RemoveMessage`.

=== Performance Optimizations ===
* Binary search for token-based cutoff (O(log n) instead of O(n))
* Token counting cached (not recalculated during binary search)
* Async support (delegates to sync implementation currently)

=== Summary Format ===
Summary inserted as HumanMessage:
```
Here is a summary of the conversation to date:

[Generated summary content]
```

Model sees this as user input, establishing context for subsequent conversation.

=== Deprecated Parameters ===
Backwards compatibility maintained:
* `max_tokens_before_summary` -> Use `trigger=("tokens", value)`
* `messages_to_keep` -> Use `keep=("messages", value)`

Warnings issued for deprecated parameters.

== Related Pages ==
* [[langchain-ai_langchain_AgentMiddleware|AgentMiddleware]] - Base middleware class
* [[Token Management Strategies]] - Optimizing agent token usage
* [[Context Window Management]] - Techniques for long conversations
* [[Memory Systems]] - Complementary memory approaches
* [[langchain-ai_langchain_trim_messages|trim_messages]] - Message trimming utility
