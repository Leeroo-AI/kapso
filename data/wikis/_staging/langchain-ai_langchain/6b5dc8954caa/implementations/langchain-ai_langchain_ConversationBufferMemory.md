{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|Memory Migration|https://python.langchain.com/docs/versions/migrating_memory/]]
|-
! Domains
| [[domain::Memory]], [[domain::Conversational_AI]], [[domain::Deprecated]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Deprecated memory classes `ConversationBufferMemory` and `ConversationStringBufferMemory` that store full conversation history without summarization.

=== Description ===

The `ConversationBufferMemory` class stores the complete conversation history (all messages) and returns it as either a list of messages or a formatted string. It's the simplest memory implementation but can exceed context limits for long conversations. `ConversationStringBufferMemory` is a string-specific variant.

=== Usage ===

These classes are deprecated since version 0.3.1. For new code, use LangGraph persistence or implement custom message history with the modern LangChain patterns.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/memory/buffer.py libs/langchain/langchain_classic/memory/buffer.py]
* '''Lines:''' 1-173

=== Signature ===
<syntaxhighlight lang="python">
@deprecated(since="0.3.1", removal="1.0.0")
class ConversationBufferMemory(BaseChatMemory):
    """A basic memory that stores the entire conversation history.

    Attributes:
        human_prefix: Prefix for human messages (default: "Human").
        ai_prefix: Prefix for AI messages (default: "AI").
        memory_key: Key to store history under (default: "history").
        return_messages: If True, return message objects; else formatted string.
    """

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"

    @property
    def buffer(self) -> Any:
        """Return conversation as messages or string based on return_messages."""

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Return history buffer under memory_key."""


@deprecated(since="0.3.1", removal="1.0.0")
class ConversationStringBufferMemory(BaseMemory):
    """String-based conversation buffer memory.

    Like ConversationBufferMemory but specifically for string-based conversations.
    """

    buffer: str = ""
    memory_key: str = "history"
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Legacy (deprecated)
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.memory.buffer import ConversationStringBufferMemory
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| inputs || dict[str, Any] || No || Chain inputs (not used for loading)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| history || str or list[BaseMessage] || Full conversation history
|}

== Usage Examples ==

=== Basic Usage (Legacy) ===
<syntaxhighlight lang="python">
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
from langchain_openai import OpenAI

# Create memory
memory = ConversationBufferMemory()

# Use with ConversationChain
conversation = ConversationChain(
    llm=OpenAI(),
    memory=memory,
    verbose=True,
)

# Have a conversation
conversation.predict(input="Hi, I'm Bob")
conversation.predict(input="What's my name?")  # Memory remembers "Bob"

# Inspect memory
print(memory.buffer)
# Human: Hi, I'm Bob
# AI: Hello Bob! Nice to meet you.
# Human: What's my name?
# AI: Your name is Bob.
</syntaxhighlight>

=== Return Messages ===
<syntaxhighlight lang="python">
# Return as message objects instead of string
memory = ConversationBufferMemory(return_messages=True)

# Load variables returns list of messages
vars = memory.load_memory_variables({})
print(vars["history"])  # [HumanMessage(...), AIMessage(...), ...]
</syntaxhighlight>

=== Custom Key ===
<syntaxhighlight lang="python">
# Use different memory key
memory = ConversationBufferMemory(memory_key="chat_history")

# Now access via chat_history
vars = memory.load_memory_variables({})
print(vars["chat_history"])
</syntaxhighlight>

=== Modern Alternative ===
<syntaxhighlight lang="python">
# Use LangGraph persistence instead
from langgraph.checkpoint import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

# Or use simple message list management
messages = []
messages.append(HumanMessage(content="Hi, I'm Bob"))
messages.append(AIMessage(content="Hello Bob!"))

# Pass messages directly to chat model
from langchain_openai import ChatOpenAI
llm = ChatOpenAI()
response = llm.invoke(messages + [HumanMessage(content="What's my name?")])
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
* [[uses_heuristic::Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic]]

