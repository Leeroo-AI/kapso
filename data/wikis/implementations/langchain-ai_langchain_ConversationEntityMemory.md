{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|Memory Migration|https://python.langchain.com/docs/versions/migrating_memory/]]
|-
! Domains
| [[domain::Memory]], [[domain::Entity_Extraction]], [[domain::Deprecated]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Deprecated `ConversationEntityMemory` class that extracts named entities from conversations and maintains summaries of each entity.

=== Description ===

The `ConversationEntityMemory` uses an LLM to extract named entities (people, places, things) from recent conversation history and generates/updates summaries for each entity. It stores entity information in a pluggable store (in-memory, Redis, SQLite, or Upstash Redis), enabling persistent memory about key entities across conversations.

=== Usage ===

This class is deprecated since version 0.3.1. It was useful for building conversational agents that need to remember facts about specific entities mentioned in discussions.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/memory/entity.py libs/langchain/langchain_classic/memory/entity.py]
* '''Lines:''' 1-611

=== Signature ===
<syntaxhighlight lang="python">
@deprecated(since="0.3.1", removal="1.0.0")
class BaseEntityStore(BaseModel, ABC):
    """Abstract base for entity stores."""

    @abstractmethod
    def get(self, key: str, default: str | None = None) -> str | None: ...
    def set(self, key: str, value: str | None) -> None: ...
    def delete(self, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...
    def clear(self) -> None: ...


class InMemoryEntityStore(BaseEntityStore):
    """In-memory entity store using a dict."""


class RedisEntityStore(BaseEntityStore):
    """Redis-backed entity store with TTL."""


class SQLiteEntityStore(BaseEntityStore):
    """SQLite-backed entity store."""


@deprecated(since="0.3.1", removal="1.0.0")
class ConversationEntityMemory(BaseChatMemory):
    """Entity extractor & summarizer memory.

    Attributes:
        llm: Language model for entity extraction and summarization.
        entity_extraction_prompt: Prompt for extracting entities.
        entity_summarization_prompt: Prompt for summarizing entities.
        entity_cache: Recently detected entity names.
        k: Number of recent message pairs to consider.
        entity_store: Backend store for entity data.
    """

    llm: BaseLanguageModel
    entity_store: BaseEntityStore = Field(default_factory=InMemoryEntityStore)
    k: int = 3  # Message pairs to consider
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Legacy (deprecated)
from langchain_classic.memory import ConversationEntityMemory
from langchain_classic.memory.entity import (
    InMemoryEntityStore,
    RedisEntityStore,
    SQLiteEntityStore,
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| inputs || dict[str, Any] || Yes || Chain inputs including user query
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| history || str or list[BaseMessage] || Recent conversation history
|-
| entities || dict[str, str] || Entity names mapped to their summaries
|}

== Usage Examples ==

=== Basic Entity Memory (Legacy) ===
<syntaxhighlight lang="python">
from langchain_classic.memory import ConversationEntityMemory
from langchain_classic.chains import ConversationChain
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)
memory = ConversationEntityMemory(llm=llm)

conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

# Build up entity knowledge
conversation.predict(input="Deven is working on a LangChain tutorial")
conversation.predict(input="He's writing about memory systems")

# Entity memory tracks "Deven"
print(memory.entity_store.get("Deven"))
# "Deven is working on a LangChain tutorial about memory systems."
</syntaxhighlight>

=== With Redis Store ===
<syntaxhighlight lang="python">
from langchain_classic.memory.entity import RedisEntityStore

# Use Redis for persistent entity storage
entity_store = RedisEntityStore(
    url="redis://localhost:6379/0",
    session_id="user_123",
    ttl=60 * 60 * 24,  # 1 day TTL
)

memory = ConversationEntityMemory(
    llm=llm,
    entity_store=entity_store,
)
</syntaxhighlight>

=== With SQLite Store ===
<syntaxhighlight lang="python">
from langchain_classic.memory.entity import SQLiteEntityStore

# Use SQLite for file-based persistence
entity_store = SQLiteEntityStore(
    db_file="entities.db",
    session_id="user_123",
)

memory = ConversationEntityMemory(
    llm=llm,
    entity_store=entity_store,
)
</syntaxhighlight>

=== Accessing Entity Data ===
<syntaxhighlight lang="python">
# Load memory variables includes entities
vars = memory.load_memory_variables({"input": "Tell me about Deven"})
print(vars["entities"])
# {"Deven": "Deven is working on a LangChain tutorial about memory systems."}
print(vars["history"])
# Recent conversation history
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
* [[uses_heuristic::Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic]]

