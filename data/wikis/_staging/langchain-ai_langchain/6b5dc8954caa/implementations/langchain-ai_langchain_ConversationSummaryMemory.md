{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|Summary Memory Guide|https://langchain-ai.github.io/langgraph/how-tos/memory/add-summary-conversation-history/]]
|-
! Domains
| [[domain::Memory]], [[domain::Summarization]], [[domain::Deprecated]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Deprecated `ConversationSummaryMemory` class that maintains a running summary of the conversation instead of storing all messages.

=== Description ===

The `ConversationSummaryMemory` continuously summarizes the conversation history using an LLM. After each turn, it updates the summary by incorporating the latest messages. This allows for much longer conversations than buffer memory since only the summary is kept, not all individual messages.

=== Usage ===

This class is deprecated since version 0.3.1. It was useful for conversations that exceeded context limits. For new code, see the LangGraph guide on adding conversation summaries.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/memory/summary.py libs/langchain/langchain_classic/memory/summary.py]
* '''Lines:''' 1-168

=== Signature ===
<syntaxhighlight lang="python">
@deprecated(since="0.2.12", removal="1.0")
class SummarizerMixin(BaseModel):
    """Mixin providing summarization capability.

    Attributes:
        llm: Language model for summarization.
        prompt: Prompt template for summarization.
        summary_message_cls: Message class for summary (default: SystemMessage).
    """

    llm: BaseLanguageModel
    prompt: BasePromptTemplate = SUMMARY_PROMPT

    def predict_new_summary(
        self,
        messages: list[BaseMessage],
        existing_summary: str,
    ) -> str:
        """Generate new summary from messages and existing summary."""


@deprecated(since="0.3.1", removal="1.0.0")
class ConversationSummaryMemory(BaseChatMemory, SummarizerMixin):
    """Continually summarizes the conversation history.

    Attributes:
        buffer: Current summary string.
        memory_key: Key to store summary under (default: "history").
    """

    buffer: str = ""
    memory_key: str = "history"

    @classmethod
    def from_messages(
        cls,
        llm: BaseLanguageModel,
        chat_memory: BaseChatMessageHistory,
        summarize_step: int = 2,
    ) -> ConversationSummaryMemory:
        """Create from existing messages, summarizing in batches."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Legacy (deprecated)
from langchain_classic.memory import ConversationSummaryMemory
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
| history || str or SystemMessage || Running conversation summary
|}

== Usage Examples ==

=== Basic Summary Memory (Legacy) ===
<syntaxhighlight lang="python">
from langchain_classic.memory import ConversationSummaryMemory
from langchain_classic.chains import ConversationChain
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)
memory = ConversationSummaryMemory(llm=llm)

conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

# Each turn updates the summary
conversation.predict(input="Hi, I'm working on a Python project")
conversation.predict(input="It's a web scraper for news sites")
conversation.predict(input="I'm having trouble with async requests")

# Check the summary
print(memory.buffer)
# "The human is working on a Python web scraper project for news sites
#  and is experiencing difficulties with async requests."
</syntaxhighlight>

=== From Existing Messages ===
<syntaxhighlight lang="python">
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# Start with existing history
chat_memory = InMemoryChatMessageHistory()
chat_memory.add_user_message("What is LangChain?")
chat_memory.add_ai_message("LangChain is a framework for building LLM apps.")
chat_memory.add_user_message("What can I build with it?")
chat_memory.add_ai_message("You can build chatbots, agents, RAG systems, and more.")

# Create summary memory from existing messages
memory = ConversationSummaryMemory.from_messages(
    llm=llm,
    chat_memory=chat_memory,
    summarize_step=2,  # Summarize 2 messages at a time
)
print(memory.buffer)  # Summary of the conversation
</syntaxhighlight>

=== Return as Message ===
<syntaxhighlight lang="python">
# Return summary as a message object
memory = ConversationSummaryMemory(llm=llm, return_messages=True)

vars = memory.load_memory_variables({})
print(vars["history"])  # [SystemMessage(content="Summary...")]
</syntaxhighlight>

=== Modern LangGraph Alternative ===
<syntaxhighlight lang="python">
# See LangGraph guide for modern summary implementation
# https://langchain-ai.github.io/langgraph/how-tos/memory/add-summary-conversation-history/

from langgraph.graph import StateGraph

# Define state with summary field
class State(TypedDict):
    messages: list[BaseMessage]
    summary: str

# Create graph with summarization logic
# See LangGraph docs for complete implementation
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
* [[uses_heuristic::Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic]]

