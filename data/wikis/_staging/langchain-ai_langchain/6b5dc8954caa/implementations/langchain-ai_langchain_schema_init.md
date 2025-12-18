{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Schemas]], [[domain::Compatibility]], [[domain::Core_Types]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Compatibility module that re-exports core LangChain types and schemas from `langchain_core` for backward compatibility.

=== Description ===

The `langchain_classic.schema` module provides backward-compatible imports for core LangChain types that have moved to `langchain_core`. It re-exports messages (AIMessage, HumanMessage, etc.), documents, outputs, prompts, agents, and other foundational types. The `RUN_KEY` constant is also defined here for run metadata.

=== Usage ===

For new code, import directly from `langchain_core`. This module exists to maintain compatibility with code that imports from `langchain.schema`.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/schema/__init__.py libs/langchain/langchain_classic/schema/__init__.py]
* '''Lines:''' 1-83

=== Signature ===
<syntaxhighlight lang="python">
# Re-exports from langchain_core
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.caches import BaseCache
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.exceptions import LangChainException, OutputParserException
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
    messages_from_dict,
    messages_to_dict,
)
from langchain_core.output_parsers import (
    BaseLLMOutputParser,
    BaseOutputParser,
    StrOutputParser,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatResult,
    Generation,
    LLMResult,
    RunInfo,
)
from langchain_core.prompts import BasePromptTemplate, format_document
from langchain_core.retrievers import BaseRetriever
from langchain_core.stores import BaseStore

from langchain_classic.base_memory import BaseMemory

RUN_KEY = "__run"
Memory = BaseMemory  # Backwards compatibility alias
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Legacy (use langchain_core instead)
from langchain_classic.schema import Document, AIMessage, HumanMessage

# Modern (preferred)
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
</syntaxhighlight>

== I/O Contract ==

This is a re-export module with no direct I/O contract.

== Usage Examples ==

=== Legacy Imports (Compatibility) ===
<syntaxhighlight lang="python">
# Old import style (still works)
from langchain_classic.schema import (
    Document,
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMemory,
    Generation,
    LLMResult,
)

# Use the types
doc = Document(page_content="Hello world")
msg = HumanMessage(content="Hi!")
</syntaxhighlight>

=== Using RUN_KEY ===
<syntaxhighlight lang="python">
from langchain_classic.schema import RUN_KEY

# RUN_KEY is used to store run metadata in chain outputs
result = chain.invoke(inputs, include_run_info=True)
run_info = result.get(RUN_KEY)
print(run_info.run_id)  # UUID of the run
</syntaxhighlight>

=== Modern Imports (Preferred) ===
<syntaxhighlight lang="python">
# Import from langchain_core directly
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
from langchain_core.outputs import Generation, LLMResult

# Use the same types
doc = Document(page_content="Hello world")
msg = HumanMessage(content="Hi!")
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]

