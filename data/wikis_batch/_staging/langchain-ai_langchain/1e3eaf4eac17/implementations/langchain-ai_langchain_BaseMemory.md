---
title: BaseMemory
type: implementation
project: langchain-ai/langchain
file: libs/langchain/langchain_classic/base_memory.py
deprecated: true
deprecation_version: 0.3.3
removal_version: 1.0.0
migration_guide: https://python.langchain.com/docs/versions/migrating_memory/
---

= BaseMemory Implementation =

== Overview ==

'''BaseMemory''' is an abstract base class that defines the interface for memory components in LangChain chains. Memory refers to state management across chain executions, enabling chains to store information about past runs and inject that context into future executions. This is particularly useful for conversational applications where context from previous interactions needs to be maintained.

'''Status:''' This class is deprecated as of version 0.3.3 and will be removed in version 1.0.0. Users should migrate to newer memory patterns as described in the migration guide.

== Code Reference ==

=== Source Location ===
<syntaxhighlight lang="text">
File: /tmp/praxium_repo_wjjl6pl8/libs/langchain/langchain_classic/base_memory.py
Lines: 1-117
Package: langchain-classic
</syntaxhighlight>

=== Class Signature ===
<syntaxhighlight lang="python">
@deprecated(
    since="0.3.3",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class BaseMemory(Serializable, ABC):
    """Abstract base class for memory in Chains."""
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from langchain_classic.base_memory import BaseMemory
</syntaxhighlight>

== Abstract Methods ==

=== memory_variables ===

'''Property (abstract):''' Returns the string keys that this memory class will add to chain inputs.

{| class="wikitable"
|+ Return Value
! Type !! Description
|-
| <code>list[str]</code> || List of memory variable names that will be injected into chain inputs
|}

=== load_memory_variables ===

'''Method Signature:'''
<syntaxhighlight lang="python">
@abstractmethod
def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
    """Return key-value pairs given the text input to the chain."""
</syntaxhighlight>

{| class="wikitable"
|+ Parameters
! Parameter !! Type !! Description
|-
| inputs || <code>dict[str, Any]</code> || The inputs to the chain
|}

{| class="wikitable"
|+ Return Value
! Type !! Description
|-
| <code>dict[str, Any]</code> || Dictionary of memory variables to inject into the chain
|}

'''Async variant:''' <code>aload_memory_variables(inputs)</code> - Automatically wraps the sync method using <code>run_in_executor</code>.

=== save_context ===

'''Method Signature:'''
<syntaxhighlight lang="python">
@abstractmethod
def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
    """Save the context of this chain run to memory."""
</syntaxhighlight>

{| class="wikitable"
|+ Parameters
! Parameter !! Type !! Description
|-
| inputs || <code>dict[str, Any]</code> || The inputs to the chain
|-
| outputs || <code>dict[str, str]</code> || The outputs of the chain
|}

'''Async variant:''' <code>asave_context(inputs, outputs)</code> - Automatically wraps the sync method using <code>run_in_executor</code>.

=== clear ===

'''Method Signature:'''
<syntaxhighlight lang="python">
@abstractmethod
def clear(self) -> None:
    """Clear memory contents."""
</syntaxhighlight>

'''Async variant:''' <code>aclear()</code> - Automatically wraps the sync method using <code>run_in_executor</code>.

== Usage Example ==

<syntaxhighlight lang="python">
from langchain_classic.base_memory import BaseMemory
from typing import Any

class SimpleMemory(BaseMemory):
    """Example memory implementation that stores key-value pairs."""

    memories: dict[str, Any] = dict()

    @property
    def memory_variables(self) -> list[str]:
        """Return all stored memory keys."""
        return list(self.memories.keys())

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Return all stored memories."""
        return self.memories

    def save_context(
        self, inputs: dict[str, Any], outputs: dict[str, str]
    ) -> None:
        """Save context - could store inputs/outputs as needed."""
        # Implementation would store relevant information
        pass

    def clear(self) -> None:
        """Clear all stored memories."""
        self.memories.clear()

# Usage in a chain
memory = SimpleMemory()
memory.memories = {"context": "Previous conversation history"}

# Load memory variables to inject into chain
context = memory.load_memory_variables({"input": "Hello"})
# Returns: {"context": "Previous conversation history"}

# Save new context after chain execution
memory.save_context(
    inputs={"input": "Hello"},
    outputs={"output": "Hi there!"}
)

# Clear memory when conversation ends
memory.clear()
</syntaxhighlight>

== Implementation Contract ==

{| class="wikitable"
|+ Required Implementations
! Method !! Return Type !! Purpose
|-
| <code>memory_variables</code> || <code>list[str]</code> || Declare which keys will be added to chain inputs
|-
| <code>load_memory_variables</code> || <code>dict[str, Any]</code> || Retrieve memory content to inject into chain
|-
| <code>save_context</code> || <code>None</code> || Store information from chain execution
|-
| <code>clear</code> || <code>None</code> || Reset memory to empty state
|}

== Design Patterns ==

=== Memory Lifecycle ===

1. '''Initialization:''' Memory object is created and optionally populated with initial state
2. '''Load:''' Before chain execution, <code>load_memory_variables()</code> retrieves context
3. '''Injection:''' Retrieved variables are merged into chain inputs
4. '''Execution:''' Chain runs with augmented inputs containing memory context
5. '''Save:''' After execution, <code>save_context()</code> stores new information
6. '''Clear:''' When context should be reset (e.g., new conversation)

=== Common Use Cases ===

* '''Conversational Memory:''' Store message history for chatbots
* '''State Tracking:''' Maintain variables across multi-step workflows
* '''Context Windows:''' Keep recent context while dropping old information
* '''Entity Memory:''' Track entities and facts mentioned in conversation

== Configuration ==

<syntaxhighlight lang="python">
model_config = ConfigDict(
    arbitrary_types_allowed=True,
)
</syntaxhighlight>

The base class allows arbitrary types in Pydantic models, enabling flexible memory implementations.

== Related Components ==

* '''Serializable''' (langchain-core) - Parent class providing serialization capabilities
* '''Chain''' (langchain-classic) - Chains that integrate memory
* '''ConversationBufferMemory''' - Concrete implementation storing full history
* '''ConversationSummaryMemory''' - Implementation that summarizes history
* '''VectorStoreRetrieverMemory''' - Implementation using vector search for retrieval

== Migration Notes ==

This class is deprecated. For new implementations:

1. Use LangGraph for state management in multi-turn applications
2. Use RunnableWithMessageHistory for simple conversational memory
3. Implement custom state management using Runnable primitives
4. See migration guide: https://python.langchain.com/docs/versions/migrating_memory/

== See Also ==

* [[langchain-ai_langchain_Chain|Chain]] - Base chain class that integrates with memory
* [[langchain-ai_langchain_ConversationChain|ConversationChain]] - Chain implementation using memory
* Migration guide for memory patterns in LangChain v1.0
