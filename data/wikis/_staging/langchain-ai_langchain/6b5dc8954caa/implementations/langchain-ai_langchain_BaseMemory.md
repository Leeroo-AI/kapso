{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|Memory Migration|https://python.langchain.com/docs/versions/migrating_memory/]]
|-
! Domains
| [[domain::Memory]], [[domain::State_Management]], [[domain::Deprecated]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Deprecated abstract base class `BaseMemory` that defines the interface for maintaining state across Chain executions.

=== Description ===

The `BaseMemory` class is an abstract base for implementing memory systems that persist information across chain invocations. It defines methods for loading memory variables, saving context after chain runs, and clearing memory. Memory classes were used with the legacy Chain API to maintain conversational context, store facts, or implement working memory for agents.

=== Usage ===

This class is deprecated since version 0.3.3 and will be removed in 1.0.0. For new code, use LangGraph's persistence mechanisms or the new memory patterns described in the migration guide.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/base_memory.py libs/langchain/langchain_classic/base_memory.py]
* '''Lines:''' 1-116

=== Signature ===
<syntaxhighlight lang="python">
@deprecated(since="0.3.3", removal="1.0.0")
class BaseMemory(Serializable, ABC):
    """Abstract base class for memory in Chains.

    Abstract Methods (must override):
        memory_variables: Property returning keys this memory adds to inputs.
        load_memory_variables: Load memory state for chain inputs.
        save_context: Save chain inputs/outputs to memory.
        clear: Clear memory contents.
    """

    @property
    @abstractmethod
    def memory_variables(self) -> list[str]:
        """The string keys this memory class will add to chain inputs."""

    @abstractmethod
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Return key-value pairs given the text input to the chain."""

    async def aload_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Async version of load_memory_variables."""

    @abstractmethod
    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Save the context of this chain run to memory."""

    async def asave_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Async save the context of this chain run to memory."""

    @abstractmethod
    def clear(self) -> None:
        """Clear memory contents."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Legacy (deprecated)
from langchain_classic.base_memory import BaseMemory

# Or from memory module
from langchain.memory import BaseMemory
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| inputs || dict[str, Any] || Yes || Chain inputs to use for memory lookup/save
|-
| outputs || dict[str, str] || Yes (for save_context) || Chain outputs to save
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| memory_variables || list[str] || Keys that memory adds to chain inputs
|-
| load_memory_variables return || dict[str, Any] || Memory context to inject into chain
|}

== Usage Examples ==

=== Custom Memory Implementation (Legacy) ===
<syntaxhighlight lang="python">
from langchain_classic.base_memory import BaseMemory
from typing import Any

class SimpleFactMemory(BaseMemory):
    """Simple memory that stores facts as key-value pairs."""

    facts: dict[str, str] = {}

    @property
    def memory_variables(self) -> list[str]:
        return ["facts"]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        # Return all stored facts as a formatted string
        fact_str = "\n".join(f"- {k}: {v}" for k, v in self.facts.items())
        return {"facts": fact_str}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        # Extract and store any facts from the conversation
        if "new_fact" in outputs:
            key, value = outputs["new_fact"].split(":", 1)
            self.facts[key.strip()] = value.strip()

    def clear(self) -> None:
        self.facts = {}
</syntaxhighlight>

=== Using with Chain (Legacy) ===
<syntaxhighlight lang="python">
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

# Create memory and chain
memory = SimpleFactMemory()
prompt = PromptTemplate.from_template(
    "Facts:\n{facts}\n\nQuestion: {question}\nAnswer:"
)
chain = LLMChain(llm=OpenAI(), prompt=prompt, memory=memory)

# Memory is automatically loaded and saved
result = chain.invoke({"question": "What is the capital of France?"})
</syntaxhighlight>

=== Modern Alternative with LangGraph ===
<syntaxhighlight lang="python">
# Modern approach uses LangGraph persistence
from langgraph.checkpoint import MemorySaver
from langgraph.graph import StateGraph

# State includes memory as part of graph state
# See migration guide for full examples
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
* [[uses_heuristic::Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic]]

