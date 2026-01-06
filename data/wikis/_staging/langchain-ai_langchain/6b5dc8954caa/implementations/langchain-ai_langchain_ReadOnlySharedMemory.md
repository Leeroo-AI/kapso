{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Memory]], [[domain::Design_Patterns]], [[domain::Chains]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Memory wrapper `ReadOnlySharedMemory` that provides read-only access to another memory instance, preventing modifications.

=== Description ===

The `ReadOnlySharedMemory` class wraps an existing memory object and exposes its read operations while making `save_context` and `clear` no-ops. This is useful when multiple chains need to share memory but only one should be able to write to it - other chains can read the shared context without accidentally modifying it.

=== Usage ===

Use this wrapper when you have multiple chains that need to access the same conversation memory but only want one chain to update it. This prevents race conditions and ensures memory consistency.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/memory/readonly.py libs/langchain/langchain_classic/memory/readonly.py]
* '''Lines:''' 1-24

=== Signature ===
<syntaxhighlight lang="python">
class ReadOnlySharedMemory(BaseMemory):
    """Memory wrapper that is read-only and cannot be changed.

    Attributes:
        memory: The underlying memory instance to wrap.
    """

    memory: BaseMemory

    @property
    def memory_variables(self) -> list[str]:
        """Return memory variables from wrapped memory."""

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Load memory variables from wrapped memory."""

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """No-op: nothing is saved."""

    def clear(self) -> None:
        """No-op: nothing to clear."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.memory import ReadOnlySharedMemory
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| memory || BaseMemory || Yes || The memory instance to wrap read-only
|-
| inputs || dict[str, Any] || Yes (for load) || Inputs for loading memory
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return || dict[str, str] || Memory variables from wrapped memory
|}

== Usage Examples ==

=== Shared Memory Between Chains ===
<syntaxhighlight lang="python">
from langchain_classic.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain_classic.chains import ConversationChain, LLMChain
from langchain_openai import OpenAI

# Create shared memory
shared_memory = ConversationBufferMemory()

# Main conversation chain can write
main_chain = ConversationChain(
    llm=OpenAI(),
    memory=shared_memory,  # Can read and write
)

# Secondary chain gets read-only access
readonly_memory = ReadOnlySharedMemory(memory=shared_memory)
analysis_chain = LLMChain(
    llm=OpenAI(),
    memory=readonly_memory,  # Can only read
    prompt=analysis_prompt,
)

# Main chain updates memory
main_chain.predict(input="Hello, I'm Bob")
main_chain.predict(input="I live in New York")

# Analysis chain can see the history but won't modify it
result = analysis_chain.predict(text="Summarize what you know")

# Memory still only contains main chain's updates
print(shared_memory.buffer)
</syntaxhighlight>

=== Multi-Agent Scenario ===
<syntaxhighlight lang="python">
# Agent architecture where coordinator writes, workers read
coordinator_memory = ConversationBufferMemory()

# Coordinator can update memory
coordinator = ConversationChain(
    llm=OpenAI(),
    memory=coordinator_memory,
)

# Multiple worker agents with read-only access
workers = [
    ConversationChain(
        llm=OpenAI(),
        memory=ReadOnlySharedMemory(memory=coordinator_memory),
    )
    for _ in range(3)
]

# Coordinator sets context
coordinator.predict(input="Task: Research AI safety")

# Workers can see context but won't pollute the shared memory
for worker in workers:
    result = worker.predict(input="What's my task?")
    # Workers see "Research AI safety" but don't save their responses
</syntaxhighlight>

=== Preventing Accidental Modifications ===
<syntaxhighlight lang="python">
# Protect memory during read-only operations
memory = ConversationBufferMemory()
memory.save_context({"input": "hi"}, {"output": "hello"})

readonly = ReadOnlySharedMemory(memory=memory)

# These are no-ops
readonly.save_context({"input": "test"}, {"output": "ignored"})
readonly.clear()

# Original memory unchanged
print(memory.buffer)  # Still contains "hi" -> "hello"
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
* [[uses_heuristic::Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic]]

