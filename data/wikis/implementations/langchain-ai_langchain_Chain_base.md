{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Docs|https://docs.langchain.com]]
|-
! Domains
| [[domain::Chains]], [[domain::Abstractions]], [[domain::LLM_Orchestration]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Abstract base class `Chain` that defines the interface for creating structured sequences of calls to LLM components with memory, callbacks, and composability support.

=== Description ===

The `Chain` class is the foundational abstraction for building multi-step LLM applications in LangChain. It extends `RunnableSerializable` and provides a standardized interface for: state management through optional `Memory` objects, observability through `Callbacks`, and composition with other chains. The class defines abstract methods `input_keys`, `output_keys`, and `_call` that concrete implementations must provide.

=== Usage ===

Subclass `Chain` when building custom multi-step pipelines that need memory persistence, callback integration, or serialization. For simpler use cases, prefer using LangChain Expression Language (LCEL) with the `|` operator instead.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/chains/base.py libs/langchain/langchain_classic/chains/base.py]
* '''Lines:''' 1-806

=== Signature ===
<syntaxhighlight lang="python">
class Chain(RunnableSerializable[dict[str, Any], dict[str, Any]], ABC):
    """Abstract base class for creating structured sequences of calls to components.

    Attributes:
        memory: Optional memory object for state persistence across calls.
        callbacks: Optional callback handlers for observability.
        verbose: Whether to run in verbose mode with intermediate logs.
        tags: Optional tags for callback identification.
        metadata: Optional metadata for callback context.
    """

    memory: BaseMemory | None = None
    callbacks: Callbacks = Field(default=None, exclude=True)
    verbose: bool = Field(default_factory=_get_verbosity)
    tags: list[str] | None = None
    metadata: builtins.dict[str, Any] | None = None

    @property
    @abstractmethod
    def input_keys(self) -> list[str]:
        """Keys expected to be in the chain input."""

    @property
    @abstractmethod
    def output_keys(self) -> list[str]:
        """Keys expected to be in the chain output."""

    @abstractmethod
    def _call(
        self,
        inputs: builtins.dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> builtins.dict[str, Any]:
        """Execute the chain (override in subclasses)."""

    def invoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the chain with the Runnable interface."""

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async execute the chain."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.chains.base import Chain
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| input || dict[str, Any] || Yes || Dictionary of inputs matching `input_keys`
|-
| config || RunnableConfig || No || Runtime configuration for callbacks, tags, metadata
|-
| return_only_outputs || bool || No || If True, only return outputs (not inputs)
|-
| include_run_info || bool || No || If True, include run metadata in response
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return || dict[str, Any] || Dictionary of outputs matching `output_keys`
|-
| RUN_KEY || RunInfo || Run metadata (if include_run_info=True)
|}

== Usage Examples ==

=== Custom Chain Subclass ===
<syntaxhighlight lang="python">
from langchain_classic.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun
from typing import Any

class MyCustomChain(Chain):
    """A custom chain that transforms input text."""

    transform_type: str = "uppercase"

    @property
    def input_keys(self) -> list[str]:
        return ["text"]

    @property
    def output_keys(self) -> list[str]:
        return ["transformed"]

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        text = inputs["text"]
        if self.transform_type == "uppercase":
            result = text.upper()
        else:
            result = text.lower()
        return {"transformed": result}

# Usage
chain = MyCustomChain(transform_type="uppercase")
result = chain.invoke({"text": "hello world"})
# {"text": "hello world", "transformed": "HELLO WORLD"}
</syntaxhighlight>

=== Using with Memory ===
<syntaxhighlight lang="python">
from langchain_classic.chains.base import Chain
from langchain_classic.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
chain = MyCustomChain(memory=memory)

# Memory variables are automatically loaded and saved
result = chain.invoke({"text": "hello"})
</syntaxhighlight>

=== LCEL Alternative (Preferred for New Code) ===
<syntaxhighlight lang="python">
# Modern approach using LCEL instead of Chain subclass
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate.from_template("Transform this: {text}")
model = ChatOpenAI()
parser = StrOutputParser()

# Compose with | operator
chain = prompt | model | parser
result = chain.invoke({"text": "hello"})
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
* [[uses_heuristic::Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic]]

