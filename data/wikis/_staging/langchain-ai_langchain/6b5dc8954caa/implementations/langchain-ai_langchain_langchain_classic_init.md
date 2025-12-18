{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Docs|https://docs.langchain.com]]
|-
! Domains
| [[domain::Deprecation]], [[domain::API_Compatibility]], [[domain::Package_Management]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Package entry point for langchain_classic that manages deprecated imports and provides backward compatibility shims for legacy LangChain code.

=== Description ===

The `langchain_classic/__init__.py` module serves as the main entry point for the deprecated langchain_classic package. It uses Python's `__getattr__` mechanism to lazily load deprecated classes and functions while emitting deprecation warnings. This allows old code to continue working while guiding users toward the modern replacements in `langchain_community`, `langchain_core`, or partner packages.

=== Usage ===

This module is imported when users `import langchain_classic`. It should not be used for new code - instead, import from the specific packages mentioned in the deprecation warnings (e.g., `langchain_community.llms.OpenAI` instead of `langchain.OpenAI`).

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/__init__.py libs/langchain/langchain_classic/__init__.py]
* '''Lines:''' 1-424

=== Signature ===
<syntaxhighlight lang="python">
def _warn_on_import(name: str, replacement: str | None = None) -> None:
    """Warn on import of deprecated module.

    Args:
        name: Name of the deprecated import.
        replacement: Suggested replacement import path.
    """

def __getattr__(name: str) -> Any:
    """Lazy loader for deprecated imports with warnings."""

__all__ = [
    "FAISS", "Anthropic", "ArxivAPIWrapper", "Banana", "BasePromptTemplate",
    "CerebriumAI", "Cohere", "ConversationChain", "ElasticVectorSearch",
    # ... many more deprecated exports
]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Legacy import (deprecated)
from langchain_classic import OpenAI  # Will emit deprecation warning

# Modern replacement
from langchain_community.llms import OpenAI  # Preferred
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| name || str || Yes || Attribute name being accessed via __getattr__
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return || Any || The requested class/function from the appropriate submodule
|-
| warning || DeprecationWarning || Emitted when deprecated imports are used
|}

== Usage Examples ==

=== Legacy Import Pattern (Deprecated) ===
<syntaxhighlight lang="python">
# This works but emits deprecation warnings
from langchain_classic import OpenAI, PromptTemplate, LLMChain

# Warning: Importing OpenAI from langchain root module is no longer supported.
# Please use langchain_community.llms.OpenAI instead.
</syntaxhighlight>

=== Modern Replacement Pattern ===
<syntaxhighlight lang="python">
# Correct modern imports
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate

# For chains, use LCEL instead
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

prompt = PromptTemplate.from_template("Tell me a {adjective} joke")
chain = prompt | OpenAI()
result = chain.invoke({"adjective": "funny"})
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
* [[uses_heuristic::Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic]]

