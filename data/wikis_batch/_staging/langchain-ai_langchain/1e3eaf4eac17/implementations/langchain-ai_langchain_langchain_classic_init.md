{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai/langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Python]], [[domain::Package Management]], [[domain::Deprecation]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==
Lazy import system with deprecation warnings for the langchain_classic package that provides backward compatibility for legacy imports.

=== Description ===
The __init__.py file serves as the main entrypoint for the langchain_classic package (formerly the langchain package). It implements a sophisticated lazy import mechanism using Python's __getattr__ protocol to provide backward compatibility while warning users about deprecated import patterns. The module intercepts attribute access and dynamically imports the requested classes from their new locations in langchain_classic, langchain_core, or langchain_community packages.

This implementation allows legacy code using old import paths to continue functioning while encouraging migration to new import patterns. It handles 60+ classes including chains (LLMChain, ConversationChain), LLMs (OpenAI, Anthropic, Cohere), prompts (PromptTemplate, FewShotPromptTemplate), utilities (WikipediaAPIWrapper, SQLDatabase), and vector stores (FAISS, ElasticVectorSearch).

=== Usage ===
This module is automatically loaded when users import from the langchain_classic package. It intercepts deprecated imports and redirects them to the correct locations while emitting deprecation warnings, except in interactive environments where warnings are suppressed to avoid polluting auto-complete experiences.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai/langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/__init__.py libs/langchain/langchain_classic/__init__.py]
* '''Lines:''' 1-425

=== Signature ===
<syntaxhighlight lang="python">
def __getattr__(name: str) -> Any
def _warn_on_import(name: str, replacement: str | None = None) -> None
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Legacy imports that trigger the lazy loading system
from langchain_classic import LLMChain  # Warns and redirects to langchain_classic.chains.LLMChain
from langchain_classic import OpenAI    # Warns and redirects to langchain_community.llms.OpenAI
from langchain_classic import PromptTemplate  # Warns and redirects to langchain_core.prompts.PromptTemplate

# Modern recommended imports
from langchain_classic.chains import LLMChain
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate
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
| class/module || Any || The imported class or module from the appropriate package
|}

== Lazy Import Mechanism ==

=== __getattr__ Protocol ===
Python's __getattr__ is invoked when an attribute is not found through normal lookup. This implementation uses it to intercept imports and dynamically load classes:

<syntaxhighlight lang="python">
def __getattr__(name: str) -> Any:
    if name == "LLMChain":
        from langchain_classic.chains import LLMChain
        _warn_on_import(name, replacement="langchain_classic.chains.LLMChain")
        return LLMChain
    # ... 60+ more cases
    msg = f"Could not find: {name}"
    raise AttributeError(msg)
</syntaxhighlight>

=== Deprecation Warning System ===
The _warn_on_import function manages deprecation warnings:
* Checks if running in interactive environment (Jupyter, IPython, REPL)
* Suppresses warnings in interactive environments to avoid polluting auto-complete
* Emits stacklevel=3 warnings to show the user's code location (not the import system)
* Provides specific replacement paths for each deprecated import

== Supported Import Categories ==

=== Agents ===
* MRKLChain → langchain_classic.agents.MRKLChain
* ReActChain → langchain_classic.agents.ReActChain
* SelfAskWithSearchChain → langchain_classic.agents.SelfAskWithSearchChain

=== Chains ===
* ConversationChain → langchain_classic.chains.ConversationChain
* LLMChain → langchain_classic.chains.LLMChain
* LLMCheckerChain → langchain_classic.chains.LLMCheckerChain
* LLMMathChain → langchain_classic.chains.LLMMathChain
* QAWithSourcesChain → langchain_classic.chains.QAWithSourcesChain
* VectorDBQA → langchain_classic.chains.VectorDBQA
* VectorDBQAWithSourcesChain → langchain_classic.chains.VectorDBQAWithSourcesChain
* LLMBashChain → Raises ImportError (moved to langchain-experimental)

=== Docstores ===
* InMemoryDocstore → langchain_community.docstore.InMemoryDocstore
* Wikipedia → langchain_community.docstore.Wikipedia

=== LLMs ===
* OpenAI, Anthropic, Cohere, HuggingFaceHub, LlamaCpp, etc. → langchain_community.llms.*
* 20+ LLM providers supported

=== Prompts ===
* PromptTemplate → langchain_core.prompts.PromptTemplate
* FewShotPromptTemplate → langchain_core.prompts.FewShotPromptTemplate
* BasePromptTemplate → langchain_core.prompts.BasePromptTemplate
* Prompt → langchain_core.prompts.PromptTemplate (renamed)

=== Utilities ===
* WikipediaAPIWrapper → langchain_community.utilities.WikipediaAPIWrapper
* GoogleSearchAPIWrapper → langchain_community.utilities.GoogleSearchAPIWrapper
* SQLDatabase → langchain_community.utilities.SQLDatabase
* 10+ utility wrappers

=== Vector Stores ===
* FAISS → langchain_community.vectorstores.FAISS
* ElasticVectorSearch → langchain_community.vectorstores.ElasticVectorSearch

=== Special Cases ===
* SerpAPIChain, SerpAPIWrapper → langchain_community.utilities.SerpAPIWrapper (backward compatibility alias)

== Version Information ==

The module retrieves version from package metadata:
<syntaxhighlight lang="python">
try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = ""
</syntaxhighlight>

== Deprecation Surface ==

The module surfaces LangChain deprecation warnings at import time:
<syntaxhighlight lang="python">
from langchain_core._api.deprecation import surface_langchain_deprecation_warnings
surface_langchain_deprecation_warnings()
</syntaxhighlight>

== Interactive Environment Detection ==

The system detects interactive environments to suppress warnings during auto-complete:
<syntaxhighlight lang="python">
from langchain_classic._api.interactive_env import is_interactive_env

if is_interactive_env():
    # No warnings in Jupyter, IPython, or Python REPL
    return
</syntaxhighlight>

== __all__ Export ==

Defines explicit exports for documentation and IDE support:
<syntaxhighlight lang="python">
__all__ = [
    "FAISS",
    "Anthropic",
    "ArxivAPIWrapper",
    # ... 60+ more exports
    "Writer",
]
</syntaxhighlight>

== Usage Examples ==

=== Legacy Import (Deprecated) ===
<syntaxhighlight lang="python">
# Old style - triggers deprecation warning
from langchain_classic import LLMChain, OpenAI, PromptTemplate

# Warning: Importing LLMChain from langchain root module is no longer supported.
# Please use langchain_classic.chains.LLMChain instead.

prompt = PromptTemplate(input_variables=["topic"], template="Tell me about {topic}")
llm = OpenAI()
chain = LLMChain(llm=llm, prompt=prompt)
</syntaxhighlight>

=== Modern Import (Recommended) ===
<syntaxhighlight lang="python">
# New style - no warnings
from langchain_classic.chains import LLMChain
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(input_variables=["topic"], template="Tell me about {topic}")
llm = OpenAI()
chain = LLMChain(llm=llm, prompt=prompt)
</syntaxhighlight>

=== Interactive Environment ===
<syntaxhighlight lang="python">
# In Jupyter notebook or IPython
from langchain_classic import LLMChain  # No warning to avoid polluting output
# Tab-completion works without warning spam
</syntaxhighlight>

=== Handling Moved Packages ===
<syntaxhighlight lang="python">
# Some chains moved to langchain-experimental
try:
    from langchain_classic import LLMBashChain
except ImportError as e:
    # ImportError: This module has been moved to langchain-experimental.
    # Install with: pip install langchain-experimental
    # Use: from langchain_experimental.llm_bash.base import LLMBashChain
    pass
</syntaxhighlight>

== Migration Guide ==

{| class="wikitable"
|-
! Old Import !! New Import !! Package
|-
| from langchain_classic import LLMChain || from langchain_classic.chains import LLMChain || langchain
|-
| from langchain_classic import OpenAI || from langchain_community.llms import OpenAI || langchain-community
|-
| from langchain_classic import PromptTemplate || from langchain_core.prompts import PromptTemplate || langchain-core
|-
| from langchain_classic import FAISS || from langchain_community.vectorstores import FAISS || langchain-community
|-
| from langchain_classic import LLMBashChain || from langchain_experimental.llm_bash.base import LLMBashChain || langchain-experimental
|}

== Related Pages ==
* [[implemented_by::Implementation:langchain-ai_langchain_Chain]]
* [[uses::Concept:Lazy_Import]]
* [[uses::Concept:Deprecation_Strategy]]
* [[uses::Concept:Backward_Compatibility]]
