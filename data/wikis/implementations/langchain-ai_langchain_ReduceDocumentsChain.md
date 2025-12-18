{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|Map-Reduce Migration|https://python.langchain.com/docs/versions/migrating_chains/map_reduce_chain/]]
|-
! Domains
| [[domain::Chains]], [[domain::Document_Processing]], [[domain::Deprecated]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Deprecated `ReduceDocumentsChain` and utility functions for recursively collapsing documents to fit within token limits.

=== Description ===

The `ReduceDocumentsChain` recursively combines documents using a collapse strategy when they exceed token limits. It splits documents into groups, collapses each group, and repeats until all documents fit. Utility functions `split_list_of_docs` and `collapse_docs` provide the core logic for splitting by token count and merging metadata.

=== Usage ===

This class is deprecated since version 0.3.1. For new code, use LangGraph's map-reduce patterns which provide better control and streaming support.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/chains/combine_documents/reduce.py libs/langchain/langchain_classic/chains/combine_documents/reduce.py]
* '''Lines:''' 1-389

=== Signature ===
<syntaxhighlight lang="python">
def split_list_of_docs(
    docs: list[Document],
    length_func: Callable,
    token_max: int,
    **kwargs: Any,
) -> list[list[Document]]:
    """Split documents into subsets meeting token constraints."""


def collapse_docs(
    docs: list[Document],
    combine_document_func: CombineDocsProtocol,
    **kwargs: Any,
) -> Document:
    """Collapse documents into one, merging metadata."""


@deprecated(since="0.3.1", removal="1.0")
class ReduceDocumentsChain(BaseCombineDocumentsChain):
    """Combine documents by recursively reducing them.

    Attributes:
        combine_documents_chain: Final chain for combining documents.
        collapse_documents_chain: Optional chain for collapsing intermediates.
        token_max: Maximum tokens per group (default: 3000).
        collapse_max_retries: Max collapse attempts before error.
    """

    combine_documents_chain: BaseCombineDocumentsChain
    collapse_documents_chain: BaseCombineDocumentsChain | None = None
    token_max: int = 3000
    collapse_max_retries: int | None = None
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Legacy (deprecated)
from langchain_classic.chains import ReduceDocumentsChain
from langchain_classic.chains.combine_documents.reduce import (
    split_list_of_docs,
    collapse_docs,
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| input_documents || list[Document] || Yes || Documents to reduce
|-
| token_max || int || No || Override default token maximum
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| output_text || str || Final combined output
|}

== Usage Examples ==

=== Basic Reduce Chain (Legacy) ===
<syntaxhighlight lang="python">
from langchain_classic.chains import ReduceDocumentsChain, StuffDocumentsChain, LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

# Create the combine chain
combine_prompt = PromptTemplate.from_template("Summarize: {context}")
llm_chain = LLMChain(llm=OpenAI(), prompt=combine_prompt)
combine_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
)

# Create reduce chain
reduce_chain = ReduceDocumentsChain(
    combine_documents_chain=combine_chain,
    token_max=3000,
)

# Will recursively collapse if docs exceed 3000 tokens
result = reduce_chain.invoke({"input_documents": docs})
</syntaxhighlight>

=== With Separate Collapse Chain ===
<syntaxhighlight lang="python">
# Use different prompt for collapse vs final combine
collapse_prompt = PromptTemplate.from_template("Condense: {context}")
collapse_llm_chain = LLMChain(llm=OpenAI(), prompt=collapse_prompt)
collapse_chain = StuffDocumentsChain(
    llm_chain=collapse_llm_chain,
    document_variable_name="context",
)

reduce_chain = ReduceDocumentsChain(
    combine_documents_chain=combine_chain,
    collapse_documents_chain=collapse_chain,  # Used for intermediate collapses
    token_max=4000,
    collapse_max_retries=5,  # Fail after 5 attempts
)
</syntaxhighlight>

=== Using Utility Functions ===
<syntaxhighlight lang="python">
from langchain_classic.chains.combine_documents.reduce import (
    split_list_of_docs,
    collapse_docs,
)

# Split documents by token count
def count_tokens(docs):
    return sum(len(d.page_content) // 4 for d in docs)

groups = split_list_of_docs(
    docs,
    length_func=count_tokens,
    token_max=1000,
)
# Returns list of document lists, each under 1000 tokens

# Collapse a group into single document
def combine_func(docs):
    return " ".join(d.page_content for d in docs)

collapsed = collapse_docs(groups[0], combine_func)
# Returns single Document with merged metadata
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
* [[uses_heuristic::Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic]]

