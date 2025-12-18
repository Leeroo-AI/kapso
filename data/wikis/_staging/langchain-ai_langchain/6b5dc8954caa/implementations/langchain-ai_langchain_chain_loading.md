{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Docs|https://docs.langchain.com]]
|-
! Domains
| [[domain::Serialization]], [[domain::Chains]], [[domain::Deprecated]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Deprecated chain serialization module that loads Chain objects from JSON/YAML configuration files with support for multiple chain types.

=== Description ===

The `loading.py` module provides functions to deserialize Chain objects from configuration files. It supports a registry of chain types (LLMChain, StuffDocumentsChain, MapReduceDocumentsChain, etc.) and recursively loads nested chains, prompts, and LLMs. The module is deprecated since version 0.2.13 - chains should now be imported directly from their modules rather than loaded from config files.

=== Usage ===

This module was used for loading chains from saved JSON/YAML files or from the (now deprecated) LangChain Hub. For new code, instantiate chains directly in Python or use LCEL composition instead.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/chains/loading.py libs/langchain/langchain_classic/chains/loading.py]
* '''Lines:''' 1-742

=== Signature ===
<syntaxhighlight lang="python">
@deprecated(
    since="0.2.13",
    message="This function is deprecated and will be removed in langchain 1.0.",
    removal="1.0",
)
def load_chain(path: str | Path, **kwargs: Any) -> Chain:
    """Unified method for loading a chain from LangChainHub or local fs.

    Args:
        path: Path to JSON/YAML file or lc:// hub path (deprecated).
        **kwargs: Additional kwargs like vectorstore, retriever, embeddings.

    Returns:
        Deserialized Chain instance.
    """

@deprecated(since="0.2.13", removal="1.0")
def load_chain_from_config(config: dict, **kwargs: Any) -> Chain:
    """Load chain from Config Dict.

    Args:
        config: Dictionary with '_type' key specifying chain type.
        **kwargs: Runtime dependencies (vectorstore, retriever, etc.).

    Returns:
        Instantiated Chain object.
    """

type_to_loader_dict = {
    "api_chain": _load_api_chain,
    "hyde_chain": _load_hyde_chain,
    "llm_chain": _load_llm_chain,
    "llm_checker_chain": _load_llm_checker_chain,
    "llm_math_chain": _load_llm_math_chain,
    "stuff_documents_chain": _load_stuff_documents_chain,
    "map_reduce_documents_chain": _load_map_reduce_documents_chain,
    "refine_documents_chain": _load_refine_documents_chain,
    "qa_with_sources_chain": _load_qa_with_sources_chain,
    "retrieval_qa": _load_retrieval_qa,
    # ... more chain types
}
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Legacy (deprecated)
from langchain_classic.chains.loading import load_chain, load_chain_from_config
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| path || str or Path || Yes || Path to JSON/YAML config file
|-
| config || dict || Yes (for load_chain_from_config) || Config dict with '_type' key
|-
| vectorstore || VectorStore || Conditional || Required for VectorDBQA chains
|-
| retriever || BaseRetriever || Conditional || Required for RetrievalQA chains
|-
| embeddings || Embeddings || Conditional || Required for HYDE chains
|-
| graph || Graph || Conditional || Required for GraphCypherQA chains
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return || Chain || Instantiated chain object of the specified type
|}

== Usage Examples ==

=== Loading from File (Deprecated) ===
<syntaxhighlight lang="python">
from langchain_classic.chains.loading import load_chain

# Load from JSON file
chain = load_chain("path/to/chain.json")

# Load from YAML file
chain = load_chain("path/to/chain.yaml")

# Load with runtime dependencies
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.load_local("my_index")
chain = load_chain("qa_chain.json", vectorstore=vectorstore)
</syntaxhighlight>

=== Config File Format ===
<syntaxhighlight lang="json">
{
    "_type": "llm_chain",
    "prompt": {
        "_type": "prompt",
        "template": "Tell me about {topic}",
        "input_variables": ["topic"]
    },
    "llm": {
        "_type": "openai",
        "model_name": "gpt-3.5-turbo"
    }
}
</syntaxhighlight>

=== Modern Alternative ===
<syntaxhighlight lang="python">
# Instead of loading from config, instantiate directly
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate.from_template("Tell me about {topic}")
llm = ChatOpenAI(model="gpt-3.5-turbo")
chain = prompt | llm | StrOutputParser()

# For persistence, use LangSmith or serialize the components individually
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
* [[uses_heuristic::Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic]]

