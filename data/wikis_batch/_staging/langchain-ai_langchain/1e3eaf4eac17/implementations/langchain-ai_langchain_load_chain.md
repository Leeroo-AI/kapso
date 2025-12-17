{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai/langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::LangChain]], [[domain::Serialization]], [[domain::Chain Loading]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==
Deprecated utilities for loading and deserializing LangChain chains from YAML/JSON configuration files.

=== Description ===
The loading.py module provides a comprehensive system for serializing and deserializing LangChain chains to/from configuration files. It implements a registry of loader functions that can reconstruct complex chain objects from declarative configurations, supporting 20+ chain types including LLMChain, MapReduceDocumentsChain, RetrievalQA, and specialized chains like HyDE and API chains.

The module handles nested chain configurations where chains can reference other chains, supports both inline configurations and file path references, loads LLMs, prompts, and embeddings from configurations, and manages special requirements like vectorstores and retrievers passed as kwargs. While deprecated since version 0.2.13 in favor of direct imports, this system provided a way to persist and share chain configurations.

=== Usage ===
This module was used to save chain configurations to files and reload them later, share chain configurations across projects, and load pre-built chains from configuration templates. It is now deprecated and will be removed in LangChain 1.0. Modern code should construct chains directly using imports.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai/langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/chains/loading.py libs/langchain/langchain_classic/chains/loading.py]
* '''Lines:''' 1-743

=== Signature ===
<syntaxhighlight lang="python">
@deprecated(since="0.2.13", removal="1.0")
def load_chain(path: str | Path, **kwargs: Any) -> Chain:
    """Unified method for loading a chain from LangChainHub or local fs."""

@deprecated(since="0.2.13", removal="1.0")
def load_chain_from_config(config: dict, **kwargs: Any) -> Chain:
    """Load chain from Config Dict."""

# Internal loader functions for specific chain types
def _load_llm_chain(config: dict, **kwargs: Any) -> LLMChain
def _load_stuff_documents_chain(config: dict, **kwargs: Any) -> StuffDocumentsChain
def _load_map_reduce_documents_chain(config: dict, **kwargs: Any) -> MapReduceDocumentsChain
def _load_refine_documents_chain(config: dict, **kwargs: Any) -> RefineDocumentsChain
def _load_retrieval_qa(config: dict, **kwargs: Any) -> RetrievalQA
# ... and 15+ more loader functions
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Deprecated
from langchain_classic.chains.loading import load_chain, load_chain_from_config

# Modern alternative - construct chains directly
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

chain = LLMChain(llm=OpenAI(), prompt=PromptTemplate(...))
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| path || str or Path || Yes (load_chain) || Path to YAML or JSON configuration file
|-
| config || dict || Yes (load_chain_from_config) || Dictionary containing chain configuration
|-
| kwargs || Any || No || Additional parameters (e.g., vectorstore, retriever, embeddings)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| chain || Chain || Reconstructed chain object ready for use
|}

== Configuration Format ==

=== Basic Structure ===
<syntaxhighlight lang="yaml">
_type: "llm_chain"
llm:
  _type: "openai"
  temperature: 0.7
  model_name: "gpt-3.5-turbo"
prompt:
  _type: "prompt"
  template: "Tell me a {adjective} joke"
  input_variables: ["adjective"]
verbose: true
</syntaxhighlight>

=== File References ===
<syntaxhighlight lang="yaml">
_type: "llm_chain"
llm_path: "./configs/my_llm.yaml"
prompt_path: "./prompts/joke_prompt.yaml"
</syntaxhighlight>

=== Nested Chains ===
<syntaxhighlight lang="yaml">
_type: "map_reduce_documents_chain"
llm_chain:
  _type: "llm_chain"
  llm:
    _type: "openai"
    temperature: 0
  prompt:
    template: "Summarize: {text}"
    input_variables: ["text"]
reduce_documents_chain:
  combine_documents_chain:
    _type: "stuff_documents_chain"
    llm_chain:
      _type: "llm_chain"
      # ...
</syntaxhighlight>

== Supported Chain Types ==

The type_to_loader_dict registry maps chain type strings to loader functions:

{| class="wikitable"
|-
! Type String !! Chain Class !! Loader Function
|-
| llm_chain || LLMChain || _load_llm_chain
|-
| stuff_documents_chain || StuffDocumentsChain || _load_stuff_documents_chain
|-
| map_reduce_documents_chain || MapReduceDocumentsChain || _load_map_reduce_documents_chain
|-
| reduce_documents_chain || ReduceDocumentsChain || _load_reduce_documents_chain
|-
| map_rerank_documents_chain || MapRerankDocumentsChain || _load_map_rerank_documents_chain
|-
| refine_documents_chain || RefineDocumentsChain || _load_refine_documents_chain
|-
| qa_with_sources_chain || QAWithSourcesChain || _load_qa_with_sources_chain
|-
| retrieval_qa || RetrievalQA || _load_retrieval_qa
|-
| retrieval_qa_with_sources_chain || RetrievalQAWithSourcesChain || _load_retrieval_qa_with_sources_chain
|-
| vector_db_qa || VectorDBQA || _load_vector_db_qa
|-
| vector_db_qa_with_sources_chain || VectorDBQAWithSourcesChain || _load_vector_db_qa_with_sources_chain
|-
| hyde_chain || HypotheticalDocumentEmbedder || _load_hyde_chain
|-
| llm_checker_chain || LLMCheckerChain || _load_llm_checker_chain
|-
| llm_math_chain || LLMMathChain || _load_llm_math_chain
|-
| api_chain || APIChain || _load_api_chain
|-
| llm_requests_chain || LLMRequestsChain || _load_llm_requests_chain
|-
| graph_cypher_chain || GraphCypherQAChain || _load_graph_cypher_chain
|-
| llm_bash_chain || N/A || Raises NotImplementedError (moved to experimental)
|-
| pal_chain || N/A || Raises NotImplementedError (moved to experimental)
|-
| sql_database_chain || N/A || Raises NotImplementedError (moved to experimental)
|}

== Core Loader Functions ==

=== _load_llm_chain ===
<syntaxhighlight lang="python">
def _load_llm_chain(config: dict, **kwargs: Any) -> LLMChain:
    """Load LLM chain from config dict."""
    # Load LLM from config or path
    if "llm" in config:
        llm_config = config.pop("llm")
        llm = load_llm_from_config(llm_config, **kwargs)
    elif "llm_path" in config:
        llm = load_llm(config.pop("llm_path"), **kwargs)
    else:
        raise ValueError("One of `llm` or `llm_path` must be present.")

    # Load prompt from config or path
    if "prompt" in config:
        prompt_config = config.pop("prompt")
        prompt = load_prompt_from_config(prompt_config)
    elif "prompt_path" in config:
        prompt = load_prompt(config.pop("prompt_path"))
    else:
        raise ValueError("One of `prompt` or `prompt_path` must be present.")

    _load_output_parser(config)
    return LLMChain(llm=llm, prompt=prompt, **config)
</syntaxhighlight>

=== _load_stuff_documents_chain ===
<syntaxhighlight lang="python">
def _load_stuff_documents_chain(config: dict, **kwargs: Any) -> StuffDocumentsChain:
    # Load LLM chain
    if "llm_chain" in config:
        llm_chain_config = config.pop("llm_chain")
        llm_chain = load_chain_from_config(llm_chain_config, **kwargs)
    elif "llm_chain_path" in config:
        llm_chain = load_chain(config.pop("llm_chain_path"), **kwargs)
    else:
        raise ValueError("One of `llm_chain` or `llm_chain_path` must be present.")

    if not isinstance(llm_chain, LLMChain):
        raise ValueError(f"Expected LLMChain, got {llm_chain}")

    # Load document prompt
    if "document_prompt" in config:
        prompt_config = config.pop("document_prompt")
        document_prompt = load_prompt_from_config(prompt_config)
    elif "document_prompt_path" in config:
        document_prompt = load_prompt(config.pop("document_prompt_path"))
    else:
        raise ValueError("One of `document_prompt` or `document_prompt_path` must be present.")

    return StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        **config,
    )
</syntaxhighlight>

=== _load_map_reduce_documents_chain ===
<syntaxhighlight lang="python">
def _load_map_reduce_documents_chain(
    config: dict,
    **kwargs: Any,
) -> MapReduceDocumentsChain:
    # Load map LLM chain
    if "llm_chain" in config:
        llm_chain_config = config.pop("llm_chain")
        llm_chain = load_chain_from_config(llm_chain_config, **kwargs)
    elif "llm_chain_path" in config:
        llm_chain = load_chain(config.pop("llm_chain_path"), **kwargs)
    else:
        raise ValueError("One of `llm_chain` or `llm_chain_path` must be present.")

    # Load reduce documents chain
    if "reduce_documents_chain" in config:
        reduce_documents_chain = load_chain_from_config(
            config.pop("reduce_documents_chain"),
            **kwargs,
        )
    elif "reduce_documents_chain_path" in config:
        reduce_documents_chain = load_chain(
            config.pop("reduce_documents_chain_path"),
            **kwargs,
        )
    else:
        reduce_documents_chain = _load_reduce_documents_chain(config, **kwargs)

    return MapReduceDocumentsChain(
        llm_chain=llm_chain,
        reduce_documents_chain=reduce_documents_chain,
        **config,
    )
</syntaxhighlight>

=== _load_retrieval_qa ===
<syntaxhighlight lang="python">
def _load_retrieval_qa(config: dict, **kwargs: Any) -> RetrievalQA:
    # Retriever must be passed via kwargs
    if "retriever" in kwargs:
        retriever = kwargs.pop("retriever")
    else:
        raise ValueError("`retriever` must be present.")

    # Load combine documents chain
    if "combine_documents_chain" in config:
        combine_documents_chain_config = config.pop("combine_documents_chain")
        combine_documents_chain = load_chain_from_config(
            combine_documents_chain_config,
            **kwargs,
        )
    elif "combine_documents_chain_path" in config:
        combine_documents_chain = load_chain(
            config.pop("combine_documents_chain_path"),
            **kwargs,
        )
    else:
        raise ValueError(
            "One of `combine_documents_chain` or "
            "`combine_documents_chain_path` must be present."
        )

    return RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        retriever=retriever,
        **config,
    )
</syntaxhighlight>

== File Loading ==

=== _load_chain_from_file ===
<syntaxhighlight lang="python">
def _load_chain_from_file(file: str | Path, **kwargs: Any) -> Chain:
    """Load chain from file."""
    file_path = Path(file) if isinstance(file, str) else file

    # Load JSON or YAML
    if file_path.suffix == ".json":
        with file_path.open() as f:
            config = json.load(f)
    elif file_path.suffix in (".yaml", ".yml"):
        with file_path.open() as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("File type must be json or yaml")

    # Override verbose and memory
    if "verbose" in kwargs:
        config["verbose"] = kwargs.pop("verbose")
    if "memory" in kwargs:
        config["memory"] = kwargs.pop("memory")

    return load_chain_from_config(config, **kwargs)
</syntaxhighlight>

== Special Cases ==

=== Moved to Experimental ===
Some chains raise NotImplementedError because they've moved to langchain-experimental for security reasons:

<syntaxhighlight lang="python">
def _load_llm_bash_chain(config: dict, **kwargs: Any) -> Any:
    """Load LLM Bash chain from config dict."""
    raise NotImplementedError(
        "LLMBash Chain is not available through LangChain anymore. "
        "The relevant code can be found in langchain_experimental, "
        "but it is not appropriate for production usage due to security "
        "concerns. Please refer to langchain-experimental repository for more details."
    )

def _load_pal_chain(config: dict, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "PALChain is not available through LangChain anymore. "
        "The relevant code can be found in langchain_experimental, "
        "but it is not appropriate for production usage due to security "
        "concerns. Please refer to langchain-experimental repository for more details."
    )

def _load_sql_database_chain(config: dict, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "SQLDatabaseChain is not available through LangChain anymore. "
        "The relevant code can be found in langchain_experimental, "
        "but it is not appropriate for production usage due to security "
        "concerns. Please refer to langchain-experimental repository for more details, "
        "or refer to this tutorial for best practices: "
        "https://python.langchain.com/docs/tutorials/sql_qa/"
    )
</syntaxhighlight>

=== Requires External Dependencies ===
Some chains require objects passed via kwargs:

<syntaxhighlight lang="python">
def _load_hyde_chain(config: dict, **kwargs: Any) -> HypotheticalDocumentEmbedder:
    # ... load llm_chain ...

    if "embeddings" in kwargs:
        embeddings = kwargs.pop("embeddings")
    else:
        raise ValueError("`embeddings` must be present.")

    return HypotheticalDocumentEmbedder(
        llm_chain=llm_chain,
        base_embeddings=embeddings,
        **config,
    )
</syntaxhighlight>

== Usage Examples ==

=== Save and Load Chain ===
<syntaxhighlight lang="python">
from langchain_classic.chains import LLMChain
from langchain_classic.chains.loading import load_chain
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

# Create chain
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write about {topic}"
)
chain = LLMChain(llm=OpenAI(), prompt=prompt)

# Save to file
chain.save("my_chain.yaml")

# Load from file
loaded_chain = load_chain("my_chain.yaml")
result = loaded_chain.invoke({"topic": "AI"})
</syntaxhighlight>

=== YAML Configuration Example ===
<syntaxhighlight lang="yaml">
# my_chain.yaml
_type: llm_chain
llm:
  _type: openai
  temperature: 0.7
  model_name: gpt-3.5-turbo
prompt:
  _type: prompt
  template: "Write about {topic}"
  input_variables:
    - topic
verbose: false
</syntaxhighlight>

=== Load with Override ===
<syntaxhighlight lang="python">
from langchain_classic.chains.loading import load_chain
from langchain_classic.memory import ConversationBufferMemory

# Load chain and override settings
memory = ConversationBufferMemory()
chain = load_chain("my_chain.yaml", verbose=True, memory=memory)
</syntaxhighlight>

=== Load from Config Dict ===
<syntaxhighlight lang="python">
from langchain_classic.chains.loading import load_chain_from_config

config = {
    "_type": "llm_chain",
    "llm": {
        "_type": "openai",
        "temperature": 0.5
    },
    "prompt": {
        "_type": "prompt",
        "template": "Question: {question}\\nAnswer:",
        "input_variables": ["question"]
    }
}

chain = load_chain_from_config(config)
result = chain.invoke({"question": "What is AI?"})
</syntaxhighlight>

=== Load Chain with Retriever ===
<syntaxhighlight lang="python">
from langchain_classic.chains.loading import load_chain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Create retriever
vectorstore = FAISS.load_local("my_index", OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Load retrieval QA chain (requires retriever kwarg)
chain = load_chain("retrieval_qa.yaml", retriever=retriever)
result = chain.invoke({"query": "What is the main topic?"})
</syntaxhighlight>

=== Complex Nested Chain ===
<syntaxhighlight lang="yaml">
# map_reduce_qa.yaml
_type: map_reduce_documents_chain
llm_chain:
  _type: llm_chain
  llm:
    _type: openai
    temperature: 0
  prompt:
    template: "Summarize this document:\\n\\n{page_content}"
    input_variables: ["page_content"]
reduce_documents_chain:
  combine_documents_chain:
    _type: stuff_documents_chain
    llm_chain:
      _type: llm_chain
      llm:
        _type: openai
        temperature: 0
      prompt:
        template: "Combine these summaries:\\n\\n{text}"
        input_variables: ["text"]
    document_prompt:
      template: "{page_content}"
      input_variables: ["page_content"]
</syntaxhighlight>

== Deprecation Notice ==

This functionality is deprecated since 0.2.13 and will be removed in 1.0:

<syntaxhighlight lang="python">
@deprecated(
    since="0.2.13",
    message=(
        "This function is deprecated and will be removed in langchain 1.0. "
        "At that point chains must be imported from their respective modules."
    ),
    removal="1.0",
)
def load_chain(path: str | Path, **kwargs: Any) -> Chain:
    # Implementation
</syntaxhighlight>

Users should migrate to direct chain construction:

<syntaxhighlight lang="python">
# Instead of loading from config
# chain = load_chain("config.yaml")

# Construct directly
from langchain_classic.chains import LLMChain
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

chain = LLMChain(
    llm=OpenAI(temperature=0.7),
    prompt=PromptTemplate(
        template="Write about {topic}",
        input_variables=["topic"]
    )
)
</syntaxhighlight>

== Related Pages ==
* [[uses::Implementation:langchain-ai_langchain_Chain]]
* [[uses::Implementation:langchain-ai_langchain_LLMChain]]
* [[uses::Concept:Serialization]]
* [[deprecated::true]]
