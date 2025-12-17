---
title: MapReduceChain
type: implementation
project: langchain-ai/langchain
file: libs/langchain/langchain_classic/chains/mapreduce.py
deprecated: true
deprecation_version: 0.2.13
removal_version: 1.0
migration_guide: https://python.langchain.com/docs/versions/migrating_chains/map_reduce_chain/
category: document_processing
---

= MapReduceChain Implementation =

== Overview ==

'''MapReduceChain''' implements the classic Map-Reduce pattern for processing large documents. It splits a document into smaller chunks, processes each chunk with an LLM (map step), and then combines the results into a final output (reduce step).

'''Status:''' This class is deprecated as of version 0.2.13 and will be removed in version 1.0. Users should migrate to LangGraph-based implementations.

== Code Reference ==

=== Source Location ===
<syntaxhighlight lang="text">
File: /tmp/praxium_repo_wjjl6pl8/libs/langchain/langchain_classic/chains/mapreduce.py
Lines: 1-117
Package: langchain-classic
</syntaxhighlight>

=== Class Signature ===
<syntaxhighlight lang="python">
@deprecated(
    since="0.2.13",
    removal="1.0",
    message=(
        "Refer to migration guide here for a recommended implementation using "
        "LangGraph: https://python.langchain.com/docs/versions/migrating_chains/map_reduce_chain/"
        ". See also LangGraph guides for map-reduce: "
        "https://langchain-ai.github.io/langgraph/how-tos/map-reduce/."
    ),
)
class MapReduceChain(Chain):
    """Map-reduce chain."""
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from langchain_classic.chains.mapreduce import MapReduceChain
</syntaxhighlight>

== Class Attributes ==

{| class="wikitable"
|+ Required Attributes
! Attribute !! Type !! Description
|-
| combine_documents_chain || <code>BaseCombineDocumentsChain</code> || Chain used to combine document chunks
|-
| text_splitter || <code>TextSplitter</code> || Text splitter for chunking the input
|}

{| class="wikitable"
|+ Optional Attributes
! Attribute !! Type !! Default !! Description
|-
| input_key || <code>str</code> || "input_text" || Key for input text in the inputs dictionary
|-
| output_key || <code>str</code> || "output_text" || Key for output in the results dictionary
|}

== Input/Output Contract ==

{| class="wikitable"
|+ Input Schema
! Key !! Type !! Description
|-
| <code>input_key</code> (default: "input_text") || <code>str</code> || Large text document to process
|-
| Additional keys || <code>Any</code> || Passed through to combine_documents_chain
|}

{| class="wikitable"
|+ Output Schema
! Key !! Type !! Description
|-
| <code>output_key</code> (default: "output_text") || <code>str</code> || Final combined result from reduce step
|}

== Factory Method ==

=== from_params ===

'''Class Method:''' Convenient constructor that builds the complete map-reduce pipeline.

<syntaxhighlight lang="python">
@classmethod
def from_params(
    cls,
    llm: BaseLanguageModel,
    prompt: BasePromptTemplate,
    text_splitter: TextSplitter,
    callbacks: Callbacks = None,
    combine_chain_kwargs: Mapping[str, Any] | None = None,
    reduce_chain_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> MapReduceChain:
    """Construct a map-reduce chain that uses the chain for map and reduce."""
</syntaxhighlight>

{| class="wikitable"
|+ Parameters
! Parameter !! Type !! Description
|-
| llm || <code>BaseLanguageModel</code> || Language model for both map and reduce steps
|-
| prompt || <code>BasePromptTemplate</code> || Prompt template for processing chunks
|-
| text_splitter || <code>TextSplitter</code> || Splitter to chunk the input text
|-
| callbacks || <code>Callbacks</code> || Optional callbacks for observability
|-
| combine_chain_kwargs || <code>Mapping[str, Any]</code> || Arguments for the combine chain
|-
| reduce_chain_kwargs || <code>Mapping[str, Any]</code> || Arguments for the reduce chain
|-
| **kwargs || <code>Any</code> || Additional arguments for MapReduceChain constructor
|}

== Implementation Details ==

=== Processing Flow ===

<syntaxhighlight lang="python">
def _call(
    self,
    inputs: dict[str, str],
    run_manager: CallbackManagerForChainRun | None = None,
) -> dict[str, str]:
    # 1. Extract and split the input text
    doc_text = inputs.pop(self.input_key)
    texts = self.text_splitter.split_text(doc_text)

    # 2. Convert to Document objects
    docs = [Document(page_content=text) for text in texts]

    # 3. Pass to combine_documents_chain (handles map-reduce)
    _inputs = {
        **inputs,
        self.combine_documents_chain.input_key: docs,
    }
    outputs = self.combine_documents_chain.run(
        _inputs,
        callbacks=_run_manager.get_child(),
    )

    # 4. Return final result
    return {self.output_key: outputs}
</syntaxhighlight>

{| class="wikitable"
|+ Processing Steps
! Step !! Action !! Component
|-
| 1 || Split input text into chunks || <code>TextSplitter</code>
|-
| 2 || Convert chunks to Documents || <code>Document</code> creation
|-
| 3 || Map: Process each chunk || <code>LLMChain</code> (within MapReduceDocumentsChain)
|-
| 4 || Reduce: Combine chunk results || <code>ReduceDocumentsChain</code>
|-
| 5 || Final combination || <code>StuffDocumentsChain</code>
|}

=== Internal Chain Structure ===

The <code>from_params</code> method builds a nested chain structure:

<syntaxhighlight lang="python">
# Map chain: processes each chunk
llm_chain = LLMChain(llm=llm, prompt=prompt, callbacks=callbacks)

# Final combine step
stuff_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    callbacks=callbacks,
    **(reduce_chain_kwargs or {})
)

# Reduce documents chain
reduce_documents_chain = ReduceDocumentsChain(
    combine_documents_chain=stuff_chain,
)

# Full map-reduce chain
combine_documents_chain = MapReduceDocumentsChain(
    llm_chain=llm_chain,
    reduce_documents_chain=reduce_documents_chain,
    callbacks=callbacks,
    **(combine_chain_kwargs or {})
)
</syntaxhighlight>

== Usage Examples ==

=== Basic Document Summarization ===

<syntaxhighlight lang="python">
from langchain_classic.chains.mapreduce import MapReduceChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter

# Define prompt for processing chunks
prompt = PromptTemplate(
    template="Summarize the following text:\n\n{text}",
    input_variables=["text"]
)

# Create text splitter
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

# Build map-reduce chain
llm = ChatOpenAI(temperature=0)
chain = MapReduceChain.from_params(
    llm=llm,
    prompt=prompt,
    text_splitter=text_splitter
)

# Process long document
long_document = "..." * 10000  # Very long text

result = chain.invoke({
    "input_text": long_document
})

print(result["output_text"])  # Final summary
</syntaxhighlight>

=== Custom Input/Output Keys ===

<syntaxhighlight lang="python">
from langchain_classic.chains.mapreduce import MapReduceChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

prompt = PromptTemplate(
    template="Extract key points from:\n\n{text}",
    input_variables=["text"]
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

llm = ChatOpenAI()

# Custom keys
chain = MapReduceChain.from_params(
    llm=llm,
    prompt=prompt,
    text_splitter=text_splitter,
    input_key="document",
    output_key="key_points"
)

result = chain.invoke({
    "document": "Long document text..."
})

print(result["key_points"])
</syntaxhighlight>

=== With Custom Reduce Chain ===

<syntaxhighlight lang="python">
from langchain_classic.chains.mapreduce import MapReduceChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import TokenTextSplitter

# Map prompt (for chunks)
map_prompt = PromptTemplate(
    template="List main topics in:\n\n{text}",
    input_variables=["text"]
)

text_splitter = TokenTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

llm = ChatOpenAI(temperature=0)

# Custom reduce chain configuration
chain = MapReduceChain.from_params(
    llm=llm,
    prompt=map_prompt,
    text_splitter=text_splitter,
    reduce_chain_kwargs={
        "document_variable_name": "topics"
    }
)

result = chain.invoke({
    "input_text": "Research paper text..."
})

print(result["output_text"])
</syntaxhighlight>

== Configuration ==

<syntaxhighlight lang="python">
model_config = ConfigDict(
    arbitrary_types_allowed=True,
    extra="forbid",
)
</syntaxhighlight>

* '''arbitrary_types_allowed:''' Allows TextSplitter and other non-Pydantic types
* '''extra="forbid":''' Prevents unexpected attributes

== Use Cases ==

=== Document Summarization ===
* Summarize long articles, papers, or reports
* Extract key insights from lengthy documents
* Create executive summaries

=== Information Extraction ===
* Extract entities, dates, or facts from large texts
* Identify patterns across document sections
* Build structured data from unstructured text

=== Content Analysis ===
* Sentiment analysis on long documents
* Topic classification
* Content quality assessment

=== Question Answering ===
* Answer questions about lengthy documents
* Extract specific information across chapters
* Build knowledge bases from large corpora

== Performance Considerations ==

{| class="wikitable"
|+ Resource Usage
! Aspect !! Consideration !! Notes
|-
| LLM Calls || N + log(N) || N map calls + logarithmic reduce calls
|-
| Parallelization || Sequential by default || Map calls happen sequentially
|-
| Memory || Linear in chunk count || All chunks held in memory
|-
| Latency || Sum of all LLM calls || Can be significant for large docs
|}

=== Optimization Strategies ===

* '''Adjust chunk size:''' Balance between context and number of calls
* '''Use faster models:''' Consider GPT-3.5 for map step, GPT-4 for reduce
* '''Enable caching:''' Cache results for repeated processing
* '''Migrate to LangGraph:''' Use parallel execution in modern implementation

== Limitations ==

=== Deprecated Status ===
* No new features will be added
* Limited community support
* Will be removed in LangChain 1.0

=== Technical Limitations ===
* Sequential execution (no parallel map)
* All chunks must fit in memory
* No streaming support
* Limited error recovery

=== Design Limitations ===
* Rigid chain structure
* Difficult to customize intermediate steps
* No built-in progress tracking
* Limited observability

== Migration Path ==

=== LangGraph Alternative ===

<syntaxhighlight lang="python">
# Modern approach using LangGraph
from langgraph.graph import StateGraph

# Define map-reduce workflow with parallel execution
# See: https://langchain-ai.github.io/langgraph/how-tos/map-reduce/

# Benefits:
# - Parallel map execution
# - Better error handling
# - Streaming support
# - More flexible control flow
</syntaxhighlight>

=== Migration Checklist ===

* Review migration guide at official docs
* Test with small documents first
* Consider using LCEL for simpler cases
* Implement proper error handling in new code
* Add progress tracking for user feedback

== Related Components ==

* '''MapReduceDocumentsChain''' (langchain-classic) - Internal chain for map-reduce logic
* '''ReduceDocumentsChain''' (langchain-classic) - Handles reduction step
* '''StuffDocumentsChain''' (langchain-classic) - Final combination
* '''LLMChain''' (langchain-classic) - Basic LLM chain for map step
* '''TextSplitter''' (langchain-text-splitters) - Document chunking
* '''Document''' (langchain-core) - Document representation

== See Also ==

* [[langchain-ai_langchain_MapReduceDocumentsChain|MapReduceDocumentsChain]] - Core map-reduce implementation
* [[langchain-ai_langchain_StuffDocumentsChain|StuffDocumentsChain]] - Simple document combination
* [[langchain-ai_langchain_LLMChain|LLMChain]] - Basic LLM chain
* [[langchain-ai_langchain_TextSplitter|TextSplitter]] - Text chunking utilities
* LangGraph map-reduce guide: https://langchain-ai.github.io/langgraph/how-tos/map-reduce/
* Migration guide for chains
