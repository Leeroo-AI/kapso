{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Chains]], [[domain::Summarization]], [[domain::Document_Processing]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Factory function `load_summarize_chain` that creates document summarization chains using different strategies (stuff, map-reduce, refine).

=== Description ===

The `load_summarize_chain` function is a convenience factory for creating summarization chains. It supports three strategies: "stuff" (combine all documents), "map_reduce" (summarize each then combine), and "refine" (iteratively refine summary with each document). Each strategy uses appropriate default prompts optimized for summarization.

=== Usage ===

Use this function when building document summarization pipelines. Choose the strategy based on document size and desired quality: "stuff" for small documents, "map_reduce" for large documents with parallelization, "refine" for highest quality sequential processing.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/chains/summarize/chain.py libs/langchain/langchain_classic/chains/summarize/chain.py]
* '''Lines:''' 1-226

=== Signature ===
<syntaxhighlight lang="python">
def load_summarize_chain(
    llm: BaseLanguageModel,
    chain_type: str = "stuff",
    verbose: bool | None = None,
    **kwargs: Any,
) -> BaseCombineDocumentsChain:
    """Load summarizing chain.

    Args:
        llm: Language Model to use in the chain.
        chain_type: Type of chain - "stuff", "map_reduce", or "refine".
        verbose: Whether to run in verbose mode.
        **kwargs: Additional arguments for specific chain types.

    Returns:
        A chain to use for summarizing documents.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.chains.summarize import load_summarize_chain
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| llm || BaseLanguageModel || Yes || Language model for summarization
|-
| chain_type || str || No || "stuff", "map_reduce", or "refine" (default: "stuff")
|-
| verbose || bool || No || Enable verbose output
|}

=== Chain Type Specific Arguments ===
{| class="wikitable"
|-
! Chain Type !! Arguments
|-
| stuff || prompt, document_variable_name
|-
| map_reduce || map_prompt, combine_prompt, reduce_llm, collapse_llm, token_max
|-
| refine || question_prompt, refine_prompt, refine_llm
|}

== Usage Examples ==

=== Basic Stuff Summarization ===
<syntaxhighlight lang="python">
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_openai import OpenAI
from langchain_core.documents import Document

llm = OpenAI(temperature=0)
chain = load_summarize_chain(llm, chain_type="stuff")

docs = [
    Document(page_content="Long article text here..."),
    Document(page_content="Another section..."),
]

summary = chain.invoke({"input_documents": docs})
print(summary["output_text"])
</syntaxhighlight>

=== Map-Reduce for Long Documents ===
<syntaxhighlight lang="python">
# Use map-reduce when documents exceed context limit
chain = load_summarize_chain(
    llm,
    chain_type="map_reduce",
    token_max=4000,
    verbose=True,
)

# Each doc is summarized independently (map)
# Then summaries are combined (reduce)
summary = chain.invoke({"input_documents": large_docs})
</syntaxhighlight>

=== Refine for High Quality ===
<syntaxhighlight lang="python">
# Refine builds up summary document by document
chain = load_summarize_chain(
    llm,
    chain_type="refine",
    verbose=True,
)

# First doc creates initial summary
# Each subsequent doc refines the summary
summary = chain.invoke({"input_documents": docs})
</syntaxhighlight>

=== Custom Prompts ===
<syntaxhighlight lang="python">
from langchain_core.prompts import PromptTemplate

# Custom stuff prompt
custom_prompt = PromptTemplate.from_template(
    "Write a concise summary of the following:\n\n{text}\n\nCONCISE SUMMARY:"
)

chain = load_summarize_chain(
    llm,
    chain_type="stuff",
    prompt=custom_prompt,
)

# Custom map-reduce prompts
map_prompt = PromptTemplate.from_template("Summarize:\n{text}")
combine_prompt = PromptTemplate.from_template("Combine summaries:\n{text}")

chain = load_summarize_chain(
    llm,
    chain_type="map_reduce",
    map_prompt=map_prompt,
    combine_prompt=combine_prompt,
)
</syntaxhighlight>

=== Full Pipeline with Document Loading ===
<syntaxhighlight lang="python">
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# Load and split document
loader = PyPDFLoader("report.pdf")
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = splitter.split_documents(documents)

# Summarize
chain = load_summarize_chain(llm, chain_type="map_reduce")
summary = chain.invoke({"input_documents": split_docs})
print(summary["output_text"])
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]

