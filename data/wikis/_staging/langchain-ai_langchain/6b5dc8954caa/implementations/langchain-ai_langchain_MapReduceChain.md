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

Deprecated `MapReduceChain` class that splits large text into chunks, processes each with an LLM, and combines the results.

=== Description ===

The `MapReduceChain` implements the map-reduce pattern for processing documents that exceed LLM context limits. It splits input text using a `TextSplitter`, applies a prompt to each chunk (map phase), then combines the results using a reduce chain. This enables summarization, analysis, or transformation of arbitrarily long documents.

=== Usage ===

This class is deprecated since 0.2.13. For new code, use LangGraph's map-reduce patterns which provide better control over the reduction process and support streaming.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/chains/mapreduce.py libs/langchain/langchain_classic/chains/mapreduce.py]
* '''Lines:''' 1-117

=== Signature ===
<syntaxhighlight lang="python">
@deprecated(since="0.2.13", removal="1.0")
class MapReduceChain(Chain):
    """Map-reduce chain for processing large documents.

    Attributes:
        combine_documents_chain: Chain to combine document chunks.
        text_splitter: Splitter for breaking input into chunks.
        input_key: Key for input text (default: "input_text").
        output_key: Key for output text (default: "output_text").
    """

    combine_documents_chain: BaseCombineDocumentsChain
    text_splitter: TextSplitter
    input_key: str = "input_text"
    output_key: str = "output_text"

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
        """Construct a map-reduce chain from parameters."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Legacy (deprecated)
from langchain_classic.chains import MapReduceChain
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| input_text || str || Yes || Large text to process
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| output_text || str || Combined result from all chunks
|}

== Usage Examples ==

=== Legacy Summarization (Deprecated) ===
<syntaxhighlight lang="python">
from langchain_classic.chains import MapReduceChain
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAI

# Create text splitter
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

# Create summarization prompt
prompt = PromptTemplate.from_template(
    "Summarize the following text:\n\n{text}\n\nSummary:"
)

# Create map-reduce chain
chain = MapReduceChain.from_params(
    llm=OpenAI(),
    prompt=prompt,
    text_splitter=text_splitter,
)

# Process large document
long_document = "..." # Very long text
result = chain.invoke({"input_text": long_document})
print(result["output_text"])
</syntaxhighlight>

=== Modern LangGraph Alternative ===
<syntaxhighlight lang="python">
# Modern approach using LangGraph
from langgraph.graph import StateGraph
from langchain_text_splitters import CharacterTextSplitter

# Define state with documents
class MapReduceState(TypedDict):
    contents: list[str]
    summaries: list[str]
    final_summary: str

# Create graph with map and reduce nodes
# See LangGraph map-reduce guide for full implementation
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
* [[uses_heuristic::Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic]]

