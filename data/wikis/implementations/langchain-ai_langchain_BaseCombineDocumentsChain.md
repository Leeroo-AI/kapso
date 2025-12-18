{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Chains]], [[domain::Document_Processing]], [[domain::Abstractions]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Abstract base class `BaseCombineDocumentsChain` that defines the interface for chains that combine multiple documents into a single output.

=== Description ===

The `BaseCombineDocumentsChain` provides a standard interface for document combination strategies (stuff, map-reduce, refine, etc.). It defines abstract methods `combine_docs` and `acombine_docs` that subclasses implement. The class also includes `prompt_length` for checking if documents fit within context limits, and handles the common input/output key patterns.

=== Usage ===

Subclass this when implementing new document combination strategies. For built-in strategies, use `create_stuff_documents_chain`, `MapReduceDocumentsChain`, or `RefineDocumentsChain`.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/chains/combine_documents/base.py libs/langchain/langchain_classic/chains/combine_documents/base.py]
* '''Lines:''' 1-278

=== Signature ===
<syntaxhighlight lang="python">
class BaseCombineDocumentsChain(Chain, ABC):
    """Base interface for chains combining documents.

    Attributes:
        input_key: Key for input documents (default: "input_documents").
        output_key: Key for output text (default: "output_text").
    """

    input_key: str = "input_documents"
    output_key: str = "output_text"

    def prompt_length(self, docs: list[Document], **kwargs: Any) -> int | None:
        """Return prompt length in tokens, or None if not applicable."""

    @abstractmethod
    def combine_docs(self, docs: list[Document], **kwargs: Any) -> tuple[str, dict]:
        """Combine documents into a single string.

        Returns:
            Tuple of (output_string, extra_return_dict).
        """

    @abstractmethod
    async def acombine_docs(
        self,
        docs: list[Document],
        **kwargs: Any,
    ) -> tuple[str, dict]:
        """Async combine documents into a single string."""


@deprecated(since="0.2.7", removal="1.0")
class AnalyzeDocumentChain(Chain):
    """Deprecated chain that splits a document then analyzes pieces.

    Splits input text, passes chunks to a CombineDocumentsChain.
    """

    input_key: str = "input_document"
    text_splitter: TextSplitter
    combine_docs_chain: BaseCombineDocumentsChain
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.chains.combine_documents.base import BaseCombineDocumentsChain
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| input_documents || list[Document] || Yes || Documents to combine
|-
| (other) || Any || No || Additional inputs passed to combine_docs
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| output_text || str || Combined document output
|-
| (extra) || Any || Additional outputs from subclass implementation
|}

== Usage Examples ==

=== Custom Combine Strategy ===
<syntaxhighlight lang="python">
from langchain_classic.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain_core.documents import Document
from typing import Any

class BulletPointCombineChain(BaseCombineDocumentsChain):
    """Combine documents as bullet points."""

    def combine_docs(
        self,
        docs: list[Document],
        **kwargs: Any,
    ) -> tuple[str, dict]:
        # Format each document as a bullet point
        bullets = [f"• {doc.page_content}" for doc in docs]
        combined = "\n".join(bullets)
        return combined, {}

    async def acombine_docs(
        self,
        docs: list[Document],
        **kwargs: Any,
    ) -> tuple[str, dict]:
        return self.combine_docs(docs, **kwargs)


# Use the custom chain
chain = BulletPointCombineChain()
result = chain.invoke({
    "input_documents": [
        Document(page_content="First point"),
        Document(page_content="Second point"),
    ]
})
print(result["output_text"])
# • First point
# • Second point
</syntaxhighlight>

=== Checking Prompt Length ===
<syntaxhighlight lang="python">
from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain

# Check if documents fit in context
chain = StuffDocumentsChain(...)
docs = [Document(page_content="...") for _ in range(100)]

# Returns token count (or None if not supported)
length = chain.prompt_length(docs)
if length and length > 4000:
    print("Documents too long, use map-reduce instead")
</syntaxhighlight>

=== AnalyzeDocumentChain (Deprecated) ===
<syntaxhighlight lang="python">
# Old way - splits one document, then combines chunks
from langchain_classic.chains.combine_documents.base import AnalyzeDocumentChain
from langchain_text_splitters import CharacterTextSplitter

chain = AnalyzeDocumentChain(
    text_splitter=CharacterTextSplitter(chunk_size=1000),
    combine_docs_chain=stuff_chain,
)

result = chain.invoke({"input_document": very_long_text})

# Modern equivalent using LCEL
from langchain_core.runnables import RunnableLambda

split_text = RunnableLambda(lambda x: text_splitter.create_documents([x]))
modern_chain = split_text | stuff_chain
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
* [[uses_heuristic::Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic]]

