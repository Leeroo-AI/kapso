{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Docs|https://python.langchain.com/docs/tutorials/rag/]]
|-
! Domains
| [[domain::RAG]], [[domain::Document_Processing]], [[domain::Chains]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Factory function `create_stuff_documents_chain` and deprecated class `StuffDocumentsChain` that combine documents by concatenating them into a single context string.

=== Description ===

The "stuff" strategy is the simplest document combination approach: format each document using a template, join them with a separator, and pass the combined string to an LLM. `create_stuff_documents_chain` returns an LCEL Runnable, while `StuffDocumentsChain` is the deprecated Chain-based equivalent. This approach works well when documents fit within the context window.

=== Usage ===

Use `create_stuff_documents_chain` when building RAG applications where the retrieved documents fit within the model's context window. For documents exceeding context limits, use map-reduce or refine strategies instead.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/chains/combine_documents/stuff.py libs/langchain/langchain_classic/chains/combine_documents/stuff.py]
* '''Lines:''' 1-291

=== Signature ===
<syntaxhighlight lang="python">
def create_stuff_documents_chain(
    llm: LanguageModelLike,
    prompt: BasePromptTemplate,
    *,
    output_parser: BaseOutputParser | None = None,
    document_prompt: BasePromptTemplate | None = None,
    document_separator: str = "\n\n",
    document_variable_name: str = "context",
) -> Runnable[dict[str, Any], Any]:
    """Create a chain for passing a list of Documents to a model.

    Args:
        llm: Language model to use.
        prompt: Prompt template (must have 'context' variable by default).
        output_parser: Parser for LLM output (default: StrOutputParser).
        document_prompt: Template for formatting each document.
        document_separator: Separator between documents (default: "\\n\\n").
        document_variable_name: Variable name for documents in prompt.

    Returns:
        LCEL Runnable accepting dict with 'context' key containing list[Document].
    """


@deprecated(since="0.2.13", removal="1.0")
class StuffDocumentsChain(BaseCombineDocumentsChain):
    """Deprecated Chain that stuffs documents into context.

    Attributes:
        llm_chain: LLMChain to call with formatted documents.
        document_prompt: Template for each document.
        document_variable_name: Variable name in prompt for documents.
        document_separator: String separator between documents.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Legacy (deprecated)
from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| context || list[Document] || Yes || Documents to combine and pass to LLM
|-
| (other) || Any || No || Additional variables from prompt template
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return || str or Any || LLM response (type depends on output_parser)
|}

== Usage Examples ==

=== Basic Stuff Chain ===
<syntaxhighlight lang="python">
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on the following context:\n\n{context}"),
    ("human", "{question}"),
])

# Create chain
llm = ChatOpenAI(model="gpt-4")
chain = create_stuff_documents_chain(llm, prompt)

# Prepare documents
docs = [
    Document(page_content="LangChain is a framework for LLM applications."),
    Document(page_content="It supports Python and JavaScript."),
]

# Invoke
answer = chain.invoke({
    "context": docs,
    "question": "What is LangChain?",
})
print(answer)
</syntaxhighlight>

=== With Custom Document Formatting ===
<syntaxhighlight lang="python">
from langchain_core.prompts import PromptTemplate

# Custom format for each document
document_prompt = PromptTemplate.from_template(
    "Source: {source}\nContent: {page_content}"
)

chain = create_stuff_documents_chain(
    llm,
    prompt,
    document_prompt=document_prompt,
    document_separator="\n---\n",
)

# Documents with metadata
docs = [
    Document(page_content="Content 1", metadata={"source": "doc1.pdf"}),
    Document(page_content="Content 2", metadata={"source": "doc2.pdf"}),
]

answer = chain.invoke({"context": docs, "question": "Summarize"})
</syntaxhighlight>

=== Full RAG Pipeline ===
<syntaxhighlight lang="python">
from langchain_classic.chains import create_retrieval_chain

# Create stuff chain for answer generation
combine_chain = create_stuff_documents_chain(llm, prompt)

# Create full RAG chain with retriever
rag_chain = create_retrieval_chain(retriever, combine_chain)

response = rag_chain.invoke({"input": "What is LangChain?"})
print(response["answer"])
print(f"Based on {len(response['context'])} documents")
</syntaxhighlight>

=== Legacy StuffDocumentsChain (Deprecated) ===
<syntaxhighlight lang="python">
# Old way - deprecated
from langchain_classic.chains import StuffDocumentsChain, LLMChain
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Summarize: {context}")
llm_chain = LLMChain(llm=OpenAI(), prompt=prompt)

chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
* [[related_to::Implementation:langchain-ai_langchain_create_retrieval_chain]]

