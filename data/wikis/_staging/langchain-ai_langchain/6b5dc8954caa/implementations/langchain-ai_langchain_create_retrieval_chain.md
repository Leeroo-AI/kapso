{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain RAG|https://python.langchain.com/docs/tutorials/rag/]]
|-
! Domains
| [[domain::RAG]], [[domain::Retrieval]], [[domain::Chains]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Factory function `create_retrieval_chain` that creates an LCEL chain combining document retrieval with answer generation.

=== Description ===

The `create_retrieval_chain` function builds a complete RAG (Retrieval-Augmented Generation) pipeline by composing a retriever with a document combination chain. It retrieves documents based on the input query, passes them as context to the answer generation chain, and returns both the retrieved context and the generated answer.

=== Usage ===

Use this function to build standard RAG applications where you want to answer questions based on a document collection. It handles the wiring between retrieval and generation, making it easy to create Q&A systems over custom data.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/chains/retrieval.py libs/langchain/langchain_classic/chains/retrieval.py]
* '''Lines:''' 1-68

=== Signature ===
<syntaxhighlight lang="python">
def create_retrieval_chain(
    retriever: BaseRetriever | Runnable[dict, RetrieverOutput],
    combine_docs_chain: Runnable[dict[str, Any], str],
) -> Runnable:
    """Create retrieval chain that retrieves documents and then passes them on.

    Args:
        retriever: Retriever that returns documents from a query. If BaseRetriever,
            expects 'input' key. Otherwise, receives full input dict.
        combine_docs_chain: Chain that takes inputs including 'context' (documents)
            and produces a string answer.

    Returns:
        LCEL Runnable returning dict with 'context' and 'answer' keys.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.chains import create_retrieval_chain
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| input || str || Yes || User query to retrieve documents for and answer
|-
| chat_history || list || No || Previous conversation (defaults to [])
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| context || list[Document] || Retrieved documents
|-
| answer || str || Generated answer based on context
|-
| input || str || Original input (passed through)
|}

== Usage Examples ==

=== Basic RAG Chain ===
<syntaxhighlight lang="python">
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

# Create retriever
vectorstore = FAISS.load_local("my_docs")
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Create answer generation chain
llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on the following context:\n\n{context}"),
    ("human", "{input}"),
])
combine_docs_chain = create_stuff_documents_chain(llm, prompt)

# Create full RAG chain
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Invoke
response = rag_chain.invoke({"input": "What is LangChain?"})
print(response["answer"])
print(f"Based on {len(response['context'])} documents")
</syntaxhighlight>

=== With History-Aware Retriever ===
<syntaxhighlight lang="python">
from langchain_classic.chains import (
    create_retrieval_chain,
    create_history_aware_retriever,
)

# Create history-aware retriever for conversational RAG
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Reformulate the question to be standalone."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_prompt
)

# Create conversational RAG chain
conversational_rag = create_retrieval_chain(
    history_retriever,
    combine_docs_chain,
)

# Use with chat history
chat_history = [
    HumanMessage("What is LangChain?"),
    AIMessage("LangChain is a framework for building LLM applications."),
]
response = conversational_rag.invoke({
    "input": "What are its main features?",
    "chat_history": chat_history,
})
</syntaxhighlight>

=== Streaming Responses ===
<syntaxhighlight lang="python">
# Stream the answer generation
async for chunk in rag_chain.astream({"input": "Explain RAG"}):
    if "answer" in chunk:
        print(chunk["answer"], end="", flush=True)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
* [[related_to::Implementation:langchain-ai_langchain_create_stuff_documents_chain]]
* [[related_to::Implementation:langchain-ai_langchain_create_history_aware_retriever]]

