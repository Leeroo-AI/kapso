---
title: create_retrieval_chain
type: implementation
project: langchain-ai/langchain
file: libs/langchain/langchain_classic/chains/retrieval.py
category: rag
---

= create_retrieval_chain Implementation =

== Overview ==

'''create_retrieval_chain''' is a function that creates a complete RAG (Retrieval-Augmented Generation) chain by combining a retriever with a document combination chain. It retrieves relevant documents and passes them to an LLM for generating answers, making it the standard way to build question-answering systems over custom documents.

This is the modern, LCEL-based approach to building retrieval chains, replacing older chain patterns like RetrievalQA.

== Code Reference ==

=== Source Location ===
<syntaxhighlight lang="text">
File: /tmp/praxium_repo_wjjl6pl8/libs/langchain/langchain_classic/chains/retrieval.py
Lines: 12-68
Package: langchain-classic
</syntaxhighlight>

=== Function Signature ===
<syntaxhighlight lang="python">
def create_retrieval_chain(
    retriever: BaseRetriever | Runnable[dict, RetrieverOutput],
    combine_docs_chain: Runnable[dict[str, Any], str],
) -> Runnable:
    """Create retrieval chain that retrieves documents and then passes them on."""
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from langchain_classic.chains import create_retrieval_chain
</syntaxhighlight>

== Parameters ==

{| class="wikitable"
|+ Input Parameters
! Parameter !! Type !! Required !! Description
|-
| retriever || <code>BaseRetriever</code> or <code>Runnable[dict, RetrieverOutput]</code> || Yes || Retriever or runnable that returns documents
|-
| combine_docs_chain || <code>Runnable[dict[str, Any], str]</code> || Yes || Chain that takes documents and produces answer
|}

=== Retriever Parameter ===

Two types accepted:

{| class="wikitable"
|+ Retriever Types
! Type !! Expected Input !! Behavior
|-
| <code>BaseRetriever</code> || Uses <code>input</code> key || Automatically extracts <code>input</code> key and passes to retriever
|-
| <code>Runnable</code> || Uses entire input dict || Passes full input dictionary to runnable
|}

=== combine_docs_chain Parameter ===

Expected to accept:
* All original inputs from the retrieval chain
* <code>context</code> key with retrieved documents
* <code>chat_history</code> key (empty list if not provided)

== Return Value ==

{| class="wikitable"
|+ Output
! Type !! Description
|-
| <code>Runnable</code> || LCEL chain that returns dictionary with <code>context</code> and <code>answer</code> keys
|}

== Output Schema ==

{| class="wikitable"
|+ Return Dictionary Keys
! Key !! Type !! Description
|-
| context || <code>list[Document]</code> || Retrieved documents used to generate answer
|-
| answer || <code>str</code> || Generated answer from combine_docs_chain
|-
| (original keys) || Various || All input keys are preserved in output
|}

== Implementation Details ==

The function builds an LCEL chain using <code>RunnablePassthrough.assign</code>:

<syntaxhighlight lang="python">
# Determine retrieval strategy based on retriever type
if not isinstance(retriever, BaseRetriever):
    # Custom Runnable: pass entire input dict
    retrieval_docs = retriever
else:
    # BaseRetriever: extract "input" key
    retrieval_docs = (lambda x: x["input"]) | retriever

# Build chain: retrieve docs, then generate answer
return (
    RunnablePassthrough.assign(
        context=retrieval_docs.with_config(run_name="retrieve_documents"),
    ).assign(answer=combine_docs_chain)
).with_config(run_name="retrieval_chain")
</syntaxhighlight>

{| class="wikitable"
|+ Chain Execution Flow
! Step !! Action !! Keys Available
|-
| 1 || Pass through original input || <code>input</code>, <code>chat_history</code>, etc.
|-
| 2 || Retrieve documents → <code>context</code> || <code>input</code>, <code>context</code>
|-
| 3 || Generate answer → <code>answer</code> || <code>input</code>, <code>context</code>, <code>answer</code>
|-
| 4 || Return full dictionary || All keys including <code>context</code> and <code>answer</code>
|}

== Usage Examples ==

=== Basic RAG Chain ===

<syntaxhighlight lang="python">
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# Create vector store and retriever
vectorstore = FAISS.from_texts(
    ["LangChain is a framework for LLM apps",
     "RAG combines retrieval with generation",
     "Vector stores enable semantic search"],
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

# Create prompt for QA
prompt = ChatPromptTemplate.from_template("""
Answer the question based on the following context:

{context}

Question: {input}

Answer:
""")

# Create document combination chain
llm = ChatOpenAI()
combine_docs_chain = create_stuff_documents_chain(llm, prompt)

# Create complete RAG chain
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Use the chain
result = retrieval_chain.invoke({"input": "What is RAG?"})

print(result["answer"])
print(f"Used {len(result['context'])} documents")
</syntaxhighlight>

=== With Chat History ===

<syntaxhighlight lang="python">
from langchain_classic.chains import (
    create_retrieval_chain,
    create_history_aware_retriever
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatOpenAI()
retriever = ...  # Your retriever

# Step 1: History-aware retriever
rephrase_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("system", "Reformulate the question based on chat history.")
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, rephrase_prompt
)

# Step 2: QA chain with history
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on this context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)

# Step 3: Complete conversational RAG chain
rag_chain = create_retrieval_chain(
    history_aware_retriever,
    combine_docs_chain
)

# Use with conversation history
result = rag_chain.invoke({
    "input": "What are the benefits?",
    "chat_history": [
        HumanMessage(content="Tell me about RAG"),
        AIMessage(content="RAG is a technique that...")
    ]
})

print(result["answer"])
</syntaxhighlight>

=== Custom Retriever Runnable ===

<syntaxhighlight lang="python">
from langchain_classic.chains import create_retrieval_chain
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

# Custom retriever that uses multiple fields
def custom_retrieve(inputs: dict) -> list[Document]:
    """Custom retrieval logic using query and filters."""
    query = inputs.get("input", "")
    filters = inputs.get("filters", {})

    # Your custom retrieval logic
    docs = your_retrieval_system.search(query, filters=filters)
    return docs

# Wrap as Runnable
custom_retriever = RunnableLambda(custom_retrieve)

# Create chain with custom retriever
retrieval_chain = create_retrieval_chain(
    custom_retriever,
    combine_docs_chain
)

# Use with additional parameters
result = retrieval_chain.invoke({
    "input": "What is AI?",
    "filters": {"category": "technology"}
})
</syntaxhighlight>

=== With Document Metadata Filtering ===

<syntaxhighlight lang="python">
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Create retriever with metadata filters
vectorstore = Chroma.from_documents(
    documents=[
        Document(page_content="...", metadata={"year": 2023}),
        Document(page_content="...", metadata={"year": 2024}),
    ],
    embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5, "filter": {"year": 2024}}
)

# Create RAG chain
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

result = retrieval_chain.invoke({"input": "Recent developments?"})
</syntaxhighlight>

=== Streaming Responses ===

<syntaxhighlight lang="python">
from langchain_classic.chains import create_retrieval_chain

# Create chain (combine_docs_chain should support streaming)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Stream the answer
for chunk in retrieval_chain.stream({"input": "Explain RAG"}):
    if "answer" in chunk:
        print(chunk["answer"], end="", flush=True)
    if "context" in chunk:
        print(f"\nUsed {len(chunk['context'])} documents")
</syntaxhighlight>

=== With Citations ===

<syntaxhighlight lang="python">
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Prompt that includes citations
citation_prompt = ChatPromptTemplate.from_template("""
Answer the question and cite your sources using [1], [2], etc.

Context:
{context}

Question: {input}

Answer with citations:
""")

combine_docs_chain = create_stuff_documents_chain(llm, citation_prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

result = retrieval_chain.invoke({"input": "What is machine learning?"})

# Answer includes citations
print(result["answer"])

# Map citations to source documents
for i, doc in enumerate(result["context"], 1):
    print(f"[{i}] {doc.metadata.get('source', 'Unknown')}")
</syntaxhighlight>

== Integration Patterns ==

=== Full LangChain Hub Example ===

<syntaxhighlight lang="python">
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic import hub
from langchain_openai import ChatOpenAI

# Pull pre-built prompt from hub
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# Build components
model = ChatOpenAI()
retriever = ...  # Your retriever

# Create chains
combine_docs_chain = create_stuff_documents_chain(model, retrieval_qa_chat_prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Use
result = retrieval_chain.invoke({"input": "Your question"})
</syntaxhighlight>

=== With RunnableWithMessageHistory ===

<syntaxhighlight lang="python">
from langchain_classic.chains import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Create RAG chain
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Add message history management
message_history = ChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    retrieval_chain,
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Use with automatic history management
result = chain_with_history.invoke(
    {"input": "What is RAG?"},
    config={"configurable": {"session_id": "user123"}}
)
</syntaxhighlight>

== Design Patterns ==

=== Passthrough with Assignment ===

The function uses <code>RunnablePassthrough.assign</code> to:
1. Preserve all input keys
2. Add <code>context</code> from retrieval
3. Add <code>answer</code> from generation

This ensures all information flows through the chain.

=== Named Steps ===

Each step is named for observability:
* <code>retrieve_documents</code>: Document retrieval step
* <code>retrieval_chain</code>: Overall chain

These names appear in LangSmith traces.

=== Automatic chat_history ===

The docstring mentions that <code>chat_history</code> is provided as <code>[]</code> if not present, enabling conversational retrieval.

== Common Use Cases ==

=== Question Answering ===
* Answer questions about documents
* Build knowledge base assistants
* Customer support bots

=== Document Search ===
* Semantic search with answer generation
* Research assistants
* Document analysis tools

=== Conversational RAG ===
* Multi-turn Q&A
* Context-aware follow-ups
* Chatbots with document grounding

=== Fact Checking ===
* Verify claims against documents
* Source attribution
* Evidence-based answering

== Performance Considerations ==

{| class="wikitable"
|+ Performance Characteristics
! Aspect !! Impact !! Optimization
|-
| Retrieval Latency || 50-200ms || Use faster vector stores, adjust k
|-
| LLM Latency || 500-2000ms || Use faster models, enable streaming
|-
| Token Usage || Variable || Use concise prompts, limit context
|-
| Concurrent Requests || Linear scaling || Use async variants
|}

=== Optimization Tips ===

* '''Limit retrieved documents:''' Reduce <code>k</code> in retriever
* '''Use smaller models:''' GPT-3.5 for simple Q&A
* '''Enable streaming:''' Better UX for long answers
* '''Cache embeddings:''' Reuse embeddings for common queries
* '''Batch processing:''' Use <code>batch</code> method for multiple queries

== Error Handling ==

<syntaxhighlight lang="python">
from langchain_classic.chains import create_retrieval_chain

retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

try:
    result = retrieval_chain.invoke({"input": "Question?"})
    print(result["answer"])

except KeyError as e:
    # Missing required input key
    print(f"Missing input: {e}")

except Exception as e:
    # Retrieval or LLM errors
    print(f"Chain error: {e}")
    # Fallback behavior
</syntaxhighlight>

== Testing ==

<syntaxhighlight lang="python">
from langchain_classic.chains import create_retrieval_chain

# Test with mock retriever
def test_retrieval_chain():
    # Mock retriever
    from langchain_core.runnables import RunnableLambda
    from langchain_core.documents import Document

    mock_retriever = RunnableLambda(
        lambda x: [Document(page_content="Test doc")]
    )

    # Mock combine chain
    mock_combine = RunnableLambda(
        lambda x: f"Answer based on: {x['context']}"
    )

    # Create chain
    chain = create_retrieval_chain(mock_retriever, mock_combine)

    # Test
    result = chain.invoke({"input": "test"})

    assert "context" in result
    assert "answer" in result
    assert len(result["context"]) == 1
</syntaxhighlight>

== Related Components ==

* '''create_history_aware_retriever''' - Adds conversational context
* '''create_stuff_documents_chain''' - Combines documents for LLM
* '''BaseRetriever''' (langchain-core) - Retriever interface
* '''RunnablePassthrough''' (langchain-core) - Passthrough primitive
* '''VectorStore.as_retriever()''' - Create retriever from vector store

== See Also ==

* [[langchain-ai_langchain_create_history_aware_retriever|create_history_aware_retriever]] - Conversational retrieval
* [[langchain-ai_langchain_create_stuff_documents_chain|create_stuff_documents_chain]] - Document combination
* [[langchain-ai_langchain_BaseRetriever|BaseRetriever]] - Retriever interface
* [[langchain-ai_langchain_RunnablePassthrough|RunnablePassthrough]] - LCEL primitive
* LangChain RAG documentation
* LangChain Hub for pre-built prompts
