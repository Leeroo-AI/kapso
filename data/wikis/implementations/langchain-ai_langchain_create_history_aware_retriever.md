{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain RAG|https://python.langchain.com/docs/tutorials/rag/]]
|-
! Domains
| [[domain::RAG]], [[domain::Retrieval]], [[domain::Conversational_AI]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Factory function `create_history_aware_retriever` that creates an LCEL chain which reformulates queries using chat history before retrieval.

=== Description ===

The `create_history_aware_retriever` function builds a retrieval chain that handles conversational context. When chat history is present, it uses an LLM to reformulate the user's input into a standalone search query that incorporates context from previous messages. When there's no history, it passes the input directly to the retriever. This enables natural multi-turn conversations over document collections.

=== Usage ===

Use this function when building conversational RAG applications where follow-up questions reference previous context (e.g., "What about the second one?" or "Can you elaborate on that?"). The returned chain can be composed with document combination chains for full Q&A pipelines.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/chains/history_aware_retriever.py libs/langchain/langchain_classic/chains/history_aware_retriever.py]
* '''Lines:''' 1-68

=== Signature ===
<syntaxhighlight lang="python">
def create_history_aware_retriever(
    llm: LanguageModelLike,
    retriever: RetrieverLike,
    prompt: BasePromptTemplate,
) -> RetrieverOutputLike:
    """Create a chain that takes conversation history and returns documents.

    Args:
        llm: Language model for generating search queries from chat history.
        retriever: Retriever that takes a string and returns documents.
        prompt: Prompt for reformulating queries (must have 'input' variable).

    Returns:
        An LCEL Runnable that accepts 'input' and optional 'chat_history',
        returning a list of Documents.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.chains import create_history_aware_retriever
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| input || str || Yes || User's current query
|-
| chat_history || list[BaseMessage] || No || Previous conversation messages
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return || list[Document] || Retrieved documents relevant to the (reformulated) query
|}

== Usage Examples ==

=== Basic Conversational Retriever ===
<syntaxhighlight lang="python">
from langchain_classic.chains import create_history_aware_retriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS

# Create retriever from vector store
vectorstore = FAISS.load_local("my_docs")
retriever = vectorstore.as_retriever()

# Create prompt for query reformulation
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and latest user question, "
               "formulate a standalone question that can be understood "
               "without the chat history. Do NOT answer the question, "
               "just reformulate it if needed, otherwise return as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create the history-aware retriever
llm = ChatOpenAI(model="gpt-4")
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Use with chat history
from langchain_core.messages import HumanMessage, AIMessage

chat_history = [
    HumanMessage(content="What is LangChain?"),
    AIMessage(content="LangChain is a framework for building LLM applications."),
]

# "it" refers to LangChain from chat history
docs = history_aware_retriever.invoke({
    "input": "What are its main components?",
    "chat_history": chat_history,
})
</syntaxhighlight>

=== Full RAG Chain with History ===
<syntaxhighlight lang="python">
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Create answer generation chain
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on the context:\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Combine into full RAG chain
rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain,
)

# Invoke with history
response = rag_chain.invoke({
    "input": "What are its main components?",
    "chat_history": chat_history,
})
print(response["answer"])
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
* [[related_to::Implementation:langchain-ai_langchain_create_retrieval_chain]]

