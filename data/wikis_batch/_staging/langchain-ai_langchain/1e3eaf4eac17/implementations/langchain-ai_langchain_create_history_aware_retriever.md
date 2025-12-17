---
title: create_history_aware_retriever
type: implementation
project: langchain-ai/langchain
file: libs/langchain/langchain_classic/chains/history_aware_retriever.py
category: conversational_rag
---

= create_history_aware_retriever Implementation =

== Overview ==

'''create_history_aware_retriever''' creates a chain that intelligently handles document retrieval in conversational contexts. When chat history exists, it uses an LLM to reformulate the user's query based on the conversation context before retrieving documents. When there's no chat history, it passes the input directly to the retriever.

This is a critical component for building '''Conversational RAG''' (Retrieval-Augmented Generation) systems where follow-up questions need to be understood in the context of previous conversation turns.

== Code Reference ==

=== Source Location ===
<syntaxhighlight lang="text">
File: /tmp/praxium_repo_wjjl6pl8/libs/langchain/langchain_classic/chains/history_aware_retriever.py
Lines: 10-68
Package: langchain-classic
</syntaxhighlight>

=== Function Signature ===
<syntaxhighlight lang="python">
def create_history_aware_retriever(
    llm: LanguageModelLike,
    retriever: RetrieverLike,
    prompt: BasePromptTemplate,
) -> RetrieverOutputLike:
    """Create a chain that takes conversation history and returns documents."""
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from langchain_classic.chains import create_history_aware_retriever
</syntaxhighlight>

== Parameters ==

{| class="wikitable"
|+ Input Parameters
! Parameter !! Type !! Required !! Description
|-
| llm || <code>LanguageModelLike</code> || Yes || Language model to generate search query from chat history
|-
| retriever || <code>RetrieverLike</code> || Yes || Retriever that takes a string and returns Document objects
|-
| prompt || <code>BasePromptTemplate</code> || Yes || Prompt template for generating search query (must include "input" variable)
|}

== Return Value ==

{| class="wikitable"
|+ Output
! Type !! Description
|-
| <code>RetrieverOutputLike</code> || LCEL Runnable that takes input and optional chat_history, returns list of Documents
|}

== Expected Input Format ==

{| class="wikitable"
|+ Runnable Input Schema
! Key !! Type !! Required !! Description
|-
| input || <code>str</code> || Yes || Current user input/question
|-
| chat_history || <code>list</code> or <code>str</code> || No || Previous conversation messages (empty list/string if no history)
|}

== Implementation Logic ==

The function creates a branching chain using '''RunnableBranch''':

<syntaxhighlight lang="python">
retrieve_documents = RunnableBranch(
    (
        # Condition: no chat history exists
        lambda x: not x.get("chat_history", False),
        # Action: pass input directly to retriever
        (lambda x: x["input"]) | retriever,
    ),
    # Default action: use LLM to reformulate query
    prompt | llm | StrOutputParser() | retriever,
).with_config(run_name="chat_retriever_chain")
</syntaxhighlight>

=== Branching Logic ===

{| class="wikitable"
|+ Decision Flow
! Condition !! Action !! Reasoning
|-
| <code>chat_history</code> is empty or missing || Pass <code>input</code> directly to retriever || No context needed; use query as-is
|-
| <code>chat_history</code> exists || Generate search query via LLM then retrieve || Reformulate query considering conversation context
|}

== Usage Examples ==

=== Basic Conversational RAG ===

<syntaxhighlight lang="python">
from langchain_classic.chains import create_history_aware_retriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Create retriever from vector store
vectorstore = FAISS.from_texts(
    ["LangChain is a framework for LLM applications",
     "RAG combines retrieval with generation",
     "Vector stores enable semantic search"],
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

# Define prompt for query reformulation
rephrase_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("system", "Given the conversation above, generate a search query to find relevant information.")
])

# Create history-aware retriever
llm = ChatOpenAI()
history_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=rephrase_prompt
)

# First query (no history)
docs = history_retriever.invoke({
    "input": "What is LangChain?",
    "chat_history": []
})
# Returns documents about LangChain

# Follow-up query (with history)
from langchain_core.messages import HumanMessage, AIMessage

docs = history_retriever.invoke({
    "input": "How does it work with RAG?",
    "chat_history": [
        HumanMessage(content="What is LangChain?"),
        AIMessage(content="LangChain is a framework for building LLM applications.")
    ]
})
# LLM reformulates to: "How does LangChain work with RAG?"
# Returns relevant documents about LangChain and RAG
</syntaxhighlight>

=== Using Hub Prompt ===

<syntaxhighlight lang="python">
from langchain_classic.chains import create_history_aware_retriever
from langchain_openai import ChatOpenAI
from langchain_classic import hub

# Pull pre-built prompt from LangChain Hub
rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

model = ChatOpenAI()
retriever = ...  # Your retriever

# Create chain with hub prompt
chat_retriever = create_history_aware_retriever(
    model, retriever, rephrase_prompt
)

# Use in conversation
result = chat_retriever.invoke({
    "input": "Tell me more about that",
    "chat_history": [
        HumanMessage(content="What is prompt engineering?"),
        AIMessage(content="Prompt engineering is...")
    ]
})
</syntaxhighlight>

=== Complete RAG Chain ===

<syntaxhighlight lang="python">
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOpenAI()
retriever = ...  # Your retriever

# Step 1: Create history-aware retriever
rephrase_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("system", "Reformulate the question to be standalone.")
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, rephrase_prompt
)

# Step 2: Create QA chain
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on this context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

qa_chain = create_stuff_documents_chain(llm, qa_prompt)

# Step 3: Combine into full RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# Use the complete chain
response = rag_chain.invoke({
    "input": "What are embeddings?",
    "chat_history": []
})

print(response["answer"])
print(response["context"])  # Retrieved documents
</syntaxhighlight>

== Validation ==

The function validates that the prompt includes the required "input" variable:

<syntaxhighlight lang="python">
if "input" not in prompt.input_variables:
    msg = (
        "Expected `input` to be a prompt variable, "
        f"but got {prompt.input_variables}"
    )
    raise ValueError(msg)
</syntaxhighlight>

== Design Patterns ==

=== Conditional Execution ===

Uses '''RunnableBranch''' to implement conditional logic in LCEL:
* '''Branch 1:''' Direct retrieval (no history)
* '''Branch 2:''' LLM reformulation → retrieval (with history)

=== Query Reformulation ===

Addresses the '''context dependency problem''' in conversations:
* User: "What is RAG?"
* AI: "RAG is Retrieval-Augmented Generation..."
* User: "How does it work?" ← Needs reformulation to "How does RAG work?"

=== Zero-Shot Chain Configuration ===

Named with <code>.with_config(run_name="chat_retriever_chain")</code> for observability in tracing tools.

== Prompt Requirements ==

The prompt template MUST include:

{| class="wikitable"
|+ Required Variables
! Variable !! Type !! Purpose
|-
| <code>input</code> || <code>str</code> || Current user query (validated)
|-
| <code>chat_history</code> || <code>list[BaseMessage]</code> || Previous conversation (optional but recommended)
|}

=== Example Prompt Structure ===

<syntaxhighlight lang="python">
ChatPromptTemplate.from_messages([
    ("system", "Reformulate the user question to be standalone."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
</syntaxhighlight>

== Common Use Cases ==

=== Multi-Turn Question Answering ===
* Customer support chatbots with document retrieval
* Technical documentation assistants
* Research paper Q&A systems

=== Follow-Up Question Handling ===
* "Tell me more about that"
* "What about X?" (where X refers to previous context)
* "How does it compare?" (comparing to previously discussed topic)

=== Context-Aware Search ===
* Session-based information retrieval
* Personalized document search
* Conversational semantic search

== Performance Considerations ==

{| class="wikitable"
|+ Performance Characteristics
! Aspect !! Impact !! Notes
|-
| LLM Calls || 0 or 1 per invocation || Only called when chat_history exists
|-
| Latency || +200-500ms || When LLM reformulation is triggered
|-
| Tokens || Variable || Depends on chat_history length
|-
| Cost || Variable || Only pay for LLM when history exists
|}

=== Optimization Tips ===

* '''Limit history length:''' Include only recent messages to reduce tokens
* '''Use faster models:''' Consider GPT-3.5 for reformulation (cheaper/faster)
* '''Cache prompts:''' Enable prompt caching for repeated patterns
* '''Streaming:''' Enable streaming for better perceived performance

== Error Handling ==

<syntaxhighlight lang="python">
try:
    docs = history_retriever.invoke({
        "input": "What is this?",
        "chat_history": chat_history
    })
except ValueError as e:
    # Handle missing input variable in prompt
    print(f"Prompt validation error: {e}")
except Exception as e:
    # Handle retrieval or LLM errors
    print(f"Retrieval error: {e}")
</syntaxhighlight>

== Integration with Other Components ==

=== Upstream ===
* '''RunnableWithMessageHistory:''' Automatic history management
* '''Memory Systems:''' Legacy memory classes
* '''Chat Interfaces:''' Frontend chat applications

=== Downstream ===
* '''create_retrieval_chain:''' Combines retriever with QA chain
* '''create_stuff_documents_chain:''' Formats retrieved documents for LLM
* '''Answer Generation:''' Final LLM call with retrieved context

== Related Components ==

* '''RunnableBranch''' (langchain-core) - Conditional execution primitive
* '''StrOutputParser''' (langchain-core) - Parses LLM string output
* '''BaseRetriever''' (langchain-core) - Retriever interface
* '''create_retrieval_chain''' - Next step in RAG pipeline
* '''RunnableWithMessageHistory''' - Automatic history management

== See Also ==

* [[langchain-ai_langchain_create_retrieval_chain|create_retrieval_chain]] - Complete RAG chain builder
* [[langchain-ai_langchain_RunnableBranch|RunnableBranch]] - Conditional execution
* [[langchain-ai_langchain_BaseRetriever|BaseRetriever]] - Retriever interface
* LangChain documentation on Conversational RAG
* LangChain Hub for pre-built prompts
