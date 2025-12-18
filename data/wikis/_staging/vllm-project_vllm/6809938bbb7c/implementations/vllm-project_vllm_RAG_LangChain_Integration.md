{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::RAG]], [[domain::LangChain]], [[domain::Vector Search]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
RAG implementation using LangChain, Milvus vector store, and vLLM for both embeddings and chat completion.

=== Description ===
This example demonstrates a complete Retrieval Augmented Generation (RAG) system that combines document retrieval with language model generation. It uses LangChain for orchestration, loads web content, splits it into chunks, stores embeddings in Milvus vector database, and answers questions using retrieved context. The system runs two separate vLLM services: one for generating embeddings and another for chat completion, showing how to integrate multiple vLLM endpoints in a RAG pipeline.

=== Usage ===
Use this example when building question-answering systems that need to reference external knowledge bases, implementing document search with semantic understanding, or creating conversational AI that grounds responses in retrieved content. It's particularly valuable for applications requiring both embedding generation and text generation with separate model optimizations.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/online_serving/retrieval_augmented_generation_with_langchain.py examples/online_serving/retrieval_augmented_generation_with_langchain.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Start embedding service (port 8000)
vllm serve ssmits/Qwen2-7B-Instruct-embed-base

# Start chat service (port 8001)
vllm serve qwen/Qwen1.5-0.5B-Chat --port 8001

# Run single question mode
python retrieval_augmented_generation_with_langchain.py

# Run interactive Q&A mode
python retrieval_augmented_generation_with_langchain.py --interactive

# Custom configuration
python retrieval_augmented_generation_with_langchain.py \
  --url "https://docs.vllm.ai/en/latest/" \
  --chunk-size 1000 \
  --chunk-overlap 200 \
  --top-k 5
</syntaxhighlight>

== Key Concepts ==

=== Multi-Service Architecture ===
The system runs two separate vLLM services on different ports: one optimized for embedding generation (port 8000) and another for chat completion (port 8001). This separation allows independent scaling and optimization of each model type.

=== Document Processing Pipeline ===
Web content is loaded, split into overlapping chunks (default 1000 characters with 200 character overlap), embedded, and stored in Milvus. The overlap ensures context continuity across chunk boundaries.

=== RAG Chain ===
LangChain's LCEL (LangChain Expression Language) creates a chain that: (1) retrieves relevant documents based on query similarity, (2) formats retrieved context, (3) constructs a prompt with question and context, and (4) generates an answer using the chat model.

=== Vector Store Integration ===
Uses Milvus as the vector database with configurable URI (defaults to local file ./milvus.db). The vector store enables semantic search by finding documents with embeddings closest to the query embedding.

== Usage Examples ==

<syntaxhighlight lang="python">
# Initialize configuration
config = {
    "vllm_embedding_endpoint": "http://localhost:8000/v1",
    "vllm_chat_endpoint": "http://localhost:8001/v1",
    "embedding_model": "ssmits/Qwen2-7B-Instruct-embed-base",
    "chat_model": "qwen/Qwen1.5-0.5B-Chat",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k": 3
}

# Load and split documents from web URL
loader = WebBaseLoader(web_paths=("https://docs.vllm.ai/en/latest/getting_started/quickstart.html",))
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config["chunk_size"],
    chunk_overlap=config["chunk_overlap"]
)
documents = text_splitter.split_documents(docs)

# Initialize vector store with embeddings
vectorstore = Milvus.from_documents(
    documents=documents,
    embedding=OpenAIEmbeddings(
        model=config["embedding_model"],
        openai_api_base=config["vllm_embedding_endpoint"]
    ),
    connection_args={"uri": "./milvus.db"}
)

# Create retriever and LLM
retriever = vectorstore.as_retriever(search_kwargs={"k": config["top_k"]})
llm = ChatOpenAI(
    model=config["chat_model"],
    openai_api_base=config["vllm_chat_endpoint"]
)

# Build QA chain with LCEL
qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Query the system
answer = qa_chain.invoke("How to install vLLM?")
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
* [[implements::Component:vLLM_Embeddings_API]]
* [[implements::Component:vLLM_Chat_Completions_API]]
* [[uses::Tool:LangChain]]
* [[uses::Tool:Milvus]]
* [[related::Implementation:vllm-project_vllm_RAG_LlamaIndex_Integration]]
