{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::RAG]], [[domain::LlamaIndex]], [[domain::Vector Search]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
RAG implementation using LlamaIndex framework with vLLM for embeddings and generation, and Milvus for vector storage.

=== Description ===
This example demonstrates how to build a RAG system using LlamaIndex as the orchestration framework. It loads web documents, creates embeddings using vLLM's embedding service, stores vectors in Milvus, and uses a separate vLLM chat service for query answering. LlamaIndex provides a different abstraction compared to LangChain, with its Settings system for global model configuration and simpler indexing patterns. The example shows proper integration with vLLM's OpenAI-compatible endpoints.

=== Usage ===
Use this example when building RAG applications with LlamaIndex, when you prefer LlamaIndex's API design over LangChain, or when integrating vLLM into existing LlamaIndex workflows. It's suitable for document QA systems, knowledge base search, and any application requiring semantic retrieval with language generation.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/online_serving/retrieval_augmented_generation_with_llamaindex.py examples/online_serving/retrieval_augmented_generation_with_llamaindex.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Start embedding service (port 8000)
vllm serve ssmits/Qwen2-7B-Instruct-embed-base

# Start chat service (port 8001)
vllm serve qwen/Qwen1.5-0.5B-Chat --port 8001

# Run single query mode
python retrieval_augmented_generation_with_llamaindex.py

# Run interactive mode
python retrieval_augmented_generation_with_llamaindex.py --interactive

# Custom configuration
python retrieval_augmented_generation_with_llamaindex.py \
  --url "https://docs.vllm.ai/en/latest/" \
  --embedding-endpoint "http://localhost:8000/v1" \
  --chat-endpoint "http://localhost:8001/v1" \
  --chunk-size 1000 \
  --top-k 5
</syntaxhighlight>

== Key Concepts ==

=== LlamaIndex Settings Pattern ===
LlamaIndex uses a global Settings object to configure embedding models, LLMs, and transformations. This differs from LangChain's more explicit component passing, providing a centralized configuration approach.

=== Storage Context and Indices ===
The StorageContext wraps the vector store, and VectorStoreIndex creates the searchable index from documents. This pattern separates storage backend concerns from indexing logic.

=== Sentence Splitting ===
Uses SentenceSplitter transformer for chunking documents with configurable size and overlap. The splitter is registered in Settings.transformations for automatic application during indexing.

=== OpenAI-Like Integration ===
Uses OpenAILikeEmbedding and OpenAILike classes to connect to vLLM's OpenAI-compatible endpoints, enabling drop-in replacement of OpenAI services with vLLM.

=== Dynamic Vector Dimension ===
The example automatically detects embedding dimensions by generating a test embedding before creating the Milvus vector store, ensuring correct index configuration.

== Usage Examples ==

<syntaxhighlight lang="python">
# Configure embedding model
Settings.embed_model = OpenAILikeEmbedding(
    api_base="http://localhost:8000/v1",
    api_key="EMPTY",
    model_name="ssmits/Qwen2-7B-Instruct-embed-base"
)

# Configure LLM
Settings.llm = OpenAILike(
    model="qwen/Qwen1.5-0.5B-Chat",
    api_key="EMPTY",
    api_base="http://localhost:8001/v1",
    context_window=128000,
    is_chat_model=True
)

# Configure document splitting
Settings.transformations = [
    SentenceSplitter(chunk_size=1000, chunk_overlap=200)
]

# Load documents from web
documents = SimpleWebPageReader(html_to_text=True).load_data(
    ["https://docs.vllm.ai/en/latest/getting_started/quickstart.html"]
)

# Create vector store
sample_emb = Settings.embed_model.get_text_embedding("test")
vector_store = MilvusVectorStore(
    uri="./milvus_demo.db",
    dim=len(sample_emb),
    overwrite=True
)

# Create index
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

# Query the index
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("How to install vLLM?")
print(response)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
* [[implements::Component:vLLM_Embeddings_API]]
* [[implements::Component:vLLM_Chat_Completions_API]]
* [[uses::Tool:LlamaIndex]]
* [[uses::Tool:Milvus]]
* [[related::Implementation:vllm-project_vllm_RAG_LangChain_Integration]]
