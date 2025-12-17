# File: `examples/online_serving/retrieval_augmented_generation_with_llamaindex.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 225 |
| Functions | `init_config`, `load_documents`, `setup_models`, `setup_vector_store`, `create_index`, `query_document`, `get_parser`, `main` |
| Imports | argparse, llama_index, typing |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** RAG with LlamaIndex integration

**Mechanism:** RAG pipeline using LlamaIndex instead of LangChain. Loads documents, generates embeddings, creates vector index in Milvus, and performs semantic search with answer generation. Configures vLLM as both embedding and chat model through LlamaIndex abstractions.

**Significance:** Alternative RAG framework example showing LlamaIndex integration. Provides choice of RAG frameworks for users preferring LlamaIndex over LangChain.
