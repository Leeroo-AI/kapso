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

**Purpose:** RAG implementation using LlamaIndex and vLLM

**Mechanism:** Similar to LangChain version but using LlamaIndex framework. Loads documents with SimpleWebPageReader, configures global Settings for embedding/LLM models (OpenAILike adapters), creates vector store with MilvusVectorStore, builds VectorStoreIndex, and queries through query_engine. Supports SentenceSplitter for chunking and interactive mode. Automatically detects embedding dimensions from test embedding.

**Significance:** Alternative RAG implementation for LlamaIndex users. Shows vLLM compatibility with LlamaIndex ecosystem through OpenAILike adapters. Important for teams already using LlamaIndex who want to integrate vLLM. Demonstrates framework flexibility - same vLLM backend works with multiple RAG frameworks. Useful for comparing LangChain vs LlamaIndex approaches to RAG.
