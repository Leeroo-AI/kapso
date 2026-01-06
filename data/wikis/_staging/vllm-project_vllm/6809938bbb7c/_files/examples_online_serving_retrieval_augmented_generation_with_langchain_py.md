# File: `examples/online_serving/retrieval_augmented_generation_with_langchain.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 257 |
| Functions | `load_and_split_documents`, `init_vectorstore`, `init_llm`, `get_qa_prompt`, `format_docs`, `create_qa_chain`, `get_parser`, `init_config`, `... +1 more` |
| Imports | argparse, langchain_community, langchain_core, langchain_milvus, langchain_openai, langchain_text_splitters, typing |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Complete RAG implementation using LangChain and vLLM

**Mechanism:** Full RAG pipeline: loads web documents with WebBaseLoader, splits into chunks with RecursiveCharacterTextSplitter, stores in Milvus vector database using vLLM embeddings (OpenAIEmbeddings adapter), retrieves relevant contexts, and generates answers using vLLM chat model (ChatOpenAI adapter). Supports interactive Q&A mode and configurable parameters (chunk size, top-k, endpoints). Uses LangChain's LCEL (pipe syntax) for chain construction.

**Significance:** Reference implementation for RAG applications with vLLM. Shows integration with LangChain ecosystem, enabling access to LangChain's extensive tooling. Critical for developers building knowledge-base question-answering, document search, or context-augmented chatbots. Demonstrates dual vLLM service pattern: separate endpoints for embeddings and chat generation.
