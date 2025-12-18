# File: `libs/text-splitters/langchain_text_splitters/sentence_transformers.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 112 |
| Classes | `SentenceTransformersTokenTextSplitter` |
| Imports | __future__, langchain_text_splitters, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Token-based text splitter using SentenceTransformer model tokenizers.

**Mechanism:** SentenceTransformersTokenTextSplitter loads a SentenceTransformer model (default "sentence-transformers/all-mpnet-base-v2"), extracts its tokenizer, and uses it to encode text into token IDs. Strips start/stop tokens from encoding before splitting. Respects the model's max_seq_length constraint. Uses split_text_on_tokens utility with configurable chunk_overlap and tokens_per_chunk.

**Significance:** Ensures text chunks align with the token limits of specific sentence transformer models used for embeddings. Critical for RAG applications using sentence transformers, as it prevents token count mismatches between splitting and embedding that could cause truncation or errors. The token-aware splitting optimizes for the downstream embedding model's capabilities.
