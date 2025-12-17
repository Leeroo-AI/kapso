# File: `libs/text-splitters/langchain_text_splitters/sentence_transformers.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 112 |
| Classes | `SentenceTransformersTokenTextSplitter` |
| Imports | __future__, langchain_text_splitters, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Splits text based on sentence transformer model tokenization to ensure chunks fit within embedding model token limits.

**Mechanism:** Inherits from TextSplitter and loads a SentenceTransformer model (default "sentence-transformers/all-mpnet-base-v2") to access its tokenizer. Initializes chunk configuration from model's max_seq_length, validating that tokens_per_chunk doesn't exceed model limits. split_text() creates a Tokenizer with encode function that strips start/end tokens (encodes text then removes first and last token IDs), then calls split_text_on_tokens() for sliding window chunking. count_tokens() method encodes text and returns token count. Uses _max_length_equal_32_bit_integer with truncation="do_not_truncate" to handle arbitrarily long inputs.

**Significance:** Critical for embedding-based RAG systems where chunks must fit within the embedding model's context window. Using the actual model tokenizer ensures accurate token counting and prevents truncation issues. Particularly important for sentence transformer models which have varying context windows (typically 128-512 tokens). Strips special tokens to maximize usable context length per chunk.
