# Heuristic: Chunk Size Configuration

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain Text Splitters|https://github.com/langchain-ai/langchain/tree/master/libs/text-splitters]]
* [[source::Doc|TextSplitter Base|libs/text-splitters/langchain_text_splitters/base.py]]
|-
! Domains
| [[domain::RAG]], [[domain::Text_Processing]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

## Overview

Configuration rules for text chunk sizes to optimize RAG retrieval quality and embedding model compatibility.

### Description

Text chunking is a critical step in RAG pipelines that balances **context preservation** against **retrieval precision**. Larger chunks preserve more context but may dilute relevance scores; smaller chunks improve precision but may lose important context. The optimal configuration depends on the embedding model's context window, the retrieval use case, and the nature of the documents.

### Usage

Apply this heuristic when:
- Configuring `RecursiveCharacterTextSplitter` or similar splitters
- Experiencing poor retrieval quality in RAG applications
- Getting chunks that are too large for your embedding model
- Seeing the warning: "Created a chunk of size X, which is longer than specified Y"

## The Insight (Rule of Thumb)

* **Action:** Set `chunk_size` between 500-2000 characters for general use cases
* **Default Value:** 4000 characters (LangChain default), but often too large for optimal retrieval
* **Recommended Starting Point:**
  - `chunk_size=1000` with `chunk_overlap=200` for most RAG applications
  - `chunk_size=500` with `chunk_overlap=100` for precise retrieval
  - `chunk_size=2000` with `chunk_overlap=400` for context-heavy tasks
* **Trade-off:** Smaller chunks = better precision but potential context loss; Larger chunks = better context but potential relevance dilution

### Chunk Overlap Guidelines

* **Action:** Set `chunk_overlap` to 10-20% of `chunk_size`
* **Constraint:** `chunk_overlap` must be strictly less than `chunk_size`
* **Trade-off:** Higher overlap = better continuity but more redundancy and storage

### Content-Specific Recommendations

| Content Type | chunk_size | chunk_overlap | Notes |
|--------------|------------|---------------|-------|
| General text | 1000 | 200 | Balanced default |
| Code | 500-1500 | 100-300 | Preserve function boundaries |
| JSON data | 2000 | 200 | Use `RecursiveJsonSplitter` |
| HTML | 1000 | 200 | Use `HTMLSectionSplitter` |
| Markdown | 1000-1500 | 200-300 | Use `MarkdownHeaderTextSplitter` |
| Legal/Technical | 1500-2000 | 300-400 | Context-heavy content |

## Reasoning

### Why These Values?

1. **Embedding Model Limits:** Most embedding models (OpenAI, Cohere, etc.) have 512-8192 token limits. A 1000-character chunk typically translates to ~200-300 tokens, safely within limits.

2. **Retrieval Quality:** Research shows that chunks of 100-500 tokens perform best for semantic search. Smaller chunks allow more precise matching.

3. **Context Window Costs:** Larger chunks consume more of the LLM's context window when used for augmentation, leaving less room for the conversation.

4. **The Warning Signal:** LangChain logs a warning when a chunk exceeds the configured size, indicating the text couldn't be split at natural boundaries.

### Empirical Evidence

From `libs/text-splitters/langchain_text_splitters/base.py:139-145`:
```python
if total > self._chunk_size:
    logger.warning(
        "Created a chunk of size %d, which is longer than the "
        "specified %d",
        total,
        self._chunk_size,
    )
```

This warning indicates the splitter couldn't find a separator within the chunk size limit—a sign that separators may need adjustment or chunk size increased.

### Validation Logic

From `libs/text-splitters/langchain_text_splitters/base.py:68-79`:
```python
if chunk_size <= 0:
    msg = f"chunk_size must be > 0, got {chunk_size}"
    raise ValueError(msg)
if chunk_overlap < 0:
    msg = f"chunk_overlap must be >= 0, got {chunk_overlap}"
    raise ValueError(msg)
if chunk_overlap > chunk_size:
    msg = (
        f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
        f"({chunk_size}), should be smaller."
    )
    raise ValueError(msg)
```

### JSON Splitter Defaults

From `libs/text-splitters/langchain_text_splitters/json.py:21-46`:
```python
max_chunk_size: int = 2000
min_chunk_size: int = 1800

def __init__(
    self, max_chunk_size: int = 2000, min_chunk_size: int | None = None
) -> None:
    self.max_chunk_size = max_chunk_size
    self.min_chunk_size = (
        min_chunk_size
        if min_chunk_size is not None
        else max(max_chunk_size - 200, 50)  # Default: 200 less than max, minimum 50
    )
```

## Related Pages

* [[applied_to::Implementation:langchain-ai_langchain_chunk_parameters]]
* [[applied_to::Implementation:langchain-ai_langchain_text_splitter_types]]
* [[applied_to::Workflow:langchain-ai_langchain_Text_Splitting_for_RAG]]
* [[applied_to::Principle:langchain-ai_langchain_Chunk_Configuration]]
