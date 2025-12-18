{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|LangChain Text Splitters|https://docs.langchain.com/oss/python/langchain/overview]]
* [[source::Blog|Document Loaders|https://python.langchain.com/docs/modules/data_connection/document_loaders/]]
|-
! Domains
| [[domain::NLP]], [[domain::RAG]], [[domain::Document_Processing]], [[domain::ETL]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Process of converting document content into chunked representations suitable for embedding, retrieval, and LLM consumption.

=== Description ===

Document Transformation is the core operation that takes loaded documents and converts them into appropriately-sized chunks. This transformation is essential in the RAG pipeline because:
* Raw documents are often too large for LLM context windows
* Retrieval works better with focused, semantically coherent chunks
* Embeddings capture meaning better at appropriate granularities

The transformation preserves metadata from source documents while adding chunk-specific information (like start_index).

=== Usage ===

Apply Document Transformation when:
* Building RAG pipelines
* Preparing documents for vector stores
* Creating searchable document indexes
* Processing documents for LLM analysis

This is typically the second step after document loading and before embedding.

== Theoretical Basis ==

Document Transformation implements **Chunking Algorithms** for information retrieval.

'''1. The Transformation Pipeline'''

<syntaxhighlight lang="text">
Document Loading          Document Transformation        Embedding & Storage
      │                           │                            │
      ▼                           ▼                            ▼
┌──────────────┐          ┌──────────────┐          ┌──────────────┐
│  Raw Files   │ ──────▶  │  Documents   │ ──────▶  │   Chunks     │
│  (PDF, TXT)  │          │  (page_content│          │  + Metadata  │
│              │          │   + metadata) │          │  + Embeddings│
└──────────────┘          └──────────────┘          └──────────────┘
</syntaxhighlight>

'''2. Split Operation Semantics'''

<syntaxhighlight lang="python">
# Pseudo-code for split operation
def split_documents(documents: list[Document], splitter) -> list[Document]:
    """Transform documents into chunks."""
    result = []

    for doc in documents:
        # Extract text and metadata
        text = doc.page_content
        metadata = doc.metadata

        # Split text into chunks
        chunks = splitter.split_text(text)

        # Create new Document for each chunk
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **metadata,  # Inherit all parent metadata
                "chunk_index": i,
            }

            if splitter.add_start_index:
                chunk_metadata["start_index"] = find_position(text, chunk)

            result.append(Document(
                page_content=chunk,
                metadata=chunk_metadata
            ))

    return result
</syntaxhighlight>

'''3. Metadata Preservation'''

<syntaxhighlight lang="python">
# Metadata flows from parent to children
original = Document(
    page_content="Very long document...",
    metadata={
        "source": "report.pdf",
        "page": 1,
        "author": "Alice",
        "created": "2024-01-01",
    }
)

# After splitting:
chunks = [
    Document(
        page_content="First chunk...",
        metadata={
            "source": "report.pdf",    # Inherited
            "page": 1,                  # Inherited
            "author": "Alice",          # Inherited
            "created": "2024-01-01",    # Inherited
            "start_index": 0,           # Added by splitter
        }
    ),
    Document(
        page_content="Second chunk...",
        metadata={
            "source": "report.pdf",
            "page": 1,
            "author": "Alice",
            "created": "2024-01-01",
            "start_index": 500,         # Different for each chunk
        }
    ),
    # ...
]
</syntaxhighlight>

'''4. Merge Algorithm'''

The `_merge_splits` algorithm combines small pieces:

<syntaxhighlight lang="python">
# Pseudo-code for merge operation
def merge_splits(splits: list[str], separator: str, chunk_size: int, overlap: int) -> list[str]:
    """Merge small splits into larger chunks."""
    chunks = []
    current = []
    current_length = 0

    for split in splits:
        split_length = length_function(split)
        separator_length = length_function(separator) if current else 0

        # Check if adding this split exceeds chunk_size
        if current_length + separator_length + split_length > chunk_size:
            if current:
                # Emit current chunk
                chunks.append(separator.join(current))

                # Handle overlap: keep trailing splits
                while current_length > overlap:
                    removed = current.pop(0)
                    current_length -= length_function(removed) + length_function(separator)

        # Add split to current chunk
        current.append(split)
        current_length += split_length + (separator_length if len(current) > 1 else 0)

    # Don't forget the last chunk
    if current:
        chunks.append(separator.join(current))

    return chunks
</syntaxhighlight>

'''5. Start Index Calculation'''

<syntaxhighlight lang="python">
# Pseudo-code for position tracking
def calculate_start_indices(original_text: str, chunks: list[str], overlap: int) -> list[int]:
    """Calculate the start position of each chunk in original text."""
    indices = []
    search_start = 0

    for i, chunk in enumerate(chunks):
        if i > 0:
            # Account for overlap - search after previous chunk start
            search_start = max(0, indices[-1] + len(chunks[i-1]) - overlap)

        # Find chunk in original text
        position = original_text.find(chunk, search_start)
        indices.append(position)

    return indices
</syntaxhighlight>

'''6. Recursive Split Decision Tree'''

<syntaxhighlight lang="text">
           Input Text
               │
               ▼
    Split by first separator
               │
       ┌───────┴───────┐
       │               │
   Each split      Splits fit in
   fits in         chunk_size?
   chunk_size?          │
       │               No
      Yes               │
       │                ▼
       ▼           Recursively split
    Merge and      with next separator
    return              │
                        │
                  (Repeat until all
                   chunks fit or
                   no separators left)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:langchain-ai_langchain_TextSplitter_split_methods]]

=== Used By Workflows ===
* Text_Splitting_Workflow (Step 4)
