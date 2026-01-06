{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|LangChain Documents|https://python.langchain.com/docs/modules/data_connection/document_transformers/]]
* [[source::Doc|LangChain Text Splitters|https://docs.langchain.com/oss/python/langchain/overview]]
|-
! Domains
| [[domain::NLP]], [[domain::RAG]], [[domain::Metadata]], [[domain::Data_Lineage]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Strategy for preserving, copying, and extending document metadata through text splitting operations.

=== Description ===

Metadata Handling ensures that information about document origin, context, and position survives the chunking process. Without proper metadata handling, you lose the ability to:
* Cite sources in RAG responses
* Filter search by document attributes
* Navigate back to original content
* Track data lineage through pipelines

This principle defines how metadata flows from parent documents to child chunks, and what additional metadata should be added during transformation.

=== Usage ===

Apply Metadata Handling when:
* Building production RAG systems needing citations
* Implementing filtered vector search
* Creating audit trails for generated content
* Tracking chunk provenance for debugging

Metadata categories:
* **Inherited:** Source, author, date, tags (from parent)
* **Computed:** start_index, chunk_index, char_count (added during split)
* **Contextual:** Page number, section header (format-specific)

== Theoretical Basis ==

Metadata Handling implements **Data Lineage** and **Provenance Tracking** patterns.

'''1. Metadata Flow Model'''

<syntaxhighlight lang="text">
Source Document                    Chunks
┌──────────────────┐         ┌──────────────────┐
│ page_content     │         │ page_content     │ (subset of parent)
│ metadata:        │    ──►  │ metadata:        │
│   source: "..."  │         │   source: "..."  │ (inherited)
│   author: "..."  │   split │   author: "..."  │ (inherited)
│   date: "..."    │         │   date: "..."    │ (inherited)
│                  │         │   start_index: N │ (computed)
│                  │         │   chunk_index: M │ (computed)
└──────────────────┘         └──────────────────┘
                             ┌──────────────────┐
                             │ page_content     │
                             │ metadata:        │
                             │   source: "..."  │
                             │   author: "..."  │
                             │   date: "..."    │
                             │   start_index: N │
                             │   chunk_index: M │
                             └──────────────────┘
</syntaxhighlight>

'''2. Deep Copy Semantics'''

<syntaxhighlight lang="python">
import copy

def create_chunk_with_metadata(
    chunk_text: str,
    parent_metadata: dict,
    chunk_index: int,
    start_index: int | None = None,
) -> Document:
    """Create chunk with properly copied metadata."""
    # Deep copy prevents mutation across chunks
    chunk_metadata = copy.deepcopy(parent_metadata)

    # Add computed fields
    chunk_metadata["chunk_index"] = chunk_index
    if start_index is not None:
        chunk_metadata["start_index"] = start_index

    return Document(
        page_content=chunk_text,
        metadata=chunk_metadata
    )
</syntaxhighlight>

'''3. Start Index Calculation'''

<syntaxhighlight lang="python">
def calculate_start_index(
    original_text: str,
    chunk: str,
    previous_end: int,
    chunk_overlap: int,
) -> int:
    """Find chunk's position in original text."""
    # Start searching after previous chunk, accounting for overlap
    search_start = max(0, previous_end - chunk_overlap)

    # Find exact position of chunk
    position = original_text.find(chunk, search_start)

    if position == -1:
        # Handle edge cases (whitespace stripping, etc.)
        # Try searching from beginning as fallback
        position = original_text.find(chunk)

    return position
</syntaxhighlight>

'''4. Metadata Inheritance Patterns'''

<syntaxhighlight lang="python">
# Pattern 1: Full inheritance (default)
# All parent metadata copied to all children
chunk.metadata = {**parent.metadata}

# Pattern 2: Selective inheritance
# Only specified fields copied
INHERIT_FIELDS = ["source", "author", "date"]
chunk.metadata = {k: v for k, v in parent.metadata.items() if k in INHERIT_FIELDS}

# Pattern 3: Namespace isolation
# Parent metadata nested to prevent conflicts
chunk.metadata = {
    "parent": parent.metadata,
    "chunk": {"index": i, "start": start_index}
}
</syntaxhighlight>

'''5. Metadata for RAG Filtering'''

<syntaxhighlight lang="python">
# Metadata enables filtered vector search
# Example: Search only in recent documents by a specific author
query_embedding = embed(query)

# Filter by metadata
results = vector_store.similarity_search(
    query_embedding,
    filter={
        "author": "Alice",
        "date": {"$gte": "2024-01-01"},
        "source": {"$contains": "technical"}
    },
    k=5
)
</syntaxhighlight>

'''6. Citation and Source Attribution'''

<syntaxhighlight lang="python">
# Using metadata for citations in RAG responses
def generate_with_citations(query: str, retrieved_chunks: list[Document]) -> str:
    # Build context with source markers
    context_parts = []
    sources = {}

    for i, chunk in enumerate(retrieved_chunks):
        source_key = f"[{i+1}]"
        context_parts.append(f"{source_key} {chunk.page_content}")

        # Build citation
        sources[source_key] = {
            "source": chunk.metadata.get("source", "Unknown"),
            "page": chunk.metadata.get("page"),
            "start_index": chunk.metadata.get("start_index"),
        }

    # Generate response with source markers
    response = llm.generate(
        f"Answer based on context. Cite sources.\n\nContext:\n{chr(10).join(context_parts)}\n\nQuestion: {query}"
    )

    # Append citation list
    citation_text = "\n\nSources:\n" + "\n".join(
        f"{k}: {v['source']}" for k, v in sources.items()
    )

    return response + citation_text
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:langchain-ai_langchain_TextSplitter_create_documents]]

=== Used By Workflows ===
* Text_Splitting_Workflow (Step 5)
