= document_metadata =

'''Sources:''' /tmp/praxium_repo_wjjl6pl8/libs/text-splitters/langchain_text_splitters/base.py:L91-117

'''Domains:''' Text Splitting, RAG, Document Processing, Information Retrieval, Data Provenance

'''Last Updated:''' 2025-12-17

== Overview ==

The document_metadata implementation handles the preservation and augmentation of document metadata during text splitting. The split_documents() and create_documents() methods automatically copy metadata from source documents to all chunks, optionally adding chunk-specific fields like start_index.

== Description ==

LangChain's TextSplitter automatically preserves metadata when splitting documents. The implementation:

1. Extracts text and metadata from input documents
2. Splits text into chunks using split_text()
3. Deep-copies metadata for each chunk
4. Optionally adds start_index to track chunk position
5. Creates new Document objects with preserved metadata

This process is transparent to users - simply call split_documents() and metadata is automatically maintained.

Key features:

'''Automatic Propagation:''' All metadata fields from source documents are copied to every chunk
'''Deep Copy:''' Each chunk gets an independent copy of metadata (no shared state)
'''Position Tracking:''' Optional start_index field records chunk position in original text
'''Immutability:''' Original document metadata is never modified

== Code Reference ==

The implementation is in libs/text-splitters/langchain_text_splitters/base.py:

<syntaxhighlight lang="python">
class TextSplitter(BaseDocumentTransformer, ABC):
    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

    def create_documents(
        self, texts: list[str], metadatas: list[dict[Any, Any]] | None = None
    ) -> list[Document]:
        """Create a list of Document objects from a list of texts."""
        metadatas_ = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            index = 0
            previous_chunk_len = 0
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(metadatas_[i])  # Deep copy
                if self._add_start_index:
                    # Add chunk position
                    offset = index + previous_chunk_len - self._chunk_overlap
                    index = text.find(chunk, max(0, offset))
                    metadata["start_index"] = index
                    previous_chunk_len = len(chunk)
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents
</syntaxhighlight>

== API ==

=== Automatic Metadata Preservation ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = [
    Document(
        page_content="Long text content...",
        metadata={"source": "doc.pdf", "page": 1, "author": "Alice"}
    )
]

# Metadata automatically preserved in all chunks
chunks = splitter.split_documents(docs)

for chunk in chunks:
    # Each chunk has the same metadata as the source
    print(chunk.metadata)  # {"source": "doc.pdf", "page": 1, "author": "Alice"}
</syntaxhighlight>

=== Adding Start Index ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Enable start index tracking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True  # Add position tracking
)

docs = [
    Document(
        page_content="Text content...",
        metadata={"source": "doc.pdf"}
    )
]

chunks = splitter.split_documents(docs)

for chunk in chunks:
    # Original metadata + start_index
    print(chunk.metadata)
    # {"source": "doc.pdf", "start_index": 0}
    # {"source": "doc.pdf", "start_index": 800}
    # etc.
</syntaxhighlight>

=== Manual Document Creation ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

texts = ["Text 1", "Text 2"]
metadatas = [
    {"source": "doc1.txt", "author": "Alice"},
    {"source": "doc2.txt", "author": "Bob"}
]

# Create documents with metadata
docs = splitter.create_documents(texts, metadatas=metadatas)
</syntaxhighlight>

== I/O Contract ==

=== Input ===

'''split_documents():'''
* documents (Iterable[Document]) - Documents with page_content and metadata

'''create_documents():'''
* texts (list[str]) - List of texts to split
* metadatas (list[dict] | None) - Metadata for each text (default: empty dicts)

=== Output ===

Both methods return:
* list[Document] - Document chunks with preserved metadata

Each output Document has:
* '''page_content''' (str) - Chunk text content
* '''metadata''' (dict) - Deep copy of original metadata, optionally augmented with start_index

=== Metadata Fields ===

'''Preserved (from source):'''
* All fields from original document metadata
* Examples: source, page, author, date, title, etc.

'''Added (if add_start_index=True):'''
* '''start_index''' (int) - Position where chunk starts in original text

== Usage Examples ==

=== Example 1: Basic Metadata Preservation ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

doc = Document(
    page_content="Chapter 1 content here. More content follows. Even more content.",
    metadata={
        "source": "book.pdf",
        "chapter": 1,
        "title": "Introduction",
        "author": "John Doe"
    }
)

chunks = splitter.split_documents([doc])

print(f"Split into {len(chunks)} chunks")
for i, chunk in enumerate(chunks, 1):
    print(f"\nChunk {i}:")
    print(f"Content: {chunk.page_content}")
    print(f"Metadata: {chunk.metadata}")
    # All chunks have the same metadata
</syntaxhighlight>

=== Example 2: Tracking Chunk Positions ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    add_start_index=True
)

doc = Document(
    page_content="First section here. Second section here. Third section here.",
    metadata={"source": "document.txt"}
)

chunks = splitter.split_documents([doc])

for chunk in chunks:
    start = chunk.metadata["start_index"]
    content = chunk.page_content
    print(f"Position {start}: {content}")
</syntaxhighlight>

=== Example 3: Multiple Documents with Different Metadata ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

docs = [
    Document(
        page_content="Content from first document...",
        metadata={"source": "doc1.pdf", "author": "Alice", "year": 2023}
    ),
    Document(
        page_content="Content from second document...",
        metadata={"source": "doc2.pdf", "author": "Bob", "year": 2024}
    ),
    Document(
        page_content="Content from third document...",
        metadata={"source": "doc3.pdf", "author": "Charlie", "year": 2025}
    )
]

chunks = splitter.split_documents(docs)

# Group chunks by source
from collections import defaultdict
by_source = defaultdict(list)

for chunk in chunks:
    source = chunk.metadata["source"]
    by_source[source].append(chunk)

for source, source_chunks in by_source.items():
    print(f"\n{source}: {len(source_chunks)} chunks")
    author = source_chunks[0].metadata["author"]
    year = source_chunks[0].metadata["year"]
    print(f"  Author: {author}, Year: {year}")
</syntaxhighlight>

=== Example 4: Filtering Retrieved Chunks by Metadata ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Simulate documents with various metadata
docs = [
    Document(
        page_content="Research findings from 2023...",
        metadata={"type": "research", "year": 2023, "author": "Alice"}
    ),
    Document(
        page_content="Research findings from 2024...",
        metadata={"type": "research", "year": 2024, "author": "Bob"}
    ),
    Document(
        page_content="Blog post about the topic...",
        metadata={"type": "blog", "year": 2024, "author": "Charlie"}
    )
]

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_documents(docs)

# Filter chunks by metadata
recent_research = [
    chunk for chunk in chunks
    if chunk.metadata.get("type") == "research" and
       chunk.metadata.get("year") >= 2024
]

print(f"Found {len(recent_research)} recent research chunks")
for chunk in recent_research:
    print(f"Author: {chunk.metadata['author']}, Year: {chunk.metadata['year']}")
</syntaxhighlight>

=== Example 5: Building Citations from Metadata ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

docs = [
    Document(
        page_content="Quantum computing shows promise for drug discovery...",
        metadata={
            "source": "Nature Journal",
            "title": "Quantum Computing in Medicine",
            "year": 2024,
            "page": 42
        }
    )
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    add_start_index=True
)

chunks = splitter.split_documents(docs)

# Generate citations
def format_citation(chunk: Document) -> str:
    meta = chunk.metadata
    source = meta.get("source", "Unknown")
    title = meta.get("title", "Untitled")
    year = meta.get("year", "n.d.")
    page = meta.get("page", "")

    citation = f"{source}. ({year}). {title}"
    if page:
        citation += f", p. {page}"
    return citation

for chunk in chunks:
    citation = format_citation(chunk)
    print(f"Citation: {citation}")
    print(f"Content: {chunk.page_content[:60]}...")
    print()
</syntaxhighlight>

=== Example 6: Complex Metadata Structures ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Nested metadata structure
doc = Document(
    page_content="Complex document content...",
    metadata={
        "source": {
            "filename": "report.pdf",
            "url": "https://example.com/report.pdf",
            "format": "pdf"
        },
        "authors": ["Alice", "Bob", "Charlie"],
        "tags": ["AI", "machine-learning", "research"],
        "date": {
            "created": "2025-01-01",
            "modified": "2025-01-15"
        },
        "access": {
            "level": "public",
            "license": "CC-BY-4.0"
        }
    }
)

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_documents([doc])

# Complex metadata is preserved via deep copy
for chunk in chunks:
    # Each chunk has independent copy of nested structures
    print(f"Source URL: {chunk.metadata['source']['url']}")
    print(f"Authors: {', '.join(chunk.metadata['authors'])}")
    print(f"Tags: {chunk.metadata['tags']}")
    print(f"License: {chunk.metadata['access']['license']}")
    print()
</syntaxhighlight>

=== Example 7: Augmenting Metadata After Splitting ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

doc = Document(
    page_content="Document content that will be split into chunks...",
    metadata={"source": "doc.pdf"}
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    add_start_index=True
)

chunks = splitter.split_documents([doc])

# Augment chunks with additional metadata
for i, chunk in enumerate(chunks):
    # Add chunk-specific metadata
    chunk.metadata["chunk_id"] = i
    chunk.metadata["total_chunks"] = len(chunks)
    chunk.metadata["chunk_length"] = len(chunk.page_content)

    print(f"Chunk {i+1}/{len(chunks)}:")
    print(f"  Position: {chunk.metadata['start_index']}")
    print(f"  Length: {chunk.metadata['chunk_length']}")
    print(f"  Content: {chunk.page_content[:40]}...")
</syntaxhighlight>

=== Example 8: Metadata Independence (Deep Copy) ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

doc = Document(
    page_content="Text to split",
    metadata={"tags": ["original"], "count": 0}
)

splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=2)
chunks = splitter.split_documents([doc])

# Modify first chunk's metadata
chunks[0].metadata["tags"].append("modified")
chunks[0].metadata["count"] = 1

# Other chunks are unaffected (deep copy ensures independence)
print(f"Chunk 0 tags: {chunks[0].metadata['tags']}")  # ["original", "modified"]
print(f"Chunk 1 tags: {chunks[1].metadata['tags']}")  # ["original"]

print(f"Chunk 0 count: {chunks[0].metadata['count']}")  # 1
print(f"Chunk 1 count: {chunks[1].metadata['count']}")  # 0

# Original document also unaffected
print(f"Original tags: {doc.metadata['tags']}")  # ["original"]
</syntaxhighlight>

== Related Pages ==

* [[langchain-ai_langchain_Metadata_Preservation|Metadata_Preservation]] - Principle of preserving metadata
* [[langchain-ai_langchain_split_text_method|split_text_method]] - Methods for splitting with metadata
* [[langchain-ai_langchain_Document_Splitting|Document_Splitting]] - Principle of document splitting
* [[langchain-ai_langchain_chunk_parameters|chunk_parameters]] - Configuration including add_start_index
