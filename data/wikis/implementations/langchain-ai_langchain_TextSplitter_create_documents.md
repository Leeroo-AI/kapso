{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Text Splitters|https://docs.langchain.com/oss/python/langchain/overview]]
|-
! Domains
| [[domain::NLP]], [[domain::RAG]], [[domain::Document_Processing]], [[domain::Metadata]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Concrete tool for creating Document objects from text chunks while preserving and extending metadata, provided by LangChain text-splitters.

=== Description ===

`TextSplitter.create_documents` converts raw text strings into `Document` objects with associated metadata. This method is particularly useful when you have text that needs splitting but want to attach metadata (like source, author, timestamp) to each resulting chunk.

Key capabilities:
* Creates Document objects from text strings
* Copies metadata to each chunk (deep copy to prevent mutation)
* Optionally calculates start_index for each chunk's position
* Handles parallel lists of texts and metadata

=== Usage ===

Use `create_documents` when:
* You have raw text strings (not Document objects)
* Need to attach metadata to chunks
* Processing text from sources without Document loader
* Building custom document pipelines

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain]
* '''File:''' libs/text-splitters/langchain_text_splitters/base.py
* '''Lines:''' L91-109

=== Signature ===
<syntaxhighlight lang="python">
class TextSplitter(ABC):
    def create_documents(
        self,
        texts: list[str],
        metadatas: list[dict[Any, Any]] | None = None,
    ) -> list[Document]:
        """Create a list of Document objects from a list of texts.

        Args:
            texts: List of text strings to split and convert to documents
            metadatas: Optional parallel list of metadata dicts
                      (one per text, deep copied to each chunk)

        Returns:
            List of Document objects with chunked page_content
            and metadata (including optional start_index)
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| texts || list[str] || Yes || List of text strings to split
|-
| metadatas || list[dict] | None || No || Parallel list of metadata dicts (one per text)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return || list[Document] || Document objects with page_content and metadata
|}

=== Metadata Fields Added ===
{| class="wikitable"
|-
! Field !! Type !! Condition !! Description
|-
| (all input metadata) || Any || Always || Deep copy of input metadata dict
|-
| start_index || int || add_start_index=True || Position of chunk in original text
|}

== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

# Raw texts without metadata
texts = [
    "First document content goes here...",
    "Second document content goes here...",
]

# Create documents (no metadata)
documents = splitter.create_documents(texts)

for doc in documents:
    print(f"Content: {doc.page_content[:50]}...")
    print(f"Metadata: {doc.metadata}")
</syntaxhighlight>

=== With Metadata ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

# Texts and parallel metadata
texts = [
    "First document content...",
    "Second document content...",
]

metadatas = [
    {"source": "doc1.txt", "author": "Alice", "date": "2024-01-01"},
    {"source": "doc2.txt", "author": "Bob", "date": "2024-01-02"},
]

# Create documents with metadata
documents = splitter.create_documents(texts, metadatas=metadatas)

for doc in documents:
    print(f"Source: {doc.metadata['source']}")
    print(f"Author: {doc.metadata['author']}")
    print(f"Content: {doc.page_content[:50]}...")
    print()
</syntaxhighlight>

=== With Start Index Tracking ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,  # Enable position tracking
)

texts = ["Long document that will be split into multiple chunks..."]
metadatas = [{"source": "original.txt"}]

documents = splitter.create_documents(texts, metadatas=metadatas)

for i, doc in enumerate(documents):
    print(f"Chunk {i}:")
    print(f"  Start Index: {doc.metadata['start_index']}")
    print(f"  Source: {doc.metadata['source']}")
    print(f"  Content: {doc.page_content[:30]}...")
</syntaxhighlight>

=== Processing Multiple Files ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    add_start_index=True,
)

# Load files and prepare metadata
file_paths = list(Path("./docs").glob("*.txt"))
texts = []
metadatas = []

for path in file_paths:
    texts.append(path.read_text())
    metadatas.append({
        "source": str(path),
        "filename": path.name,
        "size": path.stat().st_size,
    })

# Create documents with all metadata
documents = splitter.create_documents(texts, metadatas=metadatas)

print(f"Created {len(documents)} chunks from {len(file_paths)} files")
</syntaxhighlight>

=== Metadata Deep Copy Behavior ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=100)

# Metadata with nested structure
texts = ["A text that will be split into multiple chunks..."]
metadatas = [{"source": "test.txt", "nested": {"key": "original"}}]

documents = splitter.create_documents(texts, metadatas=metadatas)

# Modifying one chunk's metadata doesn't affect others
documents[0].metadata["nested"]["key"] = "modified"

# Other chunks still have original value (deep copy)
if len(documents) > 1:
    print(documents[1].metadata["nested"]["key"])  # Still "original"
</syntaxhighlight>

=== Building Custom Pipeline ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
)


def process_api_response(api_data: list[dict]) -> list:
    """Convert API response data to Document chunks."""
    texts = []
    metadatas = []

    for item in api_data:
        texts.append(item["content"])
        metadatas.append({
            "id": item["id"],
            "title": item["title"],
            "url": item["url"],
            "fetched_at": item["timestamp"],
        })

    return splitter.create_documents(texts, metadatas=metadatas)


# Usage with API response
api_response = [
    {"id": 1, "title": "Article 1", "url": "...", "content": "...", "timestamp": "..."},
    {"id": 2, "title": "Article 2", "url": "...", "content": "...", "timestamp": "..."},
]
documents = process_api_response(api_response)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:langchain-ai_langchain_Metadata_Handling]]

=== Requires Environment ===
* [[requires_env::Environment:langchain-ai_langchain_Text_Splitters_Environment]]
