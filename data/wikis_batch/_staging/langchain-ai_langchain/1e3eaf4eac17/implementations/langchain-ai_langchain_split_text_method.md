= split_text_method =

'''Sources:''' /tmp/praxium_repo_wjjl6pl8/libs/text-splitters/langchain_text_splitters/character.py:L25-151, /tmp/praxium_repo_wjjl6pl8/libs/text-splitters/langchain_text_splitters/base.py:L87-117

'''Domains:''' Text Splitting, RAG, Document Processing, Information Retrieval

'''Last Updated:''' 2025-12-17

== Overview ==

The split_text_method implementation provides methods for splitting text and documents into chunks. The primary methods are split_text() for splitting plain strings and split_documents() for splitting Document objects while preserving metadata. These methods form the core interface of all LangChain text splitters.

== Description ==

All LangChain text splitters implement two primary splitting methods:

'''split_text(text: str) -> list[str]:''' Splits a plain string into a list of smaller string chunks. This is the fundamental operation implemented by each splitter type.

'''split_documents(documents: Iterable[Document]) -> list[Document]:''' Splits a list of Document objects into a list of smaller Document chunks. Metadata from the original documents is preserved in each chunk.

Additionally, there's a helper method:

'''create_documents(texts: list[str], metadatas: list[dict] | None) -> list[Document]:''' Creates Document objects from text chunks, optionally with metadata.

The implementation follows a consistent pattern across all splitter types, with the specific splitting logic defined in each splitter's split_text() method.

== Code Reference ==

The implementation is in libs/text-splitters/langchain_text_splitters/base.py:

<syntaxhighlight lang="python">
class TextSplitter(BaseDocumentTransformer, ABC):
    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """Split text into multiple components."""

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
                metadata = copy.deepcopy(metadatas_[i])
                if self._add_start_index:
                    offset = index + previous_chunk_len - self._chunk_overlap
                    index = text.find(chunk, max(0, offset))
                    metadata["start_index"] = index
                    previous_chunk_len = len(chunk)
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents
</syntaxhighlight>

RecursiveCharacterTextSplitter implementation:

<syntaxhighlight lang="python">
class RecursiveCharacterTextSplitter(TextSplitter):
    def split_text(self, text: str) -> list[str]:
        """Split the input text into smaller chunks based on predefined separators."""
        return self._split_text(text, self._separators)

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Split incoming text and return chunks."""
        # Recursive splitting logic
        # Try separators in order, recursively split if needed
</syntaxhighlight>

== API ==

=== split_text() - Basic Text Splitting ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

text = "Your long text here..."
chunks = splitter.split_text(text)
# Returns: list[str]
</syntaxhighlight>

=== split_documents() - Document Splitting with Metadata ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = [
    Document(
        page_content="Long text...",
        metadata={"source": "file.txt", "page": 1}
    )
]

split_docs = splitter.split_documents(docs)
# Returns: list[Document]
# Each chunk preserves original metadata
</syntaxhighlight>

=== create_documents() - Creating Documents from Text ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

texts = ["Text 1", "Text 2", "Text 3"]
metadatas = [
    {"source": "doc1.txt"},
    {"source": "doc2.txt"},
    {"source": "doc3.txt"}
]

docs = splitter.create_documents(texts, metadatas=metadatas)
# Returns: list[Document]
</syntaxhighlight>

== I/O Contract ==

=== split_text() ===

'''Input:'''
* text (str) - The text to split

'''Output:'''
* list[str] - List of text chunks

'''Behavior:'''
* Chunks respect chunk_size constraint
* Adjacent chunks overlap by chunk_overlap
* Separators are handled according to keep_separator setting

=== split_documents() ===

'''Input:'''
* documents (Iterable[Document]) - Documents to split

'''Output:'''
* list[Document] - List of document chunks

'''Behavior:'''
* Extracts text and metadata from each document
* Splits text using split_text()
* Creates new Document for each chunk
* Preserves original metadata in each chunk
* Optionally adds start_index to metadata if add_start_index=True

=== create_documents() ===

'''Input:'''
* texts (list[str]) - List of texts to create documents from
* metadatas (list[dict] | None) - Optional metadata for each text

'''Output:'''
* list[Document] - List of document chunks

'''Behavior:'''
* Splits each text using split_text()
* Creates Document objects for each chunk
* Attaches corresponding metadata to each chunk
* If add_start_index=True, adds start_index to metadata

== Usage Examples ==

=== Example 1: Basic Text Splitting ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Artificial Intelligence has revolutionized many fields. Machine learning,
a subset of AI, enables computers to learn from data without explicit programming.

Deep learning, which uses neural networks with multiple layers, has achieved
remarkable success in image recognition, natural language processing, and
game playing. These advances have opened new possibilities in healthcare,
finance, and autonomous systems.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30
)

chunks = splitter.split_text(text)

print(f"Split into {len(chunks)} chunks:")
for i, chunk in enumerate(chunks, 1):
    print(f"\nChunk {i} ({len(chunk)} chars):")
    print(chunk)
</syntaxhighlight>

=== Example 2: Splitting Documents with Metadata ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Create documents with metadata
docs = [
    Document(
        page_content="Chapter 1: Introduction to AI. Artificial Intelligence...",
        metadata={"source": "ai_book.pdf", "chapter": 1, "page": 1}
    ),
    Document(
        page_content="Chapter 2: Machine Learning. Machine learning is...",
        metadata={"source": "ai_book.pdf", "chapter": 2, "page": 15}
    )
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

split_docs = splitter.split_documents(docs)

print(f"Split {len(docs)} documents into {len(split_docs)} chunks")
for doc in split_docs:
    print(f"\nMetadata: {doc.metadata}")
    print(f"Content: {doc.page_content[:50]}...")
</syntaxhighlight>

=== Example 3: Tracking Chunk Positions ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

text = "First paragraph here. Second paragraph here. Third paragraph here."

# Enable start index tracking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=30,
    chunk_overlap=5,
    add_start_index=True
)

docs = splitter.create_documents([text])

for doc in docs:
    start_idx = doc.metadata.get("start_index", "N/A")
    print(f"Position {start_idx}: {doc.page_content}")
</syntaxhighlight>

=== Example 4: Code Splitting ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

python_code = """
def calculate_statistics(data):
    '''Calculate mean and standard deviation.'''
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std_dev = variance ** 0.5
    return mean, std_dev

def normalize_data(data):
    '''Normalize data to zero mean and unit variance.'''
    mean, std_dev = calculate_statistics(data)
    return [(x - mean) / std_dev for x in data]

class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.normalized = None

    def process(self):
        self.normalized = normalize_data(self.data)
        return self.normalized
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=30
)

chunks = splitter.split_text(python_code)

print(f"Split code into {len(chunks)} chunks")
for i, chunk in enumerate(chunks, 1):
    print(f"\n=== Chunk {i} ===")
    print(chunk)
</syntaxhighlight>

=== Example 5: Batch Processing Multiple Texts ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

texts = [
    "Content from document 1...",
    "Content from document 2...",
    "Content from document 3..."
]

metadatas = [
    {"source": "doc1.txt", "author": "Alice"},
    {"source": "doc2.txt", "author": "Bob"},
    {"source": "doc3.txt", "author": "Charlie"}
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

# Create documents from texts and metadata
docs = splitter.create_documents(texts, metadatas=metadatas)

print(f"Created {len(docs)} document chunks from {len(texts)} texts")
for doc in docs:
    print(f"Source: {doc.metadata['source']}, Author: {doc.metadata['author']}")
</syntaxhighlight>

=== Example 6: Markdown Document Splitting ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document

markdown_content = """
# User Guide

## Installation

To install the package, run:

```bash
pip install mypackage
```

## Quick Start

Import the main class:

```python
from mypackage import MyClass
obj = MyClass()
```

## Advanced Usage

For advanced features, see the documentation.
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=150,
    chunk_overlap=30
)

doc = Document(
    page_content=markdown_content,
    metadata={"source": "user_guide.md", "type": "documentation"}
)

chunks = splitter.split_documents([doc])

print(f"Split markdown into {len(chunks)} chunks")
for i, chunk in enumerate(chunks, 1):
    print(f"\n=== Chunk {i} ===")
    print(f"Source: {chunk.metadata['source']}")
    print(chunk.page_content)
</syntaxhighlight>

=== Example 7: Custom Separator with Document Splitting ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Document with custom section markers
content = """
[SECTION: Overview]
This is the overview section with general information.

[SECTION: Details]
This section contains detailed information about the topic.

[SECTION: Conclusion]
Final thoughts and conclusions.
"""

splitter = RecursiveCharacterTextSplitter(
    separators=["\n[SECTION:", "\n\n", "\n", " ", ""],
    chunk_size=100,
    chunk_overlap=20,
    keep_separator=True
)

doc = Document(
    page_content=content,
    metadata={"source": "report.txt", "date": "2025-12-17"}
)

chunks = splitter.split_documents([doc])

for chunk in chunks:
    print(f"\nMetadata: {chunk.metadata}")
    print(f"Content preview: {chunk.page_content[:60]}...")
</syntaxhighlight>

== Related Pages ==

* [[langchain-ai_langchain_Document_Splitting|Document_Splitting]] - Principle of document splitting
* [[langchain-ai_langchain_document_metadata|document_metadata]] - Metadata preservation during splitting
* [[langchain-ai_langchain_text_splitter_types|text_splitter_types]] - Different text splitter implementations
* [[langchain-ai_langchain_chunk_parameters|chunk_parameters]] - Configuring chunk size and overlap
