{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Text Splitters|https://docs.langchain.com/oss/python/langchain/overview]]
|-
! Domains
| [[domain::NLP]], [[domain::RAG]], [[domain::Document_Processing]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Concrete tool for executing text splitting operations on raw text or Document objects, provided by LangChain text-splitters.

=== Description ===

`TextSplitter.split_text` and `split_documents` are the core execution methods that transform text into chunks:
* **split_text:** Takes a string, returns list of string chunks
* **split_documents:** Takes Document objects, returns Document objects with preserved metadata

The internal `_split_text` method (in RecursiveCharacterTextSplitter) implements the recursive splitting algorithm.

=== Usage ===

Use these methods when:
* Processing raw text for embedding
* Splitting loaded documents for RAG
* Chunking code files
* Preparing text for LLM context

Choose the right method:
* `split_text`: Raw strings, no metadata needed
* `split_documents`: Document objects, metadata preservation required

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain]
* '''File:''' libs/text-splitters/langchain_text_splitters/base.py
* '''Lines:''' L87-117 (split_documents)
* '''File:''' libs/text-splitters/langchain_text_splitters/character.py
* '''Lines:''' L100-151 (_split_text recursive algorithm)

=== Signature ===
<syntaxhighlight lang="python">
class TextSplitter(ABC):
    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """Split text into multiple components.

        Args:
            text: The input text to split

        Returns:
            List of text chunks
        """

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        """Split documents into smaller Document chunks.

        Args:
            documents: Iterable of Document objects to split

        Returns:
            List of Document objects with chunked page_content
            and preserved/extended metadata
        """


class RecursiveCharacterTextSplitter(TextSplitter):
    def split_text(self, text: str) -> list[str]:
        """Split text using recursive separator strategy."""
        return self._split_text(text, self._separators)

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Internal recursive splitting implementation.

        Args:
            text: Text to split
            separators: Remaining separators to try

        Returns:
            List of chunks respecting size constraints
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
</syntaxhighlight>

== I/O Contract ==

=== Inputs (split_text) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| text || str || Yes || Raw text to split into chunks
|}

=== Inputs (split_documents) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| documents || Iterable[Document] || Yes || Document objects with page_content and metadata
|}

=== Outputs (split_text) ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return || list[str] || List of text chunks
|}

=== Outputs (split_documents) ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return || list[Document] || Document objects with chunked content and preserved metadata
|}

== Usage Examples ==

=== Splitting Raw Text ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

text = """
Introduction

This is the first paragraph of a longer document.
It contains important information about the topic.

Main Content

The main body of the document goes here.
It may span multiple paragraphs and sections.

Conclusion

Final thoughts and summary go in this section.
"""

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i} ({len(chunk)} chars):")
    print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
    print()
</syntaxhighlight>

=== Splitting Document Objects ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,  # Track position in original
)

# Documents with metadata
documents = [
    Document(
        page_content="Long content from first file...",
        metadata={"source": "file1.txt", "author": "Alice"}
    ),
    Document(
        page_content="Long content from second file...",
        metadata={"source": "file2.txt", "author": "Bob"}
    ),
]

# Split preserves metadata
split_docs = splitter.split_documents(documents)

for doc in split_docs:
    print(f"Source: {doc.metadata['source']}")
    print(f"Start Index: {doc.metadata.get('start_index', 'N/A')}")
    print(f"Content: {doc.page_content[:50]}...")
    print()
</syntaxhighlight>

=== Processing Files ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

# Load and split multiple files
def load_and_split_files(file_paths: list[Path]) -> list[Document]:
    documents = []
    for path in file_paths:
        content = path.read_text()
        doc = Document(
            page_content=content,
            metadata={"source": str(path), "file_type": path.suffix}
        )
        documents.append(doc)

    return splitter.split_documents(documents)

# Usage
files = list(Path("./docs").glob("*.md"))
chunks = load_and_split_files(files)
</syntaxhighlight>

=== Batch Processing ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

# Split multiple texts
texts = [
    "First document content...",
    "Second document content...",
    "Third document content...",
]

# Split each text
all_chunks = []
for i, text in enumerate(texts):
    chunks = splitter.split_text(text)
    # Add source tracking
    for j, chunk in enumerate(chunks):
        all_chunks.append({
            "text": chunk,
            "doc_id": i,
            "chunk_id": j,
        })

print(f"Total chunks: {len(all_chunks)}")
</syntaxhighlight>

=== Recursive Splitting in Action ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Small chunk size to demonstrate recursion
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=100,
    chunk_overlap=20,
)

text = """First paragraph is short.

Second paragraph is much longer and will need to be split because it exceeds our small chunk size limit that we set for demonstration purposes.

Third paragraph.
"""

chunks = splitter.split_text(text)

# Observe how the splitter:
# 1. First tries to split by paragraphs (\n\n)
# 2. If paragraph too long, falls back to lines (\n)
# 3. If line too long, falls back to words ( )
# 4. If word too long, falls back to characters ("")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {repr(chunk)}")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:langchain-ai_langchain_Document_Transformation]]

=== Requires Environment ===
* [[requires_env::Environment:langchain-ai_langchain_Text_Splitters_Environment]]
