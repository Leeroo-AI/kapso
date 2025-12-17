= text_splitter_types =

'''Sources:''' /tmp/praxium_repo_wjjl6pl8/libs/text-splitters/langchain_text_splitters/character.py:L11-169, /tmp/praxium_repo_wjjl6pl8/libs/text-splitters/langchain_text_splitters/base.py:L44-89

'''Domains:''' Text Splitting, RAG, Document Processing

'''Last Updated:''' 2025-12-17

== Overview ==

The text_splitter_types implementation provides concrete classes for splitting text into chunks. The main types are CharacterTextSplitter for simple separator-based splitting, RecursiveCharacterTextSplitter for hierarchical splitting with multiple separators, and language-specific variants created via from_language() factory method.

== Description ==

LangChain provides three primary text splitter implementations:

'''CharacterTextSplitter''' splits text using a single separator pattern. It supports both literal separators and regex patterns, with options to keep or discard separators.

'''RecursiveCharacterTextSplitter''' implements a more sophisticated approach that tries multiple separators in order of preference, recursively splitting chunks that are still too large. This is the recommended default for most use cases.

'''Language-specific splitters''' are RecursiveCharacterTextSplitter instances pre-configured with separators appropriate for specific programming languages. These are created using the from_language() class method.

== Code Reference ==

The implementation is found in:
* libs/text-splitters/langchain_text_splitters/character.py - CharacterTextSplitter and RecursiveCharacterTextSplitter classes
* libs/text-splitters/langchain_text_splitters/base.py - TextSplitter base class and TokenTextSplitter

Key classes:
* TextSplitter (base class) - Abstract interface for all text splitters
* CharacterTextSplitter - Simple separator-based splitting
* RecursiveCharacterTextSplitter - Hierarchical multi-separator splitting
* TokenTextSplitter - Token-based splitting using tiktoken

== API ==

=== CharacterTextSplitter ===

<syntaxhighlight lang="python">
from langchain_text_splitters import CharacterTextSplitter

# Basic usage with default separator (double newline)
splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Custom separator
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200
)

# Regex separator
splitter = CharacterTextSplitter(
    separator=r"\n\n+",
    is_separator_regex=True,
    chunk_size=1000,
    chunk_overlap=200
)
</syntaxhighlight>

=== RecursiveCharacterTextSplitter ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Default separators: ["\n\n", "\n", " ", ""]
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Custom separators
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=200
)
</syntaxhighlight>

=== Language-specific Splitting ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Create Python-specific splitter
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=200
)

# Supported languages
languages = [
    Language.PYTHON,
    Language.JS,
    Language.TS,
    Language.JAVA,
    Language.GO,
    Language.RUST,
    Language.MARKDOWN,
    Language.HTML,
    # ... and 20+ more
]
</syntaxhighlight>

=== TokenTextSplitter ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Split by token count using tiktoken
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",  # GPT-4 encoding
    chunk_size=1000,  # tokens, not characters
    chunk_overlap=200
)

# Or with model name
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=1000,
    chunk_overlap=200
)
</syntaxhighlight>

== I/O Contract ==

=== Input ===

All text splitters accept:
* '''text''' (str) - The text to split into chunks
* '''chunk_size''' (int) - Maximum size of each chunk (default: 4000)
* '''chunk_overlap''' (int) - Number of characters/tokens to overlap between chunks (default: 200)

RecursiveCharacterTextSplitter additionally accepts:
* '''separators''' (list[str] | None) - List of separators to try in order
* '''is_separator_regex''' (bool) - Whether separators are regex patterns (default: False)

=== Output ===

All text splitters return:
* '''list[str]''' - List of text chunks that respect the chunk_size constraint

The split_documents() method returns:
* '''list[Document]''' - List of Document objects with page_content and metadata

== Usage Examples ==

=== Example 1: Basic Text Splitting ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Chapter 1: Introduction

This is the first chapter of the document.
It contains multiple paragraphs.

Chapter 2: Methods

This is the second chapter.
It describes the methods used.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)
print(f"Split into {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk[:50]}...")
</syntaxhighlight>

=== Example 2: Language-aware Code Splitting ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

python_code = """
class DataProcessor:
    def __init__(self, data):
        self.data = data

    def process(self):
        results = []
        for item in self.data:
            if item.is_valid():
                results.append(item.transform())
        return results

def main():
    processor = DataProcessor(load_data())
    results = processor.process()
    save_results(results)
"""

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=0
)

chunks = python_splitter.split_text(python_code)
# Each chunk will respect class and function boundaries
</syntaxhighlight>

=== Example 3: Token-based Splitting for LLM Context ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ensure chunks fit in GPT-4's context window
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=4000,  # tokens
    chunk_overlap=200
)

long_text = "..." # Your long document

chunks = splitter.split_text(long_text)
# Each chunk is guaranteed to be <= 4000 tokens
</syntaxhighlight>

=== Example 4: Custom Separator Strategy ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Split markdown with custom priority
markdown_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n## ",      # Try h2 headers first
        "\n### ",     # Then h3 headers
        "\n\n",       # Then paragraphs
        "\n",         # Then lines
        ". ",         # Then sentences
        " ",          # Then words
        ""            # Finally, characters
    ],
    chunk_size=500,
    chunk_overlap=50
)
</syntaxhighlight>

=== Example 5: Splitting Documents with Metadata ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

docs = [
    Document(
        page_content="Chapter 1 content...",
        metadata={"source": "book.pdf", "chapter": 1}
    ),
    Document(
        page_content="Chapter 2 content...",
        metadata={"source": "book.pdf", "chapter": 2}
    )
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Metadata is automatically preserved in each chunk
split_docs = splitter.split_documents(docs)
</syntaxhighlight>

== Related Pages ==

* [[langchain-ai_langchain_Splitter_Selection|Splitter_Selection]] - Principle of choosing appropriate splitters
* [[langchain-ai_langchain_chunk_parameters|chunk_parameters]] - Configuring chunk size and overlap parameters
* [[langchain-ai_langchain_separator_config|separator_config]] - Customizing separator patterns for languages
* [[langchain-ai_langchain_split_text_method|split_text_method]] - Using the split methods on text and documents
