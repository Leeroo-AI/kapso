---
title: "KonlpyTextSplitter"
repository: "langchain-ai/langchain"
commit_hash: "1e3eaf4eac17"
file_path: "libs/text-splitters/langchain_text_splitters/konlpy.py"
component_type: "Text Splitter"
component_name: "KonlpyTextSplitter"
layer: "Text Splitters"
language_support: "Korean"
---

# KonlpyTextSplitter

## Overview

The `KonlpyTextSplitter` is a specialized text splitter designed specifically for Korean language text. It leverages the Konlpy library's Kkma (Korean Knowledge Morphology Analyzer) to perform linguistically-aware sentence segmentation, making it ideal for processing Korean documents, articles, and conversational text.

**Key Features:**
- Korean-specific sentence boundary detection using Konlpy's Kkma analyzer
- Linguistically accurate splitting that respects Korean grammar and sentence structure
- Inherits text merging and chunk management from base TextSplitter
- Configurable separator for joining sentences
- Automatic import checking with helpful error messages

**Location:** `/tmp/praxium_repo_wjjl6pl8/libs/text-splitters/langchain_text_splitters/konlpy.py`

## Code Reference

### Class Definition

```python
class KonlpyTextSplitter(TextSplitter):
    """Splitting text using Konlpy package.

    It is good for splitting Korean text.
    """

    def __init__(
        self,
        separator: str = "\n\n",
        **kwargs: Any,
    ) -> None:
        """Initialize the Konlpy text splitter.

        Args:
            separator: The separator to use when merging splits
            **kwargs: Additional arguments passed to TextSplitter base class
                (chunk_size, chunk_overlap, length_function, etc.)

        Raises:
            ImportError: If konlpy package is not installed
        """
```

### Public Methods

```python
def split_text(self, text: str) -> list[str]:
    """Split incoming text and return chunks.

    Uses Konlpy's Kkma analyzer to detect sentence boundaries,
    then merges sentences according to chunk_size and separator.

    Args:
        text: Korean text to split

    Returns:
        List of text chunks split at Korean sentence boundaries
    """
```

### Internal Components

```python
# Package availability check
_HAS_KONLPY = True  # Set to False if import fails

# Kkma analyzer instance (initialized in __init__)
self.kkma = konlpy.tag.Kkma()
```

## I/O Contract

### Input

**Constructor Parameters:**
- `separator` (str, default: `"\n\n"`): String to use when joining sentences into chunks
- `**kwargs` (Any): Additional parameters for TextSplitter base class:
  - `chunk_size` (int): Maximum size of chunks
  - `chunk_overlap` (int): Overlap between chunks
  - `length_function` (Callable): Function to measure text length
  - `keep_separator` (bool): Whether to keep separators in output

**split_text() Parameters:**
- `text` (str): Korean language text to split into chunks

### Output

**split_text() returns:** `list[str]`
- List of text chunks
- Each chunk respects Korean sentence boundaries
- Chunks merged according to chunk_size and separator settings

### Dependencies

**Required Package:**
- `konlpy`: Korean language processing library
- Installation: `pip install konlpy`
- Also requires JDK/JRE for Java-based analyzers

**System Requirements:**
- Java Development Kit (JDK) or Java Runtime Environment (JRE)
- Konlpy requires Java backend for Korean morphological analysis

## Usage Examples

### Basic Korean Text Splitting

```python
from langchain_text_splitters import KonlpyTextSplitter

korean_text = """
한국어는 매우 아름다운 언어입니다. 많은 사람들이 한국어를 배우고 있습니다.
한국어 문장 분리는 영어와 다른 규칙을 따릅니다. 이 텍스트 분할기는 한국어 문장을 정확하게 인식합니다.
자연어 처리에서 올바른 문장 분리는 매우 중요합니다.
"""

# Initialize splitter
splitter = KonlpyTextSplitter(
    separator=" ",
    chunk_size=100,
    chunk_overlap=20
)

# Split the text
chunks = splitter.split_text(korean_text)

for i, chunk in enumerate(chunks):
    print(f"청크 {i+1}:")
    print(chunk)
    print(f"길이: {len(chunk)} 문자\n")
```

### Processing Korean Documents

```python
from langchain_text_splitters import KonlpyTextSplitter
from langchain_core.documents import Document

# Korean document
korean_article = """
서울은 대한민국의 수도이자 최대 도시입니다. 인구는 약 천만 명입니다.
서울에는 많은 역사적 유적지가 있습니다. 경복궁, 창덕궁, 덕수궁 등이 유명합니다.
현대적인 건축물도 많이 있습니다. 롯데타워는 서울의 랜드마크 중 하나입니다.
"""

# Create document
doc = Document(
    page_content=korean_article,
    metadata={"source": "korean_tourism", "language": "ko"}
)

# Split document
splitter = KonlpyTextSplitter(chunk_size=200)
split_docs = splitter.split_documents([doc])

for i, doc in enumerate(split_docs):
    print(f"문서 {i+1}:")
    print(f"내용: {doc.page_content}")
    print(f"메타데이터: {doc.metadata}\n")
```

### Custom Separator Configuration

```python
# Use different separators for different use cases

# Paragraph-style splitting
paragraph_splitter = KonlpyTextSplitter(
    separator="\n\n",  # Double newline between chunks
    chunk_size=500
)

# Sentence-style splitting
sentence_splitter = KonlpyTextSplitter(
    separator=" ",  # Single space between sentences
    chunk_size=200
)

# No separator (concatenate directly)
compact_splitter = KonlpyTextSplitter(
    separator="",
    chunk_size=150
)
```

### Error Handling for Missing Dependencies

```python
try:
    from langchain_text_splitters import KonlpyTextSplitter
    splitter = KonlpyTextSplitter()
except ImportError as e:
    print("Konlpy is not installed.")
    print("Install with: pip install konlpy")
    print("Also ensure Java JDK/JRE is installed for Konlpy backend")
    # Fallback to alternative splitter
    from langchain_text_splitters import CharacterTextSplitter
    splitter = CharacterTextSplitter()
```

### Integration with Korean RAG System

```python
from langchain_text_splitters import KonlpyTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Korean knowledge base
korean_docs = [
    "한국의 전통 음식에는 김치, 불고기, 비빔밥이 있습니다.",
    "한글은 세종대왕이 창제한 문자입니다.",
    "한국의 4계절은 봄, 여름, 가을, 겨울입니다."
]

# Create documents
documents = [
    Document(page_content=text, metadata={"topic": "korean_culture"})
    for text in korean_docs
]

# Split with Korean-aware splitter
splitter = KonlpyTextSplitter(chunk_size=100, chunk_overlap=10)
split_docs = splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embeddings)

# Query in Korean
results = vectorstore.similarity_search("한국 음식은 무엇이 있나요?")
for result in results:
    print(result.page_content)
```

### Batch Processing Korean Files

```python
from pathlib import Path
from langchain_text_splitters import KonlpyTextSplitter

def process_korean_files(directory: str):
    splitter = KonlpyTextSplitter(chunk_size=300, chunk_overlap=50)
    results = {}

    for filepath in Path(directory).rglob("*.txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            korean_text = f.read()

        chunks = splitter.split_text(korean_text)
        results[str(filepath)] = {
            "chunks": chunks,
            "num_chunks": len(chunks),
            "avg_length": sum(len(c) for c in chunks) / len(chunks)
        }

    return results

# Process all Korean text files
korean_chunks = process_korean_files("./korean_documents")
```

## Implementation Details

### Kkma Sentence Tokenization

The core functionality relies on Konlpy's Kkma analyzer:

```python
self.kkma = konlpy.tag.Kkma()
splits = self.kkma.sentences(text)
```

**Kkma (Korean Knowledge Morphology Analyzer):**
- Performs morphological analysis of Korean text
- Identifies sentence boundaries based on Korean grammar rules
- Handles Korean punctuation (。, ., ?, !, etc.)
- Recognizes Korean sentence-ending particles (다, 요, 니다, etc.)

### Merge Strategy

After sentence tokenization, the base class's `_merge_splits()` method is called:

```python
return self._merge_splits(splits, self._separator)
```

This method:
1. Takes the list of sentences from Kkma
2. Merges them into chunks respecting `chunk_size`
3. Adds `chunk_overlap` if configured
4. Joins sentences with `self._separator`

### Import Checking

The module uses a try-except pattern for optional dependency:

```python
try:
    import konlpy
    _HAS_KONLPY = True
except ImportError:
    _HAS_KONLPY = False
```

Then checks at initialization:
```python
if not _HAS_KONLPY:
    msg = """
        Konlpy is not installed, please install it with
        `pip install konlpy`
        """
    raise ImportError(msg)
```

## Design Patterns

### Template Method Pattern
Implements `split_text()` while relying on parent class `_merge_splits()` for chunk management.

### Strategy Pattern
Provides Korean-specific splitting strategy within the TextSplitter framework.

### Dependency Injection
Kkma analyzer is injected as instance variable, allowing future flexibility for different Korean analyzers.

## Related Components

### Parent Class
- `langchain_text_splitters.base.TextSplitter` - Base class providing chunk merging and document splitting

### Related Splitters
- `langchain_text_splitters.NLTKTextSplitter` - English sentence splitting
- `langchain_text_splitters.SpacyTextSplitter` - Multi-language splitting with spaCy
- `langchain_text_splitters.CharacterTextSplitter` - Simple character-based splitting (fallback)

### External Dependencies
- **Konlpy**: Korean NLP library (https://konlpy.org/)
- **Kkma**: Korean morphological analyzer within Konlpy
- **Java**: Required backend for Konlpy

### Use Cases
- Korean document processing for RAG systems
- Korean chatbot context management
- Korean text summarization
- Korean question-answering systems

## Testing Considerations

### Unit Tests

```python
def test_korean_sentence_splitting():
    text = "첫 번째 문장입니다. 두 번째 문장입니다."
    splitter = KonlpyTextSplitter(chunk_size=100)
    chunks = splitter.split_text(text)
    assert len(chunks) > 0

def test_chunk_size_respected():
    long_text = "문장입니다. " * 100
    splitter = KonlpyTextSplitter(chunk_size=50)
    chunks = splitter.split_text(long_text)
    for chunk in chunks:
        assert len(chunk) <= 50 + 20  # Allow some overflow

def test_separator_usage():
    text = "첫 문장. 둘째 문장."
    splitter = KonlpyTextSplitter(separator=" | ")
    result = splitter.split_text(text)
    # Verify separator used in output
```

### Integration Tests
- Test with various Korean texts (news, conversation, formal documents)
- Verify compatibility with different Korean encodings (UTF-8, EUC-KR)
- Test with mixed Korean-English text

### Edge Cases
- Empty strings
- Single character input
- Text without sentence-ending punctuation
- Very long sentences exceeding chunk_size
- Mixed Korean-English text
- Korean honorifics and formal endings

## Performance Considerations

### Time Complexity
- O(n) for Kkma sentence tokenization where n is text length
- O(m) for merging splits where m is number of sentences
- Kkma analysis is computationally expensive due to morphological processing

### Space Complexity
- O(m) where m is number of detected sentences
- Kkma maintains internal state for morphological analysis

### Optimization Tips
1. **Reuse splitter instances** - Kkma initialization is expensive
2. **Batch processing** - Process multiple texts with same splitter instance
3. **Adjust chunk_size** - Larger chunks = fewer Kkma calls on merge
4. **Consider caching** - Cache split results for frequently accessed texts

## Best Practices

1. **Install all dependencies**: Ensure both Konlpy and Java are installed
2. **Use UTF-8 encoding**: Always use UTF-8 for Korean text files
3. **Set appropriate chunk_size**: Consider Korean character density (typically 2-3x bytes per character)
4. **Test with representative data**: Korean text varies significantly by domain (formal, casual, technical)
5. **Handle mixed content**: Decide strategy for Korean-English mixed text

## Common Pitfalls

1. **Missing Java dependency**: Konlpy requires Java but error message may be unclear
2. **Encoding issues**: Using wrong encoding for Korean files leads to corruption
3. **Chunk size miscalculation**: Korean characters are multi-byte in UTF-8
4. **Performance issues**: Kkma is slower than simple splitting; don't use for huge documents without batching
5. **Separator confusion**: Korean text may already contain separators that conflict with configured separator

## Installation Guide

### Step 1: Install Java
```bash
# Ubuntu/Debian
sudo apt-get install default-jdk

# macOS
brew install openjdk

# Windows
# Download and install from https://www.oracle.com/java/technologies/downloads/
```

### Step 2: Install Konlpy
```bash
pip install konlpy
```

### Step 3: Verify Installation
```python
try:
    from konlpy.tag import Kkma
    kkma = Kkma()
    print("Konlpy installed successfully!")
except Exception as e:
    print(f"Installation issue: {e}")
```

## Version History

- Part of `langchain-text-splitters` package
- Requires Konlpy 0.4.0 or higher
- Compatible with Python 3.8+
