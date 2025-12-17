---
title: "NLTKTextSplitter"
repository: "langchain-ai/langchain"
commit_hash: "1e3eaf4eac17"
file_path: "libs/text-splitters/langchain_text_splitters/nltk.py"
component_type: "Text Splitter"
component_name: "NLTKTextSplitter"
layer: "Text Splitters"
language_support: "Multi-language"
---

# NLTKTextSplitter

## Overview

The `NLTKTextSplitter` is a text splitter that uses the Natural Language Toolkit (NLTK) for sentence boundary detection. It supports multiple languages and offers two tokenization modes: standard sentence tokenization and span-based tokenization that preserves whitespace. This makes it ideal for processing natural language text where sentence boundaries need to be linguistically accurate.

**Key Features:**
- Multi-language sentence tokenization using NLTK
- Support for 17+ languages via NLTK's Punkt tokenizer
- Two tokenization modes: standard and span-based (preserves whitespace)
- Configurable separator for joining sentences
- Inherits chunk management from base TextSplitter class
- Automatic import checking with helpful error messages

**Location:** `/tmp/praxium_repo_wjjl6pl8/libs/text-splitters/langchain_text_splitters/nltk.py`

## Code Reference

### Class Definition

```python
class NLTKTextSplitter(TextSplitter):
    """Splitting text using NLTK package."""

    def __init__(
        self,
        separator: str = "\n\n",
        language: str = "english",
        *,
        use_span_tokenize: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the NLTK splitter.

        Args:
            separator: The separator to use when merging splits
            language: Language for NLTK tokenizer (e.g., 'english', 'spanish', 'german')
            use_span_tokenize: If True, use span tokenization to preserve whitespace
            **kwargs: Additional arguments passed to TextSplitter base class

        Raises:
            ImportError: If NLTK is not installed
            ValueError: If use_span_tokenize is True but separator is not empty
        """
```

### Public Methods

```python
def split_text(self, text: str) -> list[str]:
    """Split incoming text and return chunks.

    Uses NLTK's sentence tokenizer for the specified language,
    then merges sentences according to chunk_size and separator.

    Args:
        text: Text to split into sentence-based chunks

    Returns:
        List of text chunks split at sentence boundaries
    """
```

### Internal Components

```python
# Package availability check
_HAS_NLTK = True  # Set to False if import fails

# Tokenizer (initialized in __init__)
# Either Punkt tokenizer for span_tokenize or sent_tokenize function
self._tokenizer = nltk.tokenize._get_punkt_tokenizer(self._language)  # span mode
# OR
self._tokenizer = nltk.tokenize.sent_tokenize  # standard mode
```

## I/O Contract

### Input

**Constructor Parameters:**
- `separator` (str, default: `"\n\n"`): String to use when joining sentences into chunks
- `language` (str, default: `"english"`): Language for NLTK tokenizer
- `use_span_tokenize` (bool, default: `False`): Use span tokenization to preserve whitespace
- `**kwargs` (Any): Additional parameters for TextSplitter base class:
  - `chunk_size` (int): Maximum size of chunks
  - `chunk_overlap` (int): Overlap between chunks
  - `length_function` (Callable): Function to measure text length

**split_text() Parameters:**
- `text` (str): Text to split into chunks

### Output

**split_text() returns:** `list[str]`
- List of text chunks
- Each chunk respects sentence boundaries
- Chunks merged according to chunk_size and separator settings

### Supported Languages

NLTK's Punkt tokenizer supports:
- `english`, `portuguese`, `spanish`, `german`, `french`, `italian`, `dutch`
- `polish`, `czech`, `danish`, `estonian`, `finnish`, `greek`, `norwegian`
- `slovene`, `swedish`, `turkish`, `russian`

### Dependencies

**Required Package:**
- `nltk`: Natural Language Toolkit
- Installation: `pip install nltk`
- May require downloading Punkt tokenizer data: `nltk.download('punkt')`

## Usage Examples

### Basic English Text Splitting

```python
from langchain_text_splitters import NLTKTextSplitter

text = """
Natural language processing (NLP) is a field of artificial intelligence.
It focuses on the interaction between computers and humans.
NLP enables computers to understand, interpret, and generate human language.
The applications of NLP are vast and growing rapidly.
"""

# Initialize splitter
splitter = NLTKTextSplitter(
    separator=" ",
    chunk_size=100,
    chunk_overlap=20
)

# Split the text
chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(chunk)
    print(f"Length: {len(chunk)} chars\n")
```

### Multi-Language Support

```python
from langchain_text_splitters import NLTKTextSplitter

# Spanish text
spanish_text = """
La inteligencia artificial es fascinante. Tiene muchas aplicaciones prácticas.
El procesamiento del lenguaje natural es un área importante.
Permite que las computadoras entiendan el español.
"""

spanish_splitter = NLTKTextSplitter(language="spanish", chunk_size=150)
spanish_chunks = spanish_splitter.split_text(spanish_text)

# German text
german_text = """
Künstliche Intelligenz ist revolutionär. Sie verändert unsere Welt.
Die natürliche Sprachverarbeitung ist besonders interessant.
Computer können jetzt Deutsch verstehen.
"""

german_splitter = NLTKTextSplitter(language="german", chunk_size=150)
german_chunks = german_splitter.split_text(german_text)

# French text
french_text = """
L'intelligence artificielle est incroyable. Elle a de nombreuses applications.
Le traitement du langage naturel est crucial.
Les ordinateurs peuvent maintenant comprendre le français.
"""

french_splitter = NLTKTextSplitter(language="french", chunk_size=150)
french_chunks = french_splitter.split_text(french_text)
```

### Span Tokenization (Preserving Whitespace)

```python
from langchain_text_splitters import NLTKTextSplitter

text = """First sentence.    Second sentence with extra spaces.

Third sentence after blank line.
Fourth sentence."""

# Standard tokenization (separator required)
standard_splitter = NLTKTextSplitter(
    separator=" ",
    use_span_tokenize=False
)
standard_result = standard_splitter.split_text(text)

# Span tokenization (preserves original whitespace)
span_splitter = NLTKTextSplitter(
    separator="",  # Must be empty for span tokenization
    use_span_tokenize=True
)
span_result = span_splitter.split_text(text)

print("Standard:", standard_result)
print("Span:", span_result)
# Span mode preserves the exact whitespace between sentences
```

### Processing Documents with Metadata

```python
from langchain_text_splitters import NLTKTextSplitter
from langchain_core.documents import Document

# Create documents
documents = [
    Document(
        page_content="First paragraph. It has multiple sentences. They will be split.",
        metadata={"source": "doc1.txt", "page": 1}
    ),
    Document(
        page_content="Second paragraph. This one is longer. It contains more information.",
        metadata={"source": "doc2.txt", "page": 1}
    )
]

# Split documents
splitter = NLTKTextSplitter(chunk_size=100, chunk_overlap=10)
split_docs = splitter.split_documents(documents)

for i, doc in enumerate(split_docs):
    print(f"Document {i+1}:")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}\n")
```

### Error Handling and Fallback

```python
import nltk

def create_nltk_splitter():
    try:
        # Try to create NLTK splitter
        from langchain_text_splitters import NLTKTextSplitter

        # Ensure Punkt tokenizer is downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading Punkt tokenizer...")
            nltk.download('punkt')

        return NLTKTextSplitter(chunk_size=500)

    except ImportError:
        print("NLTK not installed. Using fallback splitter.")
        from langchain_text_splitters import CharacterTextSplitter
        return CharacterTextSplitter(
            separator=". ",
            chunk_size=500
        )

splitter = create_nltk_splitter()
```

### Batch Processing Multiple Files

```python
from pathlib import Path
from langchain_text_splitters import NLTKTextSplitter

def process_text_files(directory: str, language: str = "english"):
    splitter = NLTKTextSplitter(
        language=language,
        chunk_size=500,
        chunk_overlap=50
    )

    results = {}

    for filepath in Path(directory).rglob("*.txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = splitter.split_text(text)
        results[str(filepath)] = {
            "num_chunks": len(chunks),
            "avg_chunk_size": sum(len(c) for c in chunks) / len(chunks),
            "chunks": chunks
        }

        print(f"{filepath.name}: {len(chunks)} chunks")

    return results

# Process all text files
results = process_text_files("./documents", language="english")
```

### Integration with Vector Store

```python
from langchain_text_splitters import NLTKTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Sample documents
docs = [
    Document(page_content="""
        Machine learning is a subset of AI. It enables computers to learn from data.
        Deep learning is a more advanced form. It uses neural networks with many layers.
    """, metadata={"topic": "AI"}),
    Document(page_content="""
        Natural language processing helps computers understand text.
        It has applications in translation, sentiment analysis, and chatbots.
        NLTK is a popular library for NLP tasks.
    """, metadata={"topic": "NLP"})
]

# Split documents
splitter = NLTKTextSplitter(chunk_size=200, chunk_overlap=20)
split_docs = splitter.split_documents(docs)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embeddings)

# Search
results = vectorstore.similarity_search("What is deep learning?", k=2)
for result in results:
    print(result.page_content)
    print(result.metadata)
```

## Implementation Details

### Standard Tokenization Mode

Uses NLTK's `sent_tokenize` function:

```python
self._tokenizer = nltk.tokenize.sent_tokenize
splits = self._tokenizer(text, language=self._language)
```

This:
1. Detects sentence boundaries using Punkt algorithm
2. Returns list of sentences
3. Whitespace between sentences is normalized

### Span Tokenization Mode

Uses NLTK's Punkt tokenizer directly:

```python
self._tokenizer = nltk.tokenize._get_punkt_tokenizer(self._language)
spans = list(self._tokenizer.span_tokenize(text))
```

Then reconstructs sentences preserving whitespace:

```python
splits = []
for i, (start, end) in enumerate(spans):
    if i > 0:
        prev_end = spans[i - 1][1]
        sentence = text[prev_end:start] + text[start:end]
    else:
        sentence = text[start:end]
    splits.append(sentence)
```

This preserves:
- Multiple spaces between sentences
- Newlines and blank lines
- Original formatting

### Validation

The constructor validates parameter combinations:

```python
if self._use_span_tokenize and self._separator:
    msg = "When use_span_tokenize is True, separator should be ''"
    raise ValueError(msg)
```

### Merge Strategy

After tokenization, sentences are merged:

```python
return self._merge_splits(splits, self._separator)
```

The base class's `_merge_splits()` method:
1. Combines sentences until chunk_size is reached
2. Adds chunk_overlap if configured
3. Joins sentences with the separator

## Design Patterns

### Template Method Pattern
Implements `split_text()` while delegating chunk management to parent class.

### Strategy Pattern
Provides two tokenization strategies (standard vs span) selected at initialization.

### Factory Pattern
Tokenizer selection based on `use_span_tokenize` flag.

## Related Components

### Parent Class
- `langchain_text_splitters.base.TextSplitter` - Base class for all text splitters

### Related Splitters
- `langchain_text_splitters.SpacyTextSplitter` - Alternative sentence splitter using spaCy
- `langchain_text_splitters.KonlpyTextSplitter` - Korean-specific sentence splitter
- `langchain_text_splitters.CharacterTextSplitter` - Simple character-based splitting

### External Dependencies
- **NLTK**: Natural Language Toolkit (https://www.nltk.org/)
- **Punkt**: Unsupervised sentence boundary detection algorithm

### Use Cases
- Multi-language document processing
- RAG systems with accurate sentence boundaries
- Text summarization preprocessing
- Question-answering systems
- Chatbot context management

## Testing Considerations

### Unit Tests

```python
def test_english_sentence_splitting():
    text = "First sentence. Second sentence."
    splitter = NLTKTextSplitter(chunk_size=100)
    chunks = splitter.split_text(text)
    assert len(chunks) >= 1

def test_language_support():
    spanish_text = "Primera oración. Segunda oración."
    splitter = NLTKTextSplitter(language="spanish")
    chunks = splitter.split_text(spanish_text)
    assert len(chunks) >= 1

def test_span_tokenize_validation():
    with pytest.raises(ValueError):
        NLTKTextSplitter(use_span_tokenize=True, separator=" ")

def test_whitespace_preservation():
    text = "First.    Second."
    splitter = NLTKTextSplitter(use_span_tokenize=True, separator="")
    result = splitter.split_text(text)
    # Verify whitespace preserved
```

### Integration Tests
- Test with various languages
- Verify Punkt tokenizer is downloaded
- Test with edge cases (abbreviations, decimals, etc.)

### Edge Cases
- Empty strings
- Single sentence
- Text with no sentence-ending punctuation
- Abbreviations (Dr., Mr., etc.)
- Decimal numbers (3.14)
- Very long sentences exceeding chunk_size

## Performance Considerations

### Time Complexity
- O(n) for sentence tokenization where n is text length
- Punkt algorithm is efficient but not trivial
- O(m) for merging splits where m is number of sentences

### Space Complexity
- O(m) where m is number of sentences
- Punkt tokenizer loads language model into memory (small)

### Optimization Tips
1. **Reuse splitter instances** - Avoid reloading Punkt models
2. **Download tokenizers once** - Cache NLTK data
3. **Choose appropriate language** - Don't use English tokenizer for other languages
4. **Batch processing** - Process multiple texts with same splitter

## Best Practices

1. **Download Punkt data**: Run `nltk.download('punkt')` before first use
2. **Select correct language**: Use the appropriate language for your text
3. **Use span_tokenize for formatting**: When preserving original whitespace matters
4. **Set appropriate chunk_size**: Consider sentence length in target language
5. **Handle missing dependencies**: Provide fallback splitters

## Common Pitfalls

1. **Missing Punkt data**: NLTK requires downloading tokenizer models
2. **Wrong language**: Using English tokenizer for non-English text
3. **Separator with span_tokenize**: These options are mutually exclusive
4. **Abbreviation handling**: NLTK may struggle with uncommon abbreviations
5. **Performance on huge texts**: Punkt is slower than simple regex splitting

## Installation Guide

### Step 1: Install NLTK
```bash
pip install nltk
```

### Step 2: Download Punkt Tokenizer
```python
import nltk
nltk.download('punkt')
```

Or download all NLTK data:
```python
nltk.download('all')  # Warning: Large download
```

### Step 3: Verify Installation
```python
try:
    from langchain_text_splitters import NLTKTextSplitter
    import nltk
    nltk.data.find('tokenizers/punkt')
    splitter = NLTKTextSplitter()
    print("NLTK splitter ready!")
except Exception as e:
    print(f"Issue: {e}")
```

## Version History

- Part of `langchain-text-splitters` package
- Requires NLTK 3.0+
- Compatible with Python 3.8+
- Punkt tokenizer supports 17+ languages
