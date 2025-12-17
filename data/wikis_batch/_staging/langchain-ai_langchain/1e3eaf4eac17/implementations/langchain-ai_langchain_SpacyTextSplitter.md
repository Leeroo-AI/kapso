---
title: "SpacyTextSplitter"
repository: "langchain-ai/langchain"
commit_hash: "1e3eaf4eac17"
file_path: "libs/text-splitters/langchain_text_splitters/spacy.py"
component_type: "Text Splitter"
component_name: "SpacyTextSplitter"
layer: "Text Splitters"
language_support: "Multi-language"
---

# SpacyTextSplitter

## Overview

The `SpacyTextSplitter` is a text splitter that uses spaCy's industrial-strength natural language processing library for sentence boundary detection. It supports multiple languages and NLP pipelines, with options for faster sentencizer-based splitting or more accurate model-based splitting. The splitter also provides control over whitespace handling in the output.

**Key Features:**
- Multi-language sentence tokenization using spaCy
- Two modes: full NLP pipeline (`en_core_web_sm`, etc.) or fast sentencizer
- Configurable maximum text length for large documents
- Whitespace handling control (strip or preserve)
- Support for 20+ languages via spaCy models
- Automatic import checking with helpful error messages
- Inherits chunk management from base TextSplitter

**Location:** `/tmp/praxium_repo_wjjl6pl8/libs/text-splitters/langchain_text_splitters/spacy.py`

## Code Reference

### Class Definition

```python
class SpacyTextSplitter(TextSplitter):
    """Splitting text using Spacy package.

    Per default, Spacy's `en_core_web_sm` model is used and
    its default max_length is 1000000 (it is the length of maximum character
    this model takes which can be increased for large files). For a faster, but
    potentially less accurate splitting, you can use `pipeline='sentencizer'`.
    """

    def __init__(
        self,
        separator: str = "\n\n",
        pipeline: str = "en_core_web_sm",
        max_length: int = 1_000_000,
        *,
        strip_whitespace: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the spacy text splitter.

        Args:
            separator: The separator to use when merging splits
            pipeline: spaCy pipeline name ('en_core_web_sm', 'sentencizer', etc.)
            max_length: Maximum character length for spaCy to process
            strip_whitespace: If True, strip whitespace from sentences
            **kwargs: Additional arguments passed to TextSplitter base class

        Raises:
            ImportError: If spacy package is not installed
        """
```

### Public Methods

```python
def split_text(self, text: str) -> list[str]:
    """Split incoming text and return chunks.

    Uses spaCy's sentence segmentation, then merges sentences
    according to chunk_size and separator.

    Args:
        text: Text to split into sentence-based chunks

    Returns:
        List of text chunks split at sentence boundaries
    """
```

### Module-Level Functions

```python
def _make_spacy_pipeline_for_splitting(
    pipeline: str, *, max_length: int = 1_000_000
) -> Language:
    """Create and configure a spaCy pipeline for text splitting.

    Args:
        pipeline: Pipeline name ('sentencizer' for fast, or model name for accurate)
        max_length: Maximum character length for processing

    Returns:
        Configured spaCy Language object

    Raises:
        ImportError: If spacy is not installed
    """
```

### Internal Components

```python
# Package availability check
_HAS_SPACY = True  # Set to False if import fails

# spaCy pipeline (initialized in __init__)
self._tokenizer: Language  # spaCy Language object with sentence segmentation
```

## I/O Contract

### Input

**Constructor Parameters:**
- `separator` (str, default: `"\n\n"`): String to use when joining sentences into chunks
- `pipeline` (str, default: `"en_core_web_sm"`): spaCy pipeline or model name
- `max_length` (int, default: 1,000,000): Maximum characters spaCy can process
- `strip_whitespace` (bool, default: True): Whether to strip whitespace from sentences
- `**kwargs` (Any): Additional parameters for TextSplitter base class

**split_text() Parameters:**
- `text` (str): Text to split into chunks

### Output

**split_text() returns:** `list[str]`
- List of text chunks
- Each chunk respects sentence boundaries
- Whitespace handled according to `strip_whitespace` setting
- Chunks merged according to chunk_size and separator settings

### Pipeline Options

**Fast Mode:**
- `pipeline="sentencizer"`: Uses rule-based sentence boundary detection
- Faster, lower accuracy
- Good for simple, well-formed text

**Accurate Mode:**
- `pipeline="en_core_web_sm"`: Small English model
- `pipeline="en_core_web_md"`: Medium English model
- `pipeline="en_core_web_lg"`: Large English model
- Plus models for other languages (de, fr, es, etc.)

### Supported Languages

Via spaCy models:
- English: `en_core_web_sm/md/lg`
- German: `de_core_news_sm/md/lg`
- French: `fr_core_news_sm/md/lg`
- Spanish: `es_core_news_sm/md/lg`
- Portuguese: `pt_core_news_sm/md/lg`
- Italian: `it_core_news_sm/md/lg`
- Dutch: `nl_core_news_sm/md/lg`
- Greek: `el_core_news_sm/md/lg`
- Chinese: `zh_core_web_sm/md/lg`
- Japanese: `ja_core_news_sm/md/lg`
- And 10+ more languages

### Dependencies

**Required Package:**
- `spacy`: Industrial-strength NLP library
- Installation: `pip install spacy`
- Models: `python -m spacy download en_core_web_sm`

## Usage Examples

### Basic English Text Splitting

```python
from langchain_text_splitters import SpacyTextSplitter

text = """
Natural language processing is a fascinating field. It combines linguistics and computer science.
Machine learning models have revolutionized NLP. They can understand context and semantics.
Applications include translation, sentiment analysis, and question answering.
"""

# Initialize with default model
splitter = SpacyTextSplitter(
    separator=" ",
    chunk_size=150,
    chunk_overlap=20
)

# Split text
chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(chunk)
    print(f"Length: {len(chunk)} chars\n")
```

### Fast Sentencizer Mode

```python
from langchain_text_splitters import SpacyTextSplitter

# Use sentencizer for faster splitting
fast_splitter = SpacyTextSplitter(
    pipeline="sentencizer",  # Rule-based, no model required
    separator="\n",
    chunk_size=200
)

text = """
This is the first sentence. This is the second sentence.
This is the third sentence. And this is the fourth.
"""

chunks = fast_splitter.split_text(text)
print(f"Sentencizer created {len(chunks)} chunks")
```

### Multi-Language Support

```python
from langchain_text_splitters import SpacyTextSplitter

# German text
german_text = """
Künstliche Intelligenz ist faszinierend. Sie verändert unsere Welt.
Maschinelles Lernen ermöglicht neue Anwendungen. Die Zukunft ist spannend.
"""

german_splitter = SpacyTextSplitter(
    pipeline="de_core_news_sm",  # German model
    chunk_size=150
)
german_chunks = german_splitter.split_text(german_text)

# French text
french_text = """
L'intelligence artificielle est révolutionnaire. Elle transforme notre société.
Les modèles d'apprentissage sont puissants. L'avenir est prometteur.
"""

french_splitter = SpacyTextSplitter(
    pipeline="fr_core_news_sm",  # French model
    chunk_size=150
)
french_chunks = french_splitter.split_text(french_text)
```

### Whitespace Handling

```python
from langchain_text_splitters import SpacyTextSplitter

text = "First sentence.   Second sentence with spaces.   Third sentence."

# Strip whitespace (default)
strip_splitter = SpacyTextSplitter(
    separator=" ",
    strip_whitespace=True
)
stripped = strip_splitter.split_text(text)
print("Stripped:", stripped)
# Output: ['First sentence.', 'Second sentence with spaces.', ...]

# Preserve whitespace
preserve_splitter = SpacyTextSplitter(
    separator="",
    strip_whitespace=False
)
preserved = preserve_splitter.split_text(text)
print("Preserved:", preserved)
# Output: ['First sentence.   ', 'Second sentence with spaces.   ', ...]
```

### Processing Large Documents

```python
from langchain_text_splitters import SpacyTextSplitter

# Read large file
with open("large_document.txt", "r") as f:
    large_text = f.read()

# Configure for large documents
splitter = SpacyTextSplitter(
    pipeline="en_core_web_sm",
    max_length=2_000_000,  # Increase max_length for large files
    separator="\n\n",
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(large_text)
print(f"Split {len(large_text)} chars into {len(chunks)} chunks")
```

### Integration with Documents

```python
from langchain_text_splitters import SpacyTextSplitter
from langchain_core.documents import Document

# Create documents
documents = [
    Document(
        page_content="""
        Machine learning is a subset of AI. It enables computers to learn from data.
        Deep learning uses neural networks. These models can recognize patterns.
        """,
        metadata={"source": "ml_intro.txt", "topic": "ML"}
    ),
    Document(
        page_content="""
        Natural language processing analyzes text. It helps computers understand language.
        Applications include chatbots and translation. NLP is rapidly evolving.
        """,
        metadata={"source": "nlp_intro.txt", "topic": "NLP"}
    )
]

# Split documents
splitter = SpacyTextSplitter(chunk_size=200, chunk_overlap=20)
split_docs = splitter.split_documents(documents)

for i, doc in enumerate(split_docs):
    print(f"Document {i+1}:")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}\n")
```

### Comparing Pipeline Modes

```python
from langchain_text_splitters import SpacyTextSplitter
import time

text = "Sentence one. Sentence two! Sentence three? " * 100

# Sentencizer (fast)
start = time.time()
fast_splitter = SpacyTextSplitter(pipeline="sentencizer")
fast_chunks = fast_splitter.split_text(text)
fast_time = time.time() - start

# Full model (accurate)
start = time.time()
accurate_splitter = SpacyTextSplitter(pipeline="en_core_web_sm")
accurate_chunks = accurate_splitter.split_text(text)
accurate_time = time.time() - start

print(f"Sentencizer: {len(fast_chunks)} chunks in {fast_time:.3f}s")
print(f"Full model: {len(accurate_chunks)} chunks in {accurate_time:.3f}s")
```

### Error Handling

```python
from langchain_text_splitters import SpacyTextSplitter

# Handle missing spacy
try:
    splitter = SpacyTextSplitter()
except ImportError:
    print("spaCy not installed")
    print("Install with: pip install spacy")
    print("Download model: python -m spacy download en_core_web_sm")
    # Use fallback
    from langchain_text_splitters import CharacterTextSplitter
    splitter = CharacterTextSplitter(separator=". ")

# Handle missing model
try:
    splitter = SpacyTextSplitter(pipeline="en_core_web_lg")
except OSError as e:
    print(f"Model not found: {e}")
    print("Download with: python -m spacy download en_core_web_lg")
    # Fallback to smaller model
    splitter = SpacyTextSplitter(pipeline="en_core_web_sm")
```

### Batch Processing

```python
from pathlib import Path
from langchain_text_splitters import SpacyTextSplitter

def process_text_corpus(directory: str, language: str = "en"):
    # Map language codes to spaCy models
    models = {
        "en": "en_core_web_sm",
        "de": "de_core_news_sm",
        "fr": "fr_core_news_sm",
        "es": "es_core_news_sm",
    }

    splitter = SpacyTextSplitter(
        pipeline=models.get(language, "sentencizer"),
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
            "avg_length": sum(len(c) for c in chunks) / len(chunks),
            "chunks": chunks
        }

        print(f"{filepath.name}: {len(chunks)} chunks")

    return results

# Process English documents
results = process_text_corpus("./english_docs", language="en")
```

## Implementation Details

### Pipeline Creation

The `_make_spacy_pipeline_for_splitting()` function creates the pipeline:

```python
def _make_spacy_pipeline_for_splitting(
    pipeline: str, *, max_length: int = 1_000_000
) -> Language:
    if not _HAS_SPACY:
        msg = "Spacy is not installed, please install it with `pip install spacy`."
        raise ImportError(msg)

    if pipeline == "sentencizer":
        # Fast rule-based sentence boundary detection
        sentencizer: Language = English()
        sentencizer.add_pipe("sentencizer")
    else:
        # Load full NLP model, exclude NER and tagger for speed
        sentencizer = spacy.load(pipeline, exclude=["ner", "tagger"])
        sentencizer.max_length = max_length

    return sentencizer
```

Key optimizations:
- Excludes NER (named entity recognition) and tagger for speed
- Configurable max_length for large documents
- Sentencizer mode for maximum speed

### Sentence Extraction

The `split_text()` method extracts sentences:

```python
def split_text(self, text: str) -> list[str]:
    splits = (
        s.text if self._strip_whitespace else s.text_with_ws
        for s in self._tokenizer(text).sents
    )
    return self._merge_splits(splits, self._separator)
```

spaCy provides two sentence text options:
- `s.text`: Sentence without trailing whitespace (default)
- `s.text_with_ws`: Sentence with trailing whitespace

### Merge Strategy

After sentence extraction, `_merge_splits()` from base class:
1. Combines sentences into chunks respecting chunk_size
2. Adds chunk_overlap if configured
3. Joins sentences with the separator

## Design Patterns

### Template Method Pattern
Implements `split_text()` while delegating chunk merging to parent class.

### Factory Pattern
`_make_spacy_pipeline_for_splitting()` creates appropriate pipeline based on configuration.

### Strategy Pattern
Two strategies: sentencizer (fast) or full model (accurate).

## Related Components

### Parent Class
- `langchain_text_splitters.base.TextSplitter` - Base class for all text splitters

### Related Splitters
- `langchain_text_splitters.NLTKTextSplitter` - NLTK-based sentence splitting
- `langchain_text_splitters.KonlpyTextSplitter` - Korean-specific splitting
- `langchain_text_splitters.CharacterTextSplitter` - Simple character-based splitting

### External Dependencies
- **spaCy**: Industrial NLP library (https://spacy.io/)
- **Language models**: Pre-trained models for different languages
- Models must be downloaded separately: `python -m spacy download en_core_web_sm`

### Use Cases
- Multi-language document processing
- Accurate sentence boundary detection
- Named entity preserving chunks (when not excluded)
- Scientific text processing
- RAG systems requiring linguistic accuracy

## Testing Considerations

### Unit Tests

```python
def test_spacy_splitting():
    text = "First sentence. Second sentence."
    splitter = SpacyTextSplitter(chunk_size=50)
    chunks = splitter.split_text(text)
    assert len(chunks) > 0

def test_sentencizer_mode():
    splitter = SpacyTextSplitter(pipeline="sentencizer")
    text = "One. Two. Three."
    chunks = splitter.split_text(text)
    assert len(chunks) >= 3

def test_whitespace_handling():
    text = "First.  Second."
    strip_splitter = SpacyTextSplitter(strip_whitespace=True)
    preserve_splitter = SpacyTextSplitter(strip_whitespace=False)
    # Verify different whitespace handling

def test_max_length():
    long_text = "word " * 2_000_000
    splitter = SpacyTextSplitter(max_length=3_000_000)
    chunks = splitter.split_text(long_text)
    assert len(chunks) > 0
```

### Integration Tests
- Test with different language models
- Verify model download process
- Test with various text types (news, scientific, conversational)

### Edge Cases
- Empty strings
- Single sentence
- Text without punctuation
- Mixed languages in single text
- Very long sentences
- Abbreviations (Dr., Mr., etc.)
- Decimal numbers
- Text exceeding max_length

## Performance Considerations

### Time Complexity
- O(n) for sentence segmentation where n is text length
- Full models are slower than sentencizer
- Model loading is expensive (one-time per instance)

### Space Complexity
- O(m) where m is model size (MB to GB depending on model)
- Small models: ~10-20 MB
- Medium models: ~40-50 MB
- Large models: ~500-800 MB

### Optimization Tips
1. **Use sentencizer for speed**: 10-100x faster than full models
2. **Reuse splitter instances**: Model loading is slow
3. **Exclude unnecessary components**: NER and tagger excluded by default
4. **Choose appropriate model size**: Small models often sufficient
5. **Adjust max_length**: Increase for large documents
6. **Batch processing**: Process multiple texts with same splitter

## Best Practices

1. **Choose appropriate pipeline**: Sentencizer for speed, models for accuracy
2. **Download models in advance**: Avoid runtime downloads
3. **Set max_length appropriately**: Too low truncates, too high uses memory
4. **Consider whitespace needs**: Strip for cleaner output, preserve for formatting
5. **Use language-specific models**: Better accuracy than generic models
6. **Cache splitter instances**: Avoid repeated model loading

## Common Pitfalls

1. **Missing models**: Forgetting to download spaCy models
2. **Wrong model name**: Using non-existent model names
3. **max_length too small**: Text gets truncated silently
4. **Performance expectations**: Full models are slower than expected
5. **Memory usage**: Large models consume significant RAM
6. **Language detection**: spaCy doesn't auto-detect language; must specify correct model

## Installation Guide

### Step 1: Install spaCy
```bash
pip install spacy
```

### Step 2: Download Models
```bash
# English (small, fast)
python -m spacy download en_core_web_sm

# English (large, accurate)
python -m spacy download en_core_web_lg

# Other languages
python -m spacy download de_core_news_sm  # German
python -m spacy download fr_core_news_sm  # French
python -m spacy download es_core_news_sm  # Spanish
```

### Step 3: Verify Installation
```python
try:
    from langchain_text_splitters import SpacyTextSplitter
    import spacy

    # Check model availability
    try:
        nlp = spacy.load("en_core_web_sm")
        print("en_core_web_sm available")
    except OSError:
        print("en_core_web_sm not found")
        print("Download: python -m spacy download en_core_web_sm")

    # Test splitter
    splitter = SpacyTextSplitter()
    result = splitter.split_text("Test sentence. Another sentence.")
    print(f"Splitter working: {len(result)} chunks")

except Exception as e:
    print(f"Issue: {e}")
```

## Version History

- Part of `langchain-text-splitters` package
- Requires spaCy 3.0+
- Compatible with Python 3.8+
- Supports 20+ languages via spaCy models
- Type hints compatible with Python 3.14 with type: ignore directives
