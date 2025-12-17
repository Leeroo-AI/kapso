---
title: "SentenceTransformersTokenTextSplitter"
repository: "langchain-ai/langchain"
commit_hash: "1e3eaf4eac17"
file_path: "libs/text-splitters/langchain_text_splitters/sentence_transformers.py"
component_type: "Text Splitter"
component_name: "SentenceTransformersTokenTextSplitter"
layer: "Text Splitters"
---

# SentenceTransformersTokenTextSplitter

## Overview

The `SentenceTransformersTokenTextSplitter` is a specialized text splitter that splits text based on token counts using Sentence Transformer model tokenizers. Unlike character-based splitters, this splitter ensures chunks respect the token limits of specific embedding models, making it ideal for preparing text for semantic embedding and retrieval tasks.

**Key Features:**
- Token-based splitting using Sentence Transformer model tokenizers
- Automatic token limit detection from model configuration
- Respects maximum sequence length of specific embedding models
- Strips start and stop tokens for accurate token counting
- Configurable tokens per chunk with overlap support
- Works with any Sentence Transformer model from HuggingFace

**Location:** `/tmp/praxium_repo_wjjl6pl8/libs/text-splitters/langchain_text_splitters/sentence_transformers.py`

## Code Reference

### Class Definition

```python
class SentenceTransformersTokenTextSplitter(TextSplitter):
    """Splitting text to tokens using sentence model tokenizer."""

    def __init__(
        self,
        chunk_overlap: int = 50,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        tokens_per_chunk: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter.

        Args:
            chunk_overlap: Number of tokens to overlap between chunks
            model_name: Name of the Sentence Transformer model to use
            tokens_per_chunk: Maximum tokens per chunk. If None, uses model's max_seq_length
            **kwargs: Additional arguments passed to TextSplitter base class

        Raises:
            ImportError: If sentence-transformers package is not installed
            ValueError: If tokens_per_chunk exceeds model's maximum token limit
        """
```

### Public Methods

```python
def split_text(self, text: str) -> list[str]:
    """Splits the input text into smaller components by splitting text on tokens.

    This method encodes the input text using a private _encode method, then
    strips the start and stop token IDs from the encoded result.

    Args:
        text: The input text to be split.

    Returns:
        A list of string components derived from the input text after encoding and
        processing.
    """

def count_tokens(self, *, text: str) -> int:
    """Counts the number of tokens in the given text.

    This method encodes the input text using a private _encode method and
    calculates the total number of tokens in the encoded result.

    Args:
        text: The input text for which the token count is calculated.

    Returns:
        The number of tokens in the encoded text.
    """
```

### Internal Methods

```python
def _initialize_chunk_configuration(self, *, tokens_per_chunk: int | None) -> None:
    """Initialize chunk configuration based on model and user settings."""

def _encode(self, text: str) -> list[int]:
    """Encode text to token IDs including start/end tokens."""
```

### Internal Attributes

```python
self._model: SentenceTransformer  # The loaded Sentence Transformer model
self.tokenizer  # The model's tokenizer
self.maximum_tokens_per_chunk: int  # Model's max sequence length
self.tokens_per_chunk: int  # Actual tokens per chunk to use
```

## I/O Contract

### Input

**Constructor Parameters:**
- `chunk_overlap` (int, default: 50): Number of tokens to overlap between chunks
- `model_name` (str, default: `"sentence-transformers/all-mpnet-base-v2"`): HuggingFace model name
- `tokens_per_chunk` (int | None, default: None): Max tokens per chunk (uses model max if None)
- `**kwargs` (Any): Additional parameters for TextSplitter base class

**split_text() Parameters:**
- `text` (str): Text to split into token-based chunks

**count_tokens() Parameters:**
- `text` (str): Text to count tokens for

### Output

**split_text() returns:** `list[str]`
- List of text chunks
- Each chunk contains at most `tokens_per_chunk` tokens
- Chunks overlap by `chunk_overlap` tokens
- Start and stop tokens are excluded from chunk content

**count_tokens() returns:** `int`
- Total number of tokens in the text (including start/stop tokens)

### Token Handling

The splitter handles special tokens specially:
- **Encoding**: Includes start and stop tokens
- **Chunk splitting**: Strips start/stop tokens before splitting
- **Counting**: Includes start/stop tokens in count

## Usage Examples

### Basic Token-Based Splitting

```python
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

text = """
Natural language processing enables computers to understand human language.
Machine learning models can be trained on large text datasets.
Embeddings represent text as dense vectors in semantic space.
This allows for similarity comparisons and retrieval tasks.
"""

# Initialize with default model
splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=10,
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Split text
chunks = splitter.split_text(text)

print(f"Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    token_count = splitter.count_tokens(text=chunk)
    print(f"Chunk {i+1} ({token_count} tokens):")
    print(chunk)
    print("---")
```

### Using Different Models

```python
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

# Small model (384 dimensions, 512 max tokens)
small_splitter = SentenceTransformersTokenTextSplitter(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    tokens_per_chunk=256
)

# Large model (768 dimensions, 512 max tokens)
large_splitter = SentenceTransformersTokenTextSplitter(
    model_name="sentence-transformers/all-mpnet-base-v2",
    tokens_per_chunk=384
)

# Multilingual model
multilingual_splitter = SentenceTransformersTokenTextSplitter(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    tokens_per_chunk=200
)

text = "Your text here..."
chunks = multilingual_splitter.split_text(text)
```

### Custom Token Limits

```python
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

# Default: uses model's max_seq_length
default_splitter = SentenceTransformersTokenTextSplitter()
print(f"Max tokens: {default_splitter.maximum_tokens_per_chunk}")
print(f"Using: {default_splitter.tokens_per_chunk}")

# Custom limit (must be <= model's max)
custom_splitter = SentenceTransformersTokenTextSplitter(
    tokens_per_chunk=200,
    chunk_overlap=20
)

text = "Long text content..."
chunks = custom_splitter.split_text(text)
```

### Token Counting

```python
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

splitter = SentenceTransformersTokenTextSplitter()

text = "This is a sample sentence for token counting."
token_count = splitter.count_tokens(text=text)
print(f"Text has {token_count} tokens")

# Verify chunks respect token limits
chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    count = splitter.count_tokens(text=chunk)
    print(f"Chunk {i}: {count} tokens")
    assert count <= splitter.tokens_per_chunk
```

### Integration with Embeddings

```python
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Use the same model for splitting and embedding
model_name = "sentence-transformers/all-mpnet-base-v2"

# Create splitter
splitter = SentenceTransformersTokenTextSplitter(
    model_name=model_name,
    chunk_overlap=50
)

# Create embeddings with same model
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Process document
doc = Document(
    page_content="""
    Long document content here that needs to be split into chunks
    that respect the embedding model's token limits...
    """,
    metadata={"source": "document.txt"}
)

# Split document
split_docs = splitter.split_documents([doc])

# Generate embeddings (guaranteed to fit model limits)
vectors = embeddings.embed_documents([d.page_content for d in split_docs])
print(f"Created {len(vectors)} embeddings for {len(split_docs)} chunks")
```

### Batch Processing with Token Tracking

```python
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from pathlib import Path

def process_documents_with_token_tracking(directory: str):
    splitter = SentenceTransformersTokenTextSplitter(
        model_name="sentence-transformers/all-mpnet-base-v2",
        tokens_per_chunk=384,
        chunk_overlap=50
    )

    results = {}

    for filepath in Path(directory).rglob("*.txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # Get token count before splitting
        total_tokens = splitter.count_tokens(text=text)

        # Split text
        chunks = splitter.split_text(text)

        # Verify token counts
        chunk_tokens = [splitter.count_tokens(text=c) for c in chunks]

        results[str(filepath)] = {
            "total_tokens": total_tokens,
            "num_chunks": len(chunks),
            "avg_tokens_per_chunk": sum(chunk_tokens) / len(chunk_tokens),
            "max_tokens_in_chunk": max(chunk_tokens),
            "chunks": chunks
        }

        print(f"{filepath.name}:")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Chunks: {len(chunks)}")
        print(f"  Avg tokens/chunk: {sum(chunk_tokens) / len(chunk_tokens):.1f}")

    return results

results = process_documents_with_token_tracking("./documents")
```

### Handling Long Documents

```python
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

# Load a large document
with open("large_document.txt", "r") as f:
    large_text = f.read()

# Create splitter with smaller chunks and overlap
splitter = SentenceTransformersTokenTextSplitter(
    model_name="sentence-transformers/all-mpnet-base-v2",
    tokens_per_chunk=256,  # Smaller chunks
    chunk_overlap=25       # 10% overlap
)

# Split document
chunks = splitter.split_text(large_text)

print(f"Document split into {len(chunks)} chunks")

# Verify no chunk exceeds token limit
for i, chunk in enumerate(chunks):
    token_count = splitter.count_tokens(text=chunk)
    if token_count > splitter.tokens_per_chunk:
        print(f"WARNING: Chunk {i} has {token_count} tokens (exceeds limit)")
```

### Error Handling

```python
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

# Handle missing dependency
try:
    splitter = SentenceTransformersTokenTextSplitter()
except ImportError as e:
    print("sentence-transformers not installed")
    print("Install with: pip install sentence-transformers")
    # Use fallback splitter
    from langchain_text_splitters import CharacterTextSplitter
    splitter = CharacterTextSplitter(chunk_size=1000)

# Handle invalid token limit
try:
    splitter = SentenceTransformersTokenTextSplitter(
        model_name="sentence-transformers/all-mpnet-base-v2",
        tokens_per_chunk=10000  # Exceeds model max of 512
    )
except ValueError as e:
    print(f"Invalid configuration: {e}")
```

## Implementation Details

### Model Initialization

The splitter loads the Sentence Transformer model at initialization:

```python
from sentence_transformers import SentenceTransformer

self._model = SentenceTransformer(self.model_name)
self.tokenizer = self._model.tokenizer
```

This gives access to:
- `self._model.max_seq_length`: Model's maximum sequence length
- `self.tokenizer`: HuggingFace tokenizer for encoding/decoding

### Chunk Configuration

The `_initialize_chunk_configuration()` method:

```python
def _initialize_chunk_configuration(self, *, tokens_per_chunk: int | None) -> None:
    self.maximum_tokens_per_chunk = self._model.max_seq_length

    if tokens_per_chunk is None:
        self.tokens_per_chunk = self.maximum_tokens_per_chunk
    else:
        self.tokens_per_chunk = tokens_per_chunk

    if self.tokens_per_chunk > self.maximum_tokens_per_chunk:
        msg = (
            f"The token limit of the models '{self.model_name}'"
            f" is: {self.maximum_tokens_per_chunk}."
            f" Argument tokens_per_chunk={self.tokens_per_chunk}"
            f" > maximum token limit."
        )
        raise ValueError(msg)
```

### Token Encoding

The `_encode()` method uses HuggingFace tokenizer:

```python
def _encode(self, text: str) -> list[int]:
    token_ids_with_start_and_end_token_ids = self.tokenizer.encode(
        text,
        max_length=self._max_length_equal_32_bit_integer,  # 2^32
        truncation="do_not_truncate",
    )
    return cast("list[int]", token_ids_with_start_and_end_token_ids)
```

Key points:
- `max_length` set to 2^32 to avoid truncation during encoding
- `truncation="do_not_truncate"` ensures full text is encoded
- Returns token IDs including special tokens

### Start/Stop Token Stripping

The `split_text()` method strips special tokens:

```python
def encode_strip_start_and_stop_token_ids(text: str) -> list[int]:
    return self._encode(text)[1:-1]  # Strip first and last tokens
```

This ensures:
- Chunk content doesn't include special tokens
- Token counts reflect actual content tokens
- Chunks can be concatenated without extra special tokens

### Tokenizer Object

The splitter creates a `Tokenizer` object for splitting:

```python
from langchain_text_splitters.base import Tokenizer, split_text_on_tokens

tokenizer = Tokenizer(
    chunk_overlap=self._chunk_overlap,
    tokens_per_chunk=self.tokens_per_chunk,
    decode=self.tokenizer.decode,
    encode=encode_strip_start_and_stop_token_ids,
)

return split_text_on_tokens(text=text, tokenizer=tokenizer)
```

This `Tokenizer` object:
- Encodes text to tokens
- Splits tokens into chunks respecting limits
- Decodes tokens back to strings
- Handles overlap between chunks

## Design Patterns

### Adapter Pattern
Adapts HuggingFace tokenizers to LangChain's TextSplitter interface.

### Strategy Pattern
Provides token-based splitting strategy within TextSplitter framework.

### Dependency Injection
Tokenizer encode/decode functions injected into Tokenizer object.

## Related Components

### Parent Class
- `langchain_text_splitters.base.TextSplitter` - Base class for all text splitters
- `langchain_text_splitters.base.split_text_on_tokens` - Token-based splitting function
- `langchain_text_splitters.base.Tokenizer` - Tokenizer configuration object

### Related Splitters
- `langchain_text_splitters.TokenTextSplitter` - Generic token-based splitter
- `langchain_text_splitters.CharacterTextSplitter` - Character-based splitter
- `langchain_text_splitters.RecursiveCharacterTextSplitter` - Recursive splitting

### External Dependencies
- **sentence-transformers**: Sentence embedding models
- **transformers**: HuggingFace transformers library (dependency of sentence-transformers)
- **torch**: PyTorch (dependency of sentence-transformers)

### Use Cases
- **Embedding preparation**: Split text to fit embedding model token limits
- **Semantic search**: Prepare documents for vector stores
- **RAG systems**: Ensure chunks fit in context windows
- **Text similarity**: Process documents for similarity comparison

## Testing Considerations

### Unit Tests

```python
def test_token_splitting():
    splitter = SentenceTransformersTokenTextSplitter(tokens_per_chunk=50)
    text = "Short text. " * 100  # Long repeated text
    chunks = splitter.split_text(text)
    for chunk in chunks:
        assert splitter.count_tokens(text=chunk) <= 50

def test_token_counting():
    splitter = SentenceTransformersTokenTextSplitter()
    text = "This is a test sentence."
    count = splitter.count_tokens(text=text)
    assert count > 0

def test_chunk_overlap():
    splitter = SentenceTransformersTokenTextSplitter(
        tokens_per_chunk=100,
        chunk_overlap=10
    )
    text = "Word " * 200
    chunks = splitter.split_text(text)
    # Verify overlap exists between consecutive chunks

def test_invalid_tokens_per_chunk():
    with pytest.raises(ValueError):
        SentenceTransformersTokenTextSplitter(tokens_per_chunk=10000)
```

### Integration Tests
- Test with various Sentence Transformer models
- Verify embeddings work with split chunks
- Test with different languages (multilingual models)

### Edge Cases
- Empty strings
- Single word
- Text shorter than tokens_per_chunk
- Text with special characters
- Very long documents
- Unicode text

## Performance Considerations

### Time Complexity
- O(n) for tokenization where n is text length
- Model loading is expensive (one-time cost)
- Encoding/decoding is fast with HuggingFace tokenizers

### Space Complexity
- O(m) where m is model size (hundreds of MB)
- O(k) for token IDs where k is number of tokens
- O(c) for output chunks

### Optimization Tips
1. **Reuse splitter instances** - Model loading is slow
2. **Batch processing** - Process multiple texts with same splitter
3. **Choose smaller models** - MiniLM models are faster than larger models
4. **Adjust tokens_per_chunk** - Larger chunks = fewer chunks = faster processing
5. **Cache models** - Use sentence-transformers model caching

### Model Download
First use downloads model from HuggingFace:
- Models cached in `~/.cache/torch/sentence_transformers/`
- Sizes range from ~80MB (MiniLM) to ~400MB (large models)
- Requires internet connection on first use

## Best Practices

1. **Match splitter and embeddings**: Use same model for splitting and embedding
2. **Set appropriate chunk size**: Leave headroom below max_seq_length
3. **Use chunk_overlap**: 10-20% overlap helps maintain context
4. **Cache splitter instances**: Model loading is expensive
5. **Monitor token counts**: Verify chunks stay within limits
6. **Choose appropriate model**: Balance between speed and quality

## Common Pitfalls

1. **Model mismatch**: Using different models for splitting and embedding
2. **Exceeding token limits**: Setting tokens_per_chunk too high
3. **No overlap**: Losing context between chunks
4. **First-time latency**: Model download on first use
5. **Memory usage**: Large models consume significant RAM
6. **GPU availability**: Models run on GPU if available, CPU otherwise

## Installation Guide

### Step 1: Install sentence-transformers
```bash
pip install sentence-transformers
```

This also installs:
- `transformers` (HuggingFace)
- `torch` (PyTorch)
- Other dependencies

### Step 2: Verify Installation
```python
try:
    from langchain_text_splitters import SentenceTransformersTokenTextSplitter
    splitter = SentenceTransformersTokenTextSplitter()
    print("Splitter initialized successfully!")
    print(f"Model: {splitter.model_name}")
    print(f"Max tokens: {splitter.maximum_tokens_per_chunk}")
except Exception as e:
    print(f"Installation issue: {e}")
```

### Optional: Pre-download Models
```python
from sentence_transformers import SentenceTransformer

# Download models in advance
models = [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
]

for model_name in models:
    print(f"Downloading {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"  Max tokens: {model.max_seq_length}")
```

## Version History

- Part of `langchain-text-splitters` package
- Requires `sentence-transformers` 2.0.0+
- Compatible with Python 3.8+
- Supports all HuggingFace Sentence Transformer models
