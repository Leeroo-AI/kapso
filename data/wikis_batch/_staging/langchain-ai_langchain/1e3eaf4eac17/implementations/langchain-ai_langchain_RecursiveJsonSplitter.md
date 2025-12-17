---
title: "RecursiveJsonSplitter"
repository: "langchain-ai/langchain"
commit_hash: "1e3eaf4eac17"
file_path: "libs/text-splitters/langchain_text_splitters/json.py"
component_type: "Text Splitter"
component_name: "RecursiveJsonSplitter"
layer: "Text Splitters"
---

# RecursiveJsonSplitter

## Overview

The `RecursiveJsonSplitter` is a specialized text splitter designed to partition JSON data into smaller, structured chunks while preserving the hierarchical structure of the original JSON. Unlike traditional text splitters that operate on flat strings, this splitter maintains the nested dictionary structure, ensuring each chunk is valid JSON with preserved key paths.

**Key Features:**
- Recursively splits nested JSON structures while maintaining hierarchy
- Configurable maximum and minimum chunk sizes
- Optional list-to-dictionary conversion for better chunking
- Preserves JSON validity in all output chunks
- Creates LangChain Document objects with metadata support
- Smart chunking that respects field boundaries

**Location:** `/tmp/praxium_repo_wjjl6pl8/libs/text-splitters/langchain_text_splitters/json.py`

## Code Reference

### Class Definition

```python
class RecursiveJsonSplitter:
    """Splits JSON data into smaller, structured chunks while preserving hierarchy.

    Attributes:
        max_chunk_size: The maximum size for each chunk (default: 2000)
        min_chunk_size: The minimum size for each chunk (default: max_chunk_size - 200)
    """

    def __init__(
        self,
        max_chunk_size: int = 2000,
        min_chunk_size: int | None = None
    ) -> None:
        """Initialize the chunk size configuration.

        Args:
            max_chunk_size: The maximum size for a chunk.
            min_chunk_size: The minimum size for a chunk. If None,
                defaults to the maximum chunk size minus 200, with a lower bound of 50.
        """
```

### Public Methods

```python
def split_json(
    self,
    json_data: dict[str, Any],
    convert_lists: bool = False,
) -> list[dict[str, Any]]:
    """Splits JSON into a list of JSON chunks.

    Args:
        json_data: Dictionary containing JSON data to split
        convert_lists: Whether to convert lists to dictionaries with index keys

    Returns:
        List of dictionary chunks, each preserving nested structure
    """

def split_text(
    self,
    json_data: dict[str, Any],
    convert_lists: bool = False,
    ensure_ascii: bool = True,
) -> list[str]:
    """Splits JSON into a list of JSON formatted strings.

    Args:
        json_data: Dictionary containing JSON data to split
        convert_lists: Whether to convert lists to dictionaries
        ensure_ascii: Whether to escape non-ASCII characters in output

    Returns:
        List of JSON-formatted strings
    """

def create_documents(
    self,
    texts: list[dict[str, Any]],
    convert_lists: bool = False,
    ensure_ascii: bool = True,
    metadatas: list[dict[Any, Any]] | None = None,
) -> list[Document]:
    """Create a list of Document objects from a list of json objects (dict).

    Args:
        texts: List of dictionaries to split and convert to documents
        convert_lists: Whether to convert lists to dictionaries
        ensure_ascii: Whether to escape non-ASCII characters
        metadatas: Optional list of metadata dicts, one per input text

    Returns:
        List of Document objects with JSON chunks as page_content
    """
```

### Internal Methods

```python
@staticmethod
def _json_size(data: dict[str, Any]) -> int:
    """Calculate the size of the serialized JSON object."""

@staticmethod
def _set_nested_dict(
    d: dict[str, Any],
    path: list[str],
    value: Any,
) -> None:
    """Set a value in a nested dictionary based on the given path."""

def _list_to_dict_preprocessing(self, data: Any) -> Any:
    """Recursively convert lists to dictionaries with string index keys."""

def _json_split(
    self,
    data: Any,
    current_path: list[str] | None = None,
    chunks: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Split json into maximum size dictionaries while preserving structure."""
```

## I/O Contract

### Input

**Constructor Parameters:**
- `max_chunk_size` (int, default: 2000): Maximum JSON string length per chunk
- `min_chunk_size` (int | None, default: max_chunk_size - 200): Minimum chunk size before creating new chunk

**Method Parameters:**
- `json_data` (dict[str, Any]): JSON data as Python dictionary
- `convert_lists` (bool, default: False): Convert arrays to objects with index keys
- `ensure_ascii` (bool, default: True): Escape non-ASCII characters in output
- `metadatas` (list[dict] | None): Metadata for each input document

### Output

**split_json() returns:** `list[dict[str, Any]]`
- List of dictionary chunks
- Each chunk is valid JSON-serializable dictionary
- Preserves nested structure with full key paths

**split_text() returns:** `list[str]`
- List of JSON-formatted strings
- Each string is valid JSON
- Suitable for storage or transmission

**create_documents() returns:** `list[Document]`
- List of LangChain Document objects
- `page_content` contains JSON string chunk
- `metadata` preserved from input with deep copy

### Size Calculation

Chunk size is measured as the length of the JSON string representation:
```python
len(json.dumps(data))
```

## Usage Examples

### Basic JSON Splitting

```python
from langchain_text_splitters import RecursiveJsonSplitter

# Sample nested JSON data
data = {
    "users": {
        "alice": {"age": 30, "city": "NYC", "hobbies": ["reading", "cycling"]},
        "bob": {"age": 25, "city": "SF", "hobbies": ["gaming", "cooking"]},
    },
    "metadata": {"version": "1.0", "timestamp": "2024-01-01"}
}

# Initialize splitter
splitter = RecursiveJsonSplitter(max_chunk_size=200, min_chunk_size=150)

# Split into dictionary chunks
chunks = splitter.split_json(data)
print(f"Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk}")
```

### Converting to JSON Strings

```python
# Split into JSON-formatted strings
json_strings = splitter.split_text(data)

for i, json_str in enumerate(json_strings):
    print(f"Chunk {i} size: {len(json_str)} chars")
    print(json_str)
    print("---")
```

### Handling Lists with convert_lists

```python
# Data with arrays
data_with_lists = {
    "items": [
        {"id": 1, "name": "Item A"},
        {"id": 2, "name": "Item B"},
        {"id": 3, "name": "Item C"}
    ]
}

# Convert lists to dictionaries for better chunking
chunks = splitter.split_json(data_with_lists, convert_lists=True)

# Result will have structure like:
# {"items": {"0": {"id": 1, "name": "Item A"}, ...}}
```

### Creating LangChain Documents

```python
from langchain_text_splitters import RecursiveJsonSplitter

# Multiple JSON documents
documents = [
    {"user": "alice", "data": {"score": 95, "status": "active"}},
    {"user": "bob", "data": {"score": 87, "status": "pending"}}
]

# Metadata for each document
metadatas = [
    {"source": "database", "collection": "users"},
    {"source": "database", "collection": "users"}
]

# Create Document objects
splitter = RecursiveJsonSplitter(max_chunk_size=500)
docs = splitter.create_documents(
    texts=documents,
    metadatas=metadatas,
    ensure_ascii=False
)

for doc in docs:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("---")
```

### Processing Large Nested Structures

```python
import json

# Load large JSON file
with open("large_data.json", "r") as f:
    large_json = json.load(f)

# Configure splitter for larger chunks
splitter = RecursiveJsonSplitter(
    max_chunk_size=5000,
    min_chunk_size=4500
)

# Split and save chunks
chunks = splitter.split_text(large_json, ensure_ascii=False)

for i, chunk in enumerate(chunks):
    with open(f"chunk_{i}.json", "w") as f:
        f.write(chunk)
```

## Implementation Details

### Recursive Splitting Algorithm

The core algorithm in `_json_split()`:

1. **Initialize**: Start with empty chunk list, current path tracker
2. **Iterate**: For each key-value pair in dictionary:
   - Calculate size of adding this field
   - Check if it fits in current chunk
   - If yes: Add to current chunk using path
   - If no and current chunk is big enough: Start new chunk
   - Recursively process nested structures
3. **Preserve paths**: Use `_set_nested_dict()` to maintain hierarchy
4. **Handle non-dicts**: Leaf values added directly to current path

### List to Dictionary Conversion

When `convert_lists=True`, lists are converted to dictionaries:

```python
# Input: ["a", "b", "c"]
# Output: {"0": "a", "1": "b", "2": "c"}

def _list_to_dict_preprocessing(self, data: Any) -> Any:
    if isinstance(data, list):
        return {
            str(i): self._list_to_dict_preprocessing(item)
            for i, item in enumerate(data)
        }
    # ... recursively process nested structures
```

This allows array elements to be split independently across chunks.

### Path Management

Nested paths are maintained as lists of keys:
```python
# For data structure: {"users": {"alice": {"age": 30}}}
# Path to age: ["users", "alice", "age"]

def _set_nested_dict(d, path, value):
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value
```

## Design Patterns

### Recursive Decomposition
The splitter recursively breaks down nested structures, maintaining context through path tracking.

### Builder Pattern
`create_documents()` builds complex Document objects from raw JSON chunks.

### Strategy Pattern
`convert_lists` parameter enables different strategies for handling arrays.

## Related Components

### Base Classes
- `langchain_core.documents.Document` - Document class used in `create_documents()`

### Related Splitters
- `langchain_text_splitters.CharacterTextSplitter` - Simple character-based splitting
- `langchain_text_splitters.RecursiveCharacterTextSplitter` - Recursive text splitting
- `langchain_text_splitters.TokenTextSplitter` - Token-based splitting

### Integration Points
- **Vector Stores**: Documents can be embedded and stored
- **Retrievers**: Chunks can be retrieved based on queries
- **Document Loaders**: Can process output from JSON loaders

## Testing Considerations

### Unit Tests

```python
def test_split_preserves_structure():
    data = {"a": {"b": {"c": "value"}}}
    splitter = RecursiveJsonSplitter(max_chunk_size=100)
    chunks = splitter.split_json(data)
    # Verify each chunk maintains nested structure

def test_chunk_size_limits():
    large_field = {"key": "x" * 1000}
    splitter = RecursiveJsonSplitter(max_chunk_size=500)
    chunks = splitter.split_json(large_field)
    # Verify no chunk exceeds max_chunk_size

def test_list_conversion():
    data = {"items": [1, 2, 3]}
    splitter = RecursiveJsonSplitter()
    chunks = splitter.split_json(data, convert_lists=True)
    # Verify lists converted to dicts with string indices
```

### Edge Cases
- Empty JSON objects
- Deeply nested structures
- Large single values exceeding max_chunk_size
- Arrays with mixed types
- Unicode characters with ensure_ascii

## Performance Considerations

### Time Complexity
- O(n) where n is the number of fields in the JSON
- Recursive processing adds overhead for deep nesting
- JSON serialization called frequently for size calculation

### Space Complexity
- O(k * m) where k is number of chunks, m is average chunk size
- Deep nesting increases path tracking overhead
- `convert_lists` increases memory for index keys

### Optimization Tips
1. **Adjust chunk sizes** based on use case (larger chunks = fewer splits)
2. **Avoid convert_lists** if arrays don't need independent splitting
3. **Cache large JSON objects** rather than re-splitting
4. **Consider streaming** for very large JSON files

## Best Practices

1. **Set appropriate chunk sizes** based on embedding model limits
2. **Use convert_lists** for JSON with large arrays
3. **Preserve metadata** when creating documents for traceability
4. **Test chunk validity** by parsing each output JSON string
5. **Handle Unicode** appropriately with ensure_ascii parameter

## Common Pitfalls

1. **Chunk size miscalculation**: Remember size is calculated on serialized JSON, not Python objects
2. **Lost list context**: Without convert_lists, arrays may not split well
3. **Metadata copying**: Forgetting that metadata is deep copied per chunk
4. **Empty last chunk**: Algorithm may create empty final chunk (automatically removed)
5. **Large atomic values**: Single values larger than max_chunk_size cannot be split further
