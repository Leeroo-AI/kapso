# HTMLHeaderTextSplitter

## Metadata
- **Source:** `/tmp/praxium_repo_wjjl6pl8/libs/text-splitters/langchain_text_splitters/html.py`
- **Domains:** text-splitters, html, document-processing
- **Last Updated:** 2025-12-17

## Overview

`HTMLHeaderTextSplitter` is a document splitter that processes HTML content by detecting specified header tags (h1-h6) and creating hierarchical `Document` objects that reflect the semantic structure of the original content. Each section's text is associated with metadata corresponding to the headers encountered in the document hierarchy.

### Description

The splitter performs a depth-first search traversal over the HTML DOM tree using BeautifulSoup. As it encounters header tags that match the configured splitting criteria, it:

1. Finalizes any accumulated content as a Document
2. Updates the active header hierarchy (removing headers at deeper or equal levels)
3. Yields the header text itself as a Document with appropriate metadata
4. Continues accumulating content under the new header context

The splitter maintains an "active headers" dictionary that tracks the current position in the document hierarchy. Each header is associated with its level (1-6), metadata name, and DOM depth. This allows proper handling of nested headers and ensures metadata accurately reflects the document structure.

Two modes are supported via the `return_each_element` parameter:
- **Aggregated mode** (default): Groups content under the same header hierarchy into single Documents
- **Element mode**: Returns every HTML element as a separate Document

### Usage

The splitter is particularly useful for processing documentation, articles, and structured web content where the header hierarchy provides meaningful semantic boundaries:

```python
from langchain_text_splitters.html import HTMLHeaderTextSplitter

# Define headers for splitting on h1 and h2 tags
headers_to_split_on = [
    ("h1", "Main Topic"),
    ("h2", "Sub Topic")
]

splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    return_each_element=False
)

html_content = """
<html>
    <body>
        <h1>Introduction</h1>
        <p>Welcome to the introduction section.</p>
        <h2>Background</h2>
        <p>Some background details here.</p>
        <h1>Conclusion</h1>
        <p>Final thoughts.</p>
    </body>
</html>
"""

documents = splitter.split_text(html_content)
```

## Code Reference

### Source Location
- **File:** `/tmp/praxium_repo_wjjl6pl8/libs/text-splitters/langchain_text_splitters/html.py`
- **Lines:** 83-336
- **Class:** `HTMLHeaderTextSplitter`

### Signature
```python
class HTMLHeaderTextSplitter:
    def __init__(
        self,
        headers_to_split_on: list[tuple[str, str]],
        return_each_element: bool = False,
    ) -> None:
        ...
```

### Import Statement
```python
from langchain_text_splitters.html import HTMLHeaderTextSplitter
```

## I/O Contract

### Initialization Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `headers_to_split_on` | `list[tuple[str, str]]` | Yes | - | List of (header_tag, header_name) pairs representing headers that define splitting boundaries. Example: `[("h1", "Header 1"), ("h2", "Header 2")]` |
| `return_each_element` | `bool` | No | `False` | If `True`, every HTML element (headers, paragraphs, etc.) is returned as a separate Document. If `False`, content under the same header hierarchy is aggregated. |

### Method: `split_text`

Splits HTML text into a list of Document objects.

| Aspect | Details |
|--------|---------|
| **Input** | `text: str` - The HTML text to split |
| **Output** | `list[Document]` - List of split Documents with page_content and metadata |
| **Side Effects** | Parses HTML using BeautifulSoup |

### Method: `split_text_from_url`

Fetches content from a URL and splits it.

| Aspect | Details |
|--------|---------|
| **Input** | `url: str` - URL to fetch<br>`timeout: int` - Request timeout (default: 10)<br>`**kwargs` - Additional request parameters |
| **Output** | `list[Document]` - List of split Documents |
| **Side Effects** | Makes HTTP request |
| **Exceptions** | `requests.RequestException` if HTTP request fails |

### Method: `split_text_from_file`

Splits HTML content from a file.

| Aspect | Details |
|--------|---------|
| **Input** | `file: str \| IO[str]` - File path or file-like object |
| **Output** | `list[Document]` - List of split Documents |
| **Side Effects** | Reads file if path is provided |

### Document Structure

Each returned Document contains:

| Field | Type | Description |
|-------|------|-------------|
| `page_content` | `str` | Extracted text content. Multiple text elements are joined with `"  \n"`. |
| `metadata` | `dict[str, str]` | Dictionary mapping header names to header text values. Contains all active headers at the point the content was extracted. |

### Processing Algorithm

1. **Parse HTML** using BeautifulSoup
2. **Initialize state**:
   - `active_headers`: Dictionary tracking current header hierarchy
   - `current_chunk`: List accumulating content under current headers
3. **DFS Traversal**:
   - For each node, extract direct text content (non-recursive)
   - **If header tag**:
     - Finalize current chunk (if aggregating)
     - Remove active headers at same or deeper level
     - Add new header to active headers
     - Yield header as Document
   - **If non-header tag**:
     - Remove headers out of DOM scope
     - Either yield element directly or accumulate in chunk
4. **Finalize** remaining chunk if aggregating

## Usage Examples

### Example 1: Basic Hierarchical Splitting

```python
from langchain_text_splitters.html import HTMLHeaderTextSplitter

headers_to_split_on = [
    ("h1", "Chapter"),
    ("h2", "Section"),
    ("h3", "Subsection")
]

splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

html = """
<h1>Chapter 1: Introduction</h1>
<p>This is the introduction.</p>
<h2>Section 1.1: Background</h2>
<p>Background information here.</p>
<h3>Subsection 1.1.1: History</h3>
<p>Historical context.</p>
"""

docs = splitter.split_text(html)

# docs[0]: "Chapter 1: Introduction" with metadata={"Chapter": "Chapter 1: Introduction"}
# docs[1]: "This is the introduction." with metadata={"Chapter": "Chapter 1: Introduction"}
# docs[2]: "Section 1.1: Background" with metadata={"Chapter": "Chapter 1: Introduction", "Section": "Section 1.1: Background"}
# docs[3]: "Background information here." with metadata={"Chapter": "Chapter 1: Introduction", "Section": "Section 1.1: Background"}
# ...and so on
```

### Example 2: Return Each Element

```python
splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[("h1", "Title")],
    return_each_element=True
)

html = """
<h1>My Title</h1>
<p>First paragraph.</p>
<p>Second paragraph.</p>
"""

docs = splitter.split_text(html)

# docs[0]: "My Title" with metadata={"Title": "My Title"}
# docs[1]: "First paragraph." with metadata={"Title": "My Title"}
# docs[2]: "Second paragraph." with metadata={"Title": "My Title"}
# Each element is a separate document
```

### Example 3: Split from URL

```python
splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[("h1", "Main"), ("h2", "Sub")]
)

# Fetch and split HTML from URL
docs = splitter.split_text_from_url(
    "https://example.com/article",
    timeout=15
)

for doc in docs:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("---")
```

### Example 4: Split from File

```python
splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[("h1", "Chapter"), ("h2", "Section")]
)

# From file path
docs = splitter.split_text_from_file("document.html")

# From file object
with open("document.html", "r") as f:
    docs = splitter.split_text_from_file(f)
```

### Example 5: Documentation Processing

```python
# Process technical documentation with multiple header levels
splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[
        ("h1", "API"),
        ("h2", "Class"),
        ("h3", "Method")
    ],
    return_each_element=False  # Aggregate content
)

docs = splitter.split_text(api_documentation_html)

# Filter to specific sections
method_docs = [
    doc for doc in docs
    if "Method" in doc.metadata and "authenticate" in doc.metadata["Method"]
]
```

## Related Pages

- `HTMLSectionSplitter` - Alternative HTML splitter using XSLT transformations
- `HTMLSemanticPreservingSplitter` - Advanced HTML splitter with semantic structure preservation
- `RecursiveCharacterTextSplitter` - Character-based text splitter for further chunking
- `Document` - Base document class from langchain-core
- `BeautifulSoup` - HTML parsing library (external dependency)
