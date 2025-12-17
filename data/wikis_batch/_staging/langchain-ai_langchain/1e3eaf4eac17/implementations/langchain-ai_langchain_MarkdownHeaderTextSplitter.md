# MarkdownHeaderTextSplitter

## Metadata
- **Source:** `/tmp/praxium_repo_wjjl6pl8/libs/text-splitters/langchain_text_splitters/markdown.py`
- **Domains:** text-splitters, markdown, document-processing
- **Last Updated:** 2025-12-17

## Overview

`MarkdownHeaderTextSplitter` is a document splitter that processes Markdown files by detecting headers (both standard `#` syntax and custom patterns) and creating `Document` objects with hierarchical metadata. The splitter maintains the header structure as metadata while extracting content, enabling semantic chunking of Markdown documents.

### Description

The splitter processes Markdown text line by line, tracking the header hierarchy through a stack-based approach. As it encounters headers:

1. Identifies headers using configured patterns (standard `#` or custom like `**Header**`)
2. Maintains a stack of active headers with their levels
3. Pops headers from the stack when encountering same or higher-level headers
4. Associates content with the current header context
5. Either returns each line individually or aggregates lines with common metadata

Key features:
- **Standard header support**: Recognizes Markdown headers `#` through `######`
- **Custom header patterns**: Supports non-standard header formats with configurable patterns
- **Code block awareness**: Preserves code blocks (fenced with ``` or ~~~) without treating internal content as headers
- **Metadata tracking**: Builds hierarchical metadata dictionary reflecting document structure
- **Flexible output**: Can return individual lines or aggregated chunks

The splitter is particularly useful for processing documentation, technical specifications, and structured Markdown content where header hierarchy conveys semantic meaning.

### Usage

```python
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter

# Define headers to track
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    return_each_line=False,  # Aggregate content
    strip_headers=True  # Remove header text from content
)

markdown_text = """
# Introduction
This is the introduction.

## Background
Some background information.

### Details
Detailed information here.
"""

docs = splitter.split_text(markdown_text)
```

## Code Reference

### Source Location
- **File:** `/tmp/praxium_repo_wjjl6pl8/libs/text-splitters/langchain_text_splitters/markdown.py`
- **Lines:** 23-266
- **Class:** `MarkdownHeaderTextSplitter`

### Signature
```python
class MarkdownHeaderTextSplitter:
    def __init__(
        self,
        headers_to_split_on: list[tuple[str, str]],
        return_each_line: bool = False,
        strip_headers: bool = True,
        custom_header_patterns: dict[str, int] | None = None,
    ) -> None:
        ...
```

### Import Statement
```python
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
```

## I/O Contract

### Initialization Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `headers_to_split_on` | `list[tuple[str, str]]` | Yes | - | List of (header_pattern, metadata_name) tuples defining which headers to track. Example: `[("#", "Header 1"), ("##", "Header 2")]` |
| `return_each_line` | `bool` | No | `False` | If `True`, returns each line as a separate Document. If `False`, aggregates lines with common metadata into chunks. |
| `strip_headers` | `bool` | No | `True` | If `True`, removes header text from the content. If `False`, includes header text in the Document content. |
| `custom_header_patterns` | `dict[str, int] \| None` | No | `None` | Optional dictionary mapping custom header patterns to their levels. Example: `{"**": 1, "***": 2}` treats `**Header**` as level 1. |

### Method: `split_text`

Splits Markdown text into Document objects.

| Aspect | Details |
|--------|---------|
| **Input** | `text: str` - The Markdown text to split |
| **Output** | `list[Document]` - List of Documents with page_content and hierarchical metadata |
| **Side Effects** | None (pure function) |

### Method: `aggregate_lines_to_chunks`

Combines lines with common metadata into larger chunks.

| Aspect | Details |
|--------|---------|
| **Input** | `lines: list[LineType]` - List of line dictionaries with content and metadata |
| **Output** | `list[Document]` - Aggregated Documents |
| **Side Effects** | None |

### Document Structure

Each returned Document contains:

| Field | Type | Description |
|-------|------|-------------|
| `page_content` | `str` | Extracted text content. Lines are joined with `"  \n"` (Markdown line break). |
| `metadata` | `dict[str, str]` | Dictionary mapping header metadata names to header text. Reflects the active header hierarchy at that point. |

### Processing Algorithm

1. **Split text** by newlines
2. **Initialize state**:
   - `lines_with_metadata`: Accumulated results
   - `current_content`: Lines under current headers
   - `current_metadata`: Active header metadata
   - `header_stack`: Stack tracking nested headers
   - `in_code_block`: Flag for code block detection
3. **Process each line**:
   - Skip header detection inside code blocks (``` or ~~~)
   - **If header line**:
     - Pop headers from stack at same or deeper level
     - Push new header onto stack
     - Update metadata with new header
     - Finalize current content (if any)
     - Optionally include header in content
   - **If content line**:
     - Accumulate in `current_content`
   - **If empty line** (and content exists):
     - Finalize current content chunk
4. **Aggregate or return** based on `return_each_line` setting

## Usage Examples

### Example 1: Basic Markdown Splitting

```python
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Title"),
    ("##", "Section"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

markdown = """
# My Document
This is the introduction.

## First Section
Content of first section.

## Second Section
Content of second section.
"""

docs = splitter.split_text(markdown)

# docs[0]: page_content="This is the introduction."
#          metadata={"Title": "My Document"}
# docs[1]: page_content="Content of first section."
#          metadata={"Title": "My Document", "Section": "First Section"}
# docs[2]: page_content="Content of second section."
#          metadata={"Title": "My Document", "Section": "Second Section"}
```

### Example 2: Keep Headers in Content

```python
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "H1"), ("##", "H2")],
    strip_headers=False  # Include headers in content
)

markdown = """
# Introduction
Welcome to the guide.

## Getting Started
Follow these steps.
"""

docs = splitter.split_text(markdown)

# docs[0]: page_content="# Introduction  \nWelcome to the guide."
#          metadata={"H1": "Introduction"}
# docs[1]: page_content="## Getting Started  \nFollow these steps."
#          metadata={"H1": "Introduction", "H2": "Getting Started"}
```

### Example 3: Return Each Line

```python
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "Header")],
    return_each_line=True
)

markdown = """
# My Header
First line.
Second line.
"""

docs = splitter.split_text(markdown)

# docs[0]: page_content="First line." metadata={"Header": "My Header"}
# docs[1]: page_content="Second line." metadata={"Header": "My Header"}
# Each non-empty line is a separate document
```

### Example 4: Custom Header Patterns

```python
# Support wiki-style headers like **Header** and ***Subheader***
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("**", "Main"), ("***", "Sub")],
    custom_header_patterns={"**": 1, "***": 2}
)

markdown = """
**Introduction**
This is the introduction.

***Background***
Background details.
"""

docs = splitter.split_text(markdown)

# docs[0]: page_content="This is the introduction."
#          metadata={"Main": "Introduction"}
# docs[1]: page_content="Background details."
#          metadata={"Main": "Introduction", "Sub": "Background"}
```

### Example 5: Multi-level Hierarchy

```python
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Chapter"),
        ("##", "Section"),
        ("###", "Subsection"),
    ]
)

markdown = """
# Chapter 1: Getting Started

## Installation
Instructions for installation.

### Prerequisites
You need Python 3.8+.

### Download
Download from the website.

## Configuration
Configure the application.

# Chapter 2: Advanced Topics
"""

docs = splitter.split_text(markdown)

# Each document has appropriate hierarchical metadata:
# - Installation content: {"Chapter": "Chapter 1: Getting Started", "Section": "Installation"}
# - Prerequisites: {"Chapter": "Chapter 1: Getting Started", "Section": "Installation", "Subsection": "Prerequisites"}
# - Download: {"Chapter": "Chapter 1: Getting Started", "Section": "Installation", "Subsection": "Download"}
# - Configuration: {"Chapter": "Chapter 1: Getting Started", "Section": "Configuration"}
```

### Example 6: Code Block Handling

```python
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "Section")]
)

markdown = """
# Code Example
Here's some code:

```python
# This is not a header
def hello():
    print("Hello")
```

Back to normal text.
"""

docs = splitter.split_text(markdown)

# The "# This is not a header" inside the code block
# is not treated as a header - it's preserved as code
```

## Related Pages

- `MarkdownTextSplitter` - Simpler recursive character splitter for Markdown
- `ExperimentalMarkdownSyntaxTextSplitter` - Advanced Markdown splitter with whitespace preservation
- `RecursiveCharacterTextSplitter` - Base class for recursive text splitting
- `HTMLHeaderTextSplitter` - Similar splitter for HTML documents
- `Document` - Base document class from langchain-core
