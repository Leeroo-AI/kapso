---
title: "JSXTextSplitter (JSFrameworkTextSplitter)"
repository: "langchain-ai/langchain"
commit_hash: "1e3eaf4eac17"
file_path: "libs/text-splitters/langchain_text_splitters/jsx.py"
component_type: "Text Splitter"
component_name: "JSFrameworkTextSplitter"
layer: "Text Splitters"
---

# JSXTextSplitter (JSFrameworkTextSplitter)

## Overview

The `JSFrameworkTextSplitter` (also known as JSXTextSplitter) is a specialized text splitter designed for modern JavaScript framework code including React (JSX), Vue, and Svelte. It extends `RecursiveCharacterTextSplitter` to intelligently split component-based code by detecting custom component tags and combining them with JavaScript syntax separators.

**Key Features:**
- Automatically detects and extracts custom component tags from code (e.g., `<UserProfile>`, `<Button>`)
- Uses detected tags as natural splitting boundaries along with JavaScript syntax
- Supports React JSX, Vue, and Svelte component syntax
- Configurable chunk size and overlap
- Inherits all capabilities of RecursiveCharacterTextSplitter
- Preserves component structure and readability in chunks

**Location:** `/tmp/praxium_repo_wjjl6pl8/libs/text-splitters/langchain_text_splitters/jsx.py`

## Code Reference

### Class Definition

```python
class JSFrameworkTextSplitter(RecursiveCharacterTextSplitter):
    """Text splitter that handles React (JSX), Vue, and Svelte code.

    This splitter extends RecursiveCharacterTextSplitter to handle
    React (JSX), Vue, and Svelte code by:

    1. Detecting and extracting custom component tags from the text
    2. Using those tags as additional separators along with standard JS syntax

    The splitter combines:
    * Custom component tags as separators (e.g. <Component, <div)
    * JavaScript syntax elements (function, const, if, etc)
    * Standard text splitting on newlines
    """

    def __init__(
        self,
        separators: list[str] | None = None,
        chunk_size: int = 2000,
        chunk_overlap: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize the JS Framework text splitter.

        Args:
            separators: Optional list of custom separator strings to use
            chunk_size: Maximum size of chunks to return
            chunk_overlap: Overlap in characters between chunks
            **kwargs: Additional arguments to pass to parent class
        """
```

### Public Methods

```python
def split_text(self, text: str) -> list[str]:
    """Split text into chunks.

    This method splits the text into chunks by:
    * Extracting unique opening component tags using regex
    * Creating separators list with extracted tags and JS separators
    * Splitting the text using the separators by calling the parent class method

    Args:
        text: String containing code to split

    Returns:
        List of text chunks split on component and JS boundaries
    """
```

## I/O Contract

### Input

**Constructor Parameters:**
- `separators` (list[str] | None, default: None): Custom separators to prepend to detected separators
- `chunk_size` (int, default: 2000): Maximum size of chunks in characters
- `chunk_overlap` (int, default: 0): Number of characters to overlap between chunks
- `**kwargs` (Any): Additional parameters passed to RecursiveCharacterTextSplitter

**split_text() Parameters:**
- `text` (str): JavaScript framework code (JSX, Vue, Svelte) to split

### Output

**split_text() returns:** `list[str]`
- List of text chunks
- Each chunk respects component and syntax boundaries
- Chunks are approximately `chunk_size` characters with `chunk_overlap` between them

### Tag Detection

Uses regex pattern to extract opening tags:
```python
r"<\s*([a-zA-Z0-9]+)[^>]*>"
```

This matches:
- Standard HTML tags: `<div>`, `<span>`, `<button>`
- Custom components: `<UserProfile>`, `<MyComponent>`
- With attributes: `<Button className="primary">`
- With spacing: `< Button >`

Excludes:
- Self-closing tags: `<Component />`
- Closing tags: `</Component>`

## Usage Examples

### Basic React/JSX Splitting

```python
from langchain_text_splitters import JSFrameworkTextSplitter

jsx_code = """
import React from 'react';

function UserProfile({ name, age }) {
  return (
    <div className="profile">
      <Header title={name} />
      <div className="content">
        <Avatar src={`/images/${name}.jpg`} />
        <InfoSection>
          <p>Name: {name}</p>
          <p>Age: {age}</p>
        </InfoSection>
      </div>
      <Footer />
    </div>
  );
}

export default UserProfile;
"""

# Initialize splitter
splitter = JSFrameworkTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Split the code
chunks = splitter.split_text(jsx_code)

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i} ---")
    print(chunk)
    print(f"Length: {len(chunk)} chars\n")
```

### Vue Component Splitting

```python
vue_code = """
<template>
  <div class="user-profile">
    <HeaderComponent :title="userName" />
    <MainContent>
      <UserAvatar :src="avatarUrl" />
      <UserInfo :name="userName" :age="userAge" />
    </MainContent>
    <FooterComponent />
  </div>
</template>

<script>
export default {
  name: 'UserProfile',
  data() {
    return {
      userName: 'John Doe',
      userAge: 30,
      avatarUrl: '/images/john.jpg'
    }
  }
}
</script>
"""

splitter = JSFrameworkTextSplitter(chunk_size=400)
chunks = splitter.split_text(vue_code)
```

### Svelte Component Splitting

```python
svelte_code = """
<script>
  let name = 'John';
  let age = 30;
</script>

<div class="profile">
  <Header title={name} />
  <MainSection>
    <Avatar src="/images/{name}.jpg" />
    <InfoBox>
      <p>Name: {name}</p>
      <p>Age: {age}</p>
    </InfoBox>
  </MainSection>
  <Footer />
</div>
"""

splitter = JSFrameworkTextSplitter(chunk_size=300)
chunks = splitter.split_text(svelte_code)
```

### Custom Separators

```python
# Add custom separators in addition to auto-detected ones
splitter = JSFrameworkTextSplitter(
    separators=["// TODO:", "/* SECTION */"],
    chunk_size=1000,
    chunk_overlap=100
)

code_with_comments = """
// TODO: Refactor this component
function MyComponent() {
  // TODO: Add error handling
  return <div><Button>Click me</Button></div>
}
"""

chunks = splitter.split_text(code_with_comments)
```

### Processing Multiple Files

```python
import os
from pathlib import Path

def split_jsx_files(directory: str, chunk_size: int = 2000):
    splitter = JSFrameworkTextSplitter(chunk_size=chunk_size)
    results = {}

    for filepath in Path(directory).rglob("*.jsx"):
        with open(filepath, "r") as f:
            code = f.read()

        chunks = splitter.split_text(code)
        results[str(filepath)] = chunks

        print(f"{filepath}: {len(chunks)} chunks")

    return results

# Split all JSX files in a project
chunks_by_file = split_jsx_files("./src/components")
```

### Integration with Document Loaders

```python
from langchain_text_splitters import JSFrameworkTextSplitter
from langchain_core.documents import Document

# Load JSX files as documents
jsx_files = ["Component1.jsx", "Component2.jsx"]
documents = []

for filepath in jsx_files:
    with open(filepath, "r") as f:
        content = f.read()
    doc = Document(
        page_content=content,
        metadata={"source": filepath, "type": "jsx"}
    )
    documents.append(doc)

# Split documents
splitter = JSFrameworkTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = splitter.split_documents(documents)

print(f"Original: {len(documents)} docs")
print(f"After splitting: {len(split_docs)} chunks")
```

## Implementation Details

### Component Tag Detection

The `split_text()` method follows this algorithm:

1. **Extract Tags**: Use regex to find all opening tags
```python
opening_tags = re.findall(r"<\s*([a-zA-Z0-9]+)[^>]*>", text)
```

2. **Deduplicate**: Maintain order, remove duplicates
```python
component_tags = []
for tag in opening_tags:
    if tag not in component_tags:
        component_tags.append(tag)
```

3. **Create Separators**: Format tags as separators
```python
component_separators = [f"<{tag}" for tag in component_tags]
```

### JavaScript Separators

Pre-defined JS syntax separators:
```python
js_separators = [
    "\nexport ", " export ",
    "\nfunction ", "\nasync function ", " async function ",
    "\nconst ", "\nlet ", "\nvar ",
    "\nclass ", " class ",
    "\nif ", " if ",
    "\nfor ", " for ",
    "\nwhile ", " while ",
    "\nswitch ", " switch ",
    "\ncase ", " case ",
    "\ndefault ", " default ",
]
```

These capture:
- Module exports
- Function declarations (sync/async)
- Variable declarations
- Class definitions
- Control flow statements

### Separator Precedence

Final separator list prioritization:
```python
separators = (
    self._separators          # User-provided (highest priority)
    + js_separators           # JavaScript syntax
    + component_separators    # Detected component tags
    + ["<>", "\n\n", "&&\n", "||\n"]  # JSX fragments and logical operators
)
```

### Inheritance from RecursiveCharacterTextSplitter

After building the separator list, the parent class's `split_text()` method is called, which:
1. Attempts to split on each separator in order
2. Recursively splits chunks that are too large
3. Merges small chunks respecting chunk_size and overlap

## Design Patterns

### Template Method Pattern
Overrides `split_text()` to customize separator detection while reusing parent splitting logic.

### Strategy Pattern
Different separator strategies for different JS frameworks, unified through tag detection.

### Decorator Pattern
Enhances RecursiveCharacterTextSplitter with framework-aware capabilities.

## Related Components

### Parent Class
- `langchain_text_splitters.RecursiveCharacterTextSplitter` - Base recursive splitting implementation

### Related Splitters
- `langchain_text_splitters.CharacterTextSplitter` - Simple character-based splitting
- `langchain_text_splitters.TokenTextSplitter` - Token-based splitting
- `langchain_text_splitters.RecursiveJsonSplitter` - JSON-aware splitting

### Use Cases
- **Code Search**: Split large component files for semantic search
- **Documentation**: Generate documentation from component code
- **Code Analysis**: Analyze components independently
- **RAG Systems**: Build retrieval systems over codebases

## Testing Considerations

### Unit Tests

```python
def test_component_tag_detection():
    code = "<MyComponent>\n<AnotherComponent>"
    splitter = JSFrameworkTextSplitter(chunk_size=100)
    chunks = splitter.split_text(code)
    # Verify components split correctly

def test_javascript_syntax_splitting():
    code = "function a() {}\nfunction b() {}"
    splitter = JSFrameworkTextSplitter(chunk_size=50)
    chunks = splitter.split_text(code)
    # Verify function boundaries respected

def test_custom_separators():
    splitter = JSFrameworkTextSplitter(
        separators=["// SECTION"],
        chunk_size=100
    )
    code = "// SECTION A\ncode\n// SECTION B\nmore code"
    chunks = splitter.split_text(code)
    # Verify custom separators used
```

### Edge Cases
- Empty strings
- Code without components (plain JavaScript)
- Malformed JSX/HTML tags
- Very large single components
- Nested components with same names
- Self-closing tags (should be ignored)

## Performance Considerations

### Time Complexity
- O(n) for regex tag extraction
- O(n * m) where n is text length, m is number of separators (from parent class)
- Tag deduplication is O(k) where k is unique tags

### Space Complexity
- O(k) for storing unique component tags
- O(c) for output chunks where c is number of chunks
- Separator list grows with unique components found

### Optimization Tips
1. **Pre-split large files** before processing if very large
2. **Cache splitter instances** to reuse compiled regexes
3. **Adjust chunk_size** based on component complexity
4. **Use chunk_overlap** to maintain context across splits

## Best Practices

1. **Set appropriate chunk_size** based on embedding model limits (typically 512-8192 tokens)
2. **Use chunk_overlap** (10-20% of chunk_size) to maintain context
3. **Combine with metadata** to track file source and component names
4. **Test on representative code** from your codebase
5. **Handle imports separately** if building documentation systems

## Common Pitfalls

1. **Too small chunk_size**: May split components mid-declaration
2. **No overlap**: May lose context between chunks
3. **Ignoring metadata**: Difficult to trace chunks back to source files
4. **Mixed languages**: Assumes JavaScript-like syntax; may not work well with TypeScript decorators or other syntax extensions
5. **Comment handling**: Comments within components may affect splitting
6. **Minified code**: Not designed for minified/compressed code

## Version History

- Part of `langchain-text-splitters` package
- Supports modern JavaScript frameworks (React, Vue, Svelte)
- Extends RecursiveCharacterTextSplitter foundation
