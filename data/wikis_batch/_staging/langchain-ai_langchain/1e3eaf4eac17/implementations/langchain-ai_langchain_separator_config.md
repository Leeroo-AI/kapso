= separator_config =

'''Sources:''' /tmp/praxium_repo_wjjl6pl8/libs/text-splitters/langchain_text_splitters/character.py:L171-793

'''Domains:''' Text Splitting, RAG, Code Processing, Natural Language Processing

'''Last Updated:''' 2025-12-17

== Overview ==

The separator_config implementation provides language-specific separator patterns and customization options for text splitting. The get_separators_for_language() static method returns pre-configured separator lists for 25+ programming languages, while custom separators can be provided directly to splitters.

== Description ==

LangChain's RecursiveCharacterTextSplitter supports three ways to configure separators:

'''Default Configuration:''' Using built-in separators ["\n\n", "\n", " ", ""] for general text.

'''Language-specific Configuration:''' Using get_separators_for_language() to get separators optimized for specific programming languages. These separators respect syntactic boundaries like class definitions, functions, and control flow statements.

'''Custom Configuration:''' Providing a custom list of separators for specialized content or domain-specific formats.

Separators are tried in order, with the first matching separator used for splitting. If chunks are still too large, the splitter recursively applies remaining separators.

== Code Reference ==

The implementation is in libs/text-splitters/langchain_text_splitters/character.py:

<syntaxhighlight lang="python">
class RecursiveCharacterTextSplitter(TextSplitter):
    def __init__(
        self,
        separators: list[str] | None = None,
        keep_separator: bool | Literal["start", "end"] = True,
        is_separator_regex: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._is_separator_regex = is_separator_regex

    @staticmethod
    def get_separators_for_language(language: Language) -> list[str]:
        """Retrieve separators specific to the given language."""
        # Returns language-specific separator list
        # 25+ languages supported with optimized patterns
</syntaxhighlight>

== API ==

=== Default Separators ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Uses default separators: ["\n\n", "\n", " ", ""]
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
</syntaxhighlight>

=== Language-specific Separators ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Method 1: Use from_language() factory method
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=200
)

# Method 2: Get separators and pass manually
python_seps = RecursiveCharacterTextSplitter.get_separators_for_language(
    Language.PYTHON
)
splitter = RecursiveCharacterTextSplitter(
    separators=python_seps,
    is_separator_regex=True,  # Language separators use regex
    chunk_size=1000,
    chunk_overlap=200
)
</syntaxhighlight>

=== Custom Separators ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Custom separator list for specific format
custom_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n## ",      # H2 headers
        "\n### ",     # H3 headers
        "\n\n",       # Paragraphs
        "\n",         # Lines
        ". ",         # Sentences
        " ",          # Words
        ""            # Characters
    ],
    chunk_size=1000,
    chunk_overlap=200
)

# Using regex patterns
regex_splitter = RecursiveCharacterTextSplitter(
    separators=[
        r"\n#{1,6} ",   # Any markdown header
        r"\n\n+",       # Multiple blank lines
        r"\.\s+",       # Sentence boundaries
        " ",
        ""
    ],
    is_separator_regex=True,
    chunk_size=1000,
    chunk_overlap=200
)
</syntaxhighlight>

=== Supported Languages ===

<syntaxhighlight lang="python">
from langchain_text_splitters import Language

# Programming languages
languages = [
    Language.PYTHON,
    Language.JS,
    Language.TS,
    Language.JAVA,
    Language.GO,
    Language.RUST,
    Language.CPP,
    Language.C,
    Language.CSHARP,
    Language.RUBY,
    Language.PHP,
    Language.SWIFT,
    Language.KOTLIN,
    Language.SCALA,
    Language.ELIXIR,
    Language.HASKELL,
    Language.LUA,
    Language.POWERSHELL,
    Language.COBOL,
    Language.VISUALBASIC6,
    Language.SOL,  # Solidity
    Language.PROTO,  # Protocol Buffers
    Language.R,
]

# Markup/document languages
doc_languages = [
    Language.MARKDOWN,
    Language.HTML,
    Language.LATEX,
    Language.RST,  # ReStructuredText
]
</syntaxhighlight>

== I/O Contract ==

=== Input ===

RecursiveCharacterTextSplitter accepts:
* '''separators''' (list[str] | None) - List of separator patterns to try in order. Default: ["\n\n", "\n", " ", ""]
* '''is_separator_regex''' (bool) - Whether separators are regex patterns. Default: False
* '''keep_separator''' (bool | Literal["start", "end"]) - Where to keep separators in output. Default: True

get_separators_for_language() accepts:
* '''language''' (Language) - Language enum value for which to retrieve separators

=== Output ===

get_separators_for_language() returns:
* '''list[str]''' - Ordered list of separator patterns optimized for the language

The separators are tried in order during splitting, with earlier separators taking precedence.

== Usage Examples ==

=== Example 1: Python Code Splitting ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

python_code = """
class DataProcessor:
    def __init__(self, config):
        self.config = config

    def process(self, data):
        results = []
        for item in data:
            if item.is_valid():
                results.append(self.transform(item))
        return results

    def transform(self, item):
        return item.value * 2

def main():
    processor = DataProcessor(config)
    data = load_data()
    results = processor.process(data)
    save_results(results)
"""

# Python-specific splitter respects class and function boundaries
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0
)

chunks = splitter.split_text(python_code)
# Chunks will keep classes and methods intact when possible
</syntaxhighlight>

=== Example 2: JavaScript/TypeScript Splitting ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

js_code = """
function calculateTotal(items) {
    let total = 0;
    for (const item of items) {
        if (item.isActive) {
            total += item.price;
        }
    }
    return total;
}

class ShoppingCart {
    constructor() {
        this.items = [];
    }

    addItem(item) {
        this.items.push(item);
    }

    getTotal() {
        return calculateTotal(this.items);
    }
}

const cart = new ShoppingCart();
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS,
    chunk_size=400,
    chunk_overlap=50
)

chunks = splitter.split_text(js_code)
</syntaxhighlight>

=== Example 3: Markdown Document Splitting ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

markdown_doc = """
# Main Title

This is the introduction paragraph.

## Section 1

Content for section 1 with multiple paragraphs.

This is the second paragraph in section 1.

### Subsection 1.1

Details for subsection 1.1.

## Section 2

Content for section 2.

```python
def example():
    pass
```

More content after the code block.
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=200,
    chunk_overlap=20
)

chunks = splitter.split_text(markdown_doc)
# Splits at headers, then code blocks, then paragraphs
</syntaxhighlight>

=== Example 4: Custom Separators for Domain-specific Format ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Custom format: sections marked with "---"
document = """
Section: Introduction
---
This is the introduction content.
Multiple paragraphs here.

Section: Methods
---
This describes the methods.
More details about methodology.

Section: Results
---
Results are presented here.
"""

# Custom separators for this format
splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\nSection:",  # Split on section markers
        "---",         # Split on section dividers
        "\n\n",        # Then paragraphs
        "\n",          # Then lines
        " ",           # Then words
        ""
    ],
    chunk_size=150,
    chunk_overlap=20
)

chunks = splitter.split_text(document)
</syntaxhighlight>

=== Example 5: Viewing Language-specific Separators ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Inspect Python separators
python_seps = RecursiveCharacterTextSplitter.get_separators_for_language(
    Language.PYTHON
)
print("Python separators:", python_seps)
# ["\nclass ", "\ndef ", "\n\tdef ", "\n\n", "\n", " ", ""]

# Inspect JavaScript separators
js_seps = RecursiveCharacterTextSplitter.get_separators_for_language(
    Language.JS
)
print("JavaScript separators:", js_seps)
# ["\nfunction ", "\nconst ", "\nlet ", "\nvar ", "\nclass ",
#  "\nif ", "\nfor ", "\nwhile ", "\nswitch ", "\ncase ",
#  "\ndefault ", "\n\n", "\n", " ", ""]

# Inspect Markdown separators
md_seps = RecursiveCharacterTextSplitter.get_separators_for_language(
    Language.MARKDOWN
)
print("Markdown separators:", md_seps)
# ["\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n",
#  "\n___+\n", "\n\n", "\n", " ", ""]
</syntaxhighlight>

=== Example 6: Regex Separators for Flexible Matching ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Use regex for flexible separator matching
splitter = RecursiveCharacterTextSplitter(
    separators=[
        r"\n#+\s+",      # Markdown headers (any level)
        r"\n\n+",        # Multiple blank lines
        r"[.!?]\s+",     # Sentence boundaries
        r"\s+",          # Any whitespace
        ""
    ],
    is_separator_regex=True,
    chunk_size=500,
    chunk_overlap=50
)
</syntaxhighlight>

=== Example 7: Separator Preservation Options ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = "def foo():\n    pass\n\ndef bar():\n    pass"

# Keep separators at start (default for language splitters)
splitter1 = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=50,
    chunk_overlap=0
    # keep_separator=True by default
)
chunks1 = splitter1.split_text(text)
# Chunks start with "\ndef "

# Keep separators at end
splitter2 = RecursiveCharacterTextSplitter(
    separators=["\ndef ", "\n", " ", ""],
    keep_separator="end",
    is_separator_regex=True,
    chunk_size=50,
    chunk_overlap=0
)
chunks2 = splitter2.split_text(text)
# Chunks end with "\ndef "

# Discard separators
splitter3 = RecursiveCharacterTextSplitter(
    separators=["\ndef ", "\n", " ", ""],
    keep_separator=False,
    is_separator_regex=True,
    chunk_size=50,
    chunk_overlap=0
)
chunks3 = splitter3.split_text(text)
# Separators removed from chunks
</syntaxhighlight>

== Related Pages ==

* [[langchain-ai_langchain_Separator_Configuration|Separator_Configuration]] - Principle of configuring separators
* [[langchain-ai_langchain_text_splitter_types|text_splitter_types]] - Text splitter implementations
* [[langchain-ai_langchain_chunk_parameters|chunk_parameters]] - Configuring chunk size and overlap
* [[langchain-ai_langchain_split_text_method|split_text_method]] - Using splitters to split text
