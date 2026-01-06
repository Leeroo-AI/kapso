{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Text Splitters|https://docs.langchain.com/oss/python/langchain/overview]]
|-
! Domains
| [[domain::NLP]], [[domain::RAG]], [[domain::Document_Processing]], [[domain::Chunking]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Concrete tool for splitting text recursively using a hierarchy of separators until chunks meet size requirements, provided by LangChain text-splitters.

=== Description ===

`RecursiveCharacterTextSplitter` is LangChain's most versatile text splitter. It attempts to split text using a prioritized list of separators (e.g., paragraphs → sentences → words → characters), recursively trying smaller separators when larger ones produce chunks exceeding the size limit.

Key features:
* **Recursive splitting:** Falls back to finer separators when needed
* **Language-aware:** `from_language()` provides separators for 30+ programming languages
* **Configurable:** Custom separator lists for domain-specific text
* **Overlap support:** Maintains context between chunks

=== Usage ===

Use `RecursiveCharacterTextSplitter` when:
* Splitting general text for RAG pipelines
* Processing documents for LLM context windows
* Chunking code with language-aware splitting
* Need semantic boundaries (paragraphs, sentences) respected

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain]
* '''File:''' libs/text-splitters/langchain_text_splitters/character.py
* '''Lines:''' L81-169

=== Signature ===
<syntaxhighlight lang="python">
class RecursiveCharacterTextSplitter(TextSplitter):
    """Splitting text by recursively looking at characters.

    Recursively tries to split by different characters to find one
    that works.
    """

    def __init__(
        self,
        separators: list[str] | None = None,
        keep_separator: bool | Literal["start", "end"] = True,
        is_separator_regex: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create a new RecursiveCharacterTextSplitter.

        Args:
            separators: List of separators to try in order (default: ["\\n\\n", "\\n", " ", ""])
            keep_separator: Keep separator in chunks (True="start", False=drop, "end"=append)
            is_separator_regex: Treat separators as regex patterns
            **kwargs: TextSplitter args (chunk_size, chunk_overlap, length_function, etc.)
        """

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks using recursive separator strategy."""

    @classmethod
    def from_language(
        cls,
        language: Language,
        **kwargs: Any,
    ) -> RecursiveCharacterTextSplitter:
        """Create splitter with language-specific separators.

        Args:
            language: Programming language (Language enum)
            **kwargs: Additional splitter configuration

        Returns:
            Configured splitter for the specified language.
        """

    @staticmethod
    def get_separators_for_language(language: Language) -> list[str]:
        """Get separator list for a specific language."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Or with Language enum:
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
</syntaxhighlight>

== I/O Contract ==

=== Inputs (Constructor) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| separators || list[str] | None || No || Ordered separator list (default: ["\\n\\n", "\\n", " ", ""])
|-
| keep_separator || bool | "start" | "end" || No || How to handle separators in output
|-
| is_separator_regex || bool || No || Treat separators as regex patterns
|-
| chunk_size || int || No || Maximum chunk size (default: 4000)
|-
| chunk_overlap || int || No || Overlap between chunks (default: 200)
|-
| length_function || Callable[[str], int] || No || Function to measure length (default: len)
|}

=== Inputs (split_text) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| text || str || Yes || Text to split into chunks
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return || list[str] || List of text chunks respecting size constraints
|}

== Usage Examples ==

=== Basic Text Splitting ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

text = """
Long document text here...

Multiple paragraphs with various content.

Each paragraph may be split further if needed.
"""

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {len(chunk)} chars")
</syntaxhighlight>

=== Language-Aware Code Splitting ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Create Python-aware splitter
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=2000,
    chunk_overlap=200,
)

python_code = '''
class MyClass:
    """A sample class."""

    def method_one(self):
        pass

    def method_two(self):
        pass


def standalone_function():
    """A standalone function."""
    return 42
'''

chunks = python_splitter.split_text(python_code)
# Splits at class/function boundaries when possible
</syntaxhighlight>

=== Custom Separators ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Custom separators for markdown
markdown_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n## ",      # H2 headers
        "\n### ",     # H3 headers
        "\n#### ",    # H4 headers
        "\n\n",       # Paragraphs
        "\n",         # Lines
        " ",          # Words
        "",           # Characters
    ],
    chunk_size=1500,
    chunk_overlap=100,
)
</syntaxhighlight>

=== Splitting Documents ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# Split Document objects (preserves metadata)
documents = [
    Document(page_content="First document content...", metadata={"source": "doc1.txt"}),
    Document(page_content="Second document content...", metadata={"source": "doc2.txt"}),
]

split_docs = splitter.split_documents(documents)
# Each chunk inherits metadata from parent document
</syntaxhighlight>

=== Available Languages ===
<syntaxhighlight lang="python">
from langchain_text_splitters import Language

# Supported languages include:
supported = [
    Language.PYTHON,
    Language.JS,
    Language.TS,
    Language.GO,
    Language.RUST,
    Language.JAVA,
    Language.CPP,
    Language.C,
    Language.CSHARP,
    Language.PHP,
    Language.RUBY,
    Language.SWIFT,
    Language.KOTLIN,
    Language.SCALA,
    Language.MARKDOWN,
    Language.HTML,
    Language.LATEX,
    Language.SOL,  # Solidity
    # ... and more
]
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:langchain-ai_langchain_Splitter_Strategy_Selection]]

=== Requires Environment ===
* [[requires_env::Environment:langchain-ai_langchain_Text_Splitters_Environment]]
