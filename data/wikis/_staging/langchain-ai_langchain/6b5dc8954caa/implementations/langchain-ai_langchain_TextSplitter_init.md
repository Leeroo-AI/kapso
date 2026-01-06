{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Text Splitters|https://docs.langchain.com/oss/python/langchain/overview]]
|-
! Domains
| [[domain::NLP]], [[domain::RAG]], [[domain::Document_Processing]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Concrete tool for configuring chunk size and overlap parameters in text splitters, provided by LangChain text-splitters.

=== Description ===

`TextSplitter.__init__` is the base constructor that configures fundamental chunking parameters inherited by all splitter implementations. These parameters control:
* Maximum chunk size
* Overlap between consecutive chunks
* How length is measured
* Separator handling
* Metadata generation

Getting these parameters right is critical for RAG quality and LLM context utilization.

=== Usage ===

Use these parameters when:
* Configuring any TextSplitter subclass
* Tuning RAG retrieval quality
* Optimizing for specific model context windows
* Balancing retrieval precision vs. recall

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain]
* '''File:''' libs/text-splitters/langchain_text_splitters/base.py
* '''Lines:''' L47-85

=== Signature ===
<syntaxhighlight lang="python">
class TextSplitter(ABC):
    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        keep_separator: bool | Literal["start", "end"] = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
    ) -> None:
        """Create a new TextSplitter.

        Args:
            chunk_size: Maximum size of chunks to return (must be > 0)
            chunk_overlap: Overlap in characters between chunks (must be >= 0 and < chunk_size)
            length_function: Function that measures the length of given chunks
            keep_separator: Whether to keep the separator and where to place it
                           (True='start', False=drop, 'end'=append)
            add_start_index: If True, includes chunk's start index in metadata
            strip_whitespace: If True, strips whitespace from chunk start and end

        Raises:
            ValueError: If chunk_size <= 0
            ValueError: If chunk_overlap < 0
            ValueError: If chunk_overlap > chunk_size
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| chunk_size || int || No || Maximum chunk size (default: 4000)
|-
| chunk_overlap || int || No || Overlap between chunks (default: 200)
|-
| length_function || Callable[[str], int] || No || Length measurement function (default: len)
|-
| keep_separator || bool | "start" | "end" || No || Separator handling (default: False)
|-
| add_start_index || bool || No || Include start_index in metadata (default: False)
|-
| strip_whitespace || bool || No || Strip chunk whitespace (default: True)
|}

=== Validation Rules ===
{| class="wikitable"
|-
! Rule !! Error
|-
| chunk_size > 0 || ValueError: "chunk_size must be > 0"
|-
| chunk_overlap >= 0 || ValueError: "chunk_overlap must be >= 0"
|-
| chunk_overlap < chunk_size || ValueError: "chunk overlap larger than chunk size"
|}

== Usage Examples ==

=== Basic Configuration ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Standard configuration for general text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Maximum 1000 characters per chunk
    chunk_overlap=200,    # 200 character overlap between chunks
)

chunks = splitter.split_text(long_document)
</syntaxhighlight>

=== RAG-Optimized Configuration ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration for RAG with embeddings
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,        # Smaller chunks = more precise retrieval
    chunk_overlap=50,      # Minimal overlap to reduce redundancy
    add_start_index=True,  # Track position for citation
)

chunks = splitter.split_documents(documents)
# Each chunk has metadata["start_index"] for reference
</syntaxhighlight>

=== High-Overlap for Context Continuity ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# High overlap for long-form analysis
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=500,    # 25% overlap ensures context continuity
)
</syntaxhighlight>

=== Keep Separator Configuration ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Keep separator at start of chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    keep_separator="start",  # "\n\nParagraph text..." instead of "Paragraph text..."
)

# Keep separator at end of chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    keep_separator="end",    # "...text.\n\n" instead of "...text."
)

# Drop separators (default)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    keep_separator=False,    # Separators removed from output
)
</syntaxhighlight>

=== Parameter Tuning Guidelines ===
<syntaxhighlight lang="python">
# chunk_size guidance:
# - 100-500: Fine-grained retrieval (Q&A, specific facts)
# - 500-1000: Balanced (general RAG)
# - 1000-2000: Broad context (summarization, analysis)
# - 2000+: Large context models (GPT-4o, Claude)

# chunk_overlap guidance:
# - 0: No overlap (smallest storage, risk context loss)
# - 10-20%: Light overlap (good balance)
# - 20-30%: Medium overlap (important for coherent retrieval)
# - 30%+: Heavy overlap (redundant but thorough)

# Example configurations:
configs = {
    "qa_precise": {"chunk_size": 300, "chunk_overlap": 30},
    "rag_balanced": {"chunk_size": 800, "chunk_overlap": 100},
    "analysis_broad": {"chunk_size": 1500, "chunk_overlap": 300},
}
</syntaxhighlight>

=== Validation Examples ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# These will raise ValueError:
try:
    splitter = RecursiveCharacterTextSplitter(chunk_size=0)
except ValueError as e:
    print(e)  # "chunk_size must be > 0"

try:
    splitter = RecursiveCharacterTextSplitter(chunk_overlap=-1)
except ValueError as e:
    print(e)  # "chunk_overlap must be >= 0"

try:
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=200)
except ValueError as e:
    print(e)  # "Got a larger chunk overlap (200) than chunk size (100)"
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:langchain-ai_langchain_Chunk_Size_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:langchain-ai_langchain_Text_Splitters_Environment]]
