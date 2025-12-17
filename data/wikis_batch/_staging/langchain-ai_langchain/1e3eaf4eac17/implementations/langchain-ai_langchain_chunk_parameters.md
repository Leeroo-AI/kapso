= chunk_parameters =

'''Sources:''' /tmp/praxium_repo_wjjl6pl8/libs/text-splitters/langchain_text_splitters/base.py:L47-86

'''Domains:''' Text Splitting, RAG, Document Processing, Parameter Tuning

'''Last Updated:''' 2025-12-17

== Overview ==

The chunk_parameters implementation provides the configuration options for controlling text chunk size and overlap in LangChain text splitters. These parameters are defined in the TextSplitter base class constructor and affect all splitting operations.

== Description ==

All LangChain text splitters inherit from TextSplitter and accept the same core configuration parameters. These parameters control:

* How large each chunk can be (chunk_size)
* How much adjacent chunks overlap (chunk_overlap)
* How chunk length is measured (length_function)
* Whether separators are preserved in output (keep_separator)
* Whether to track chunk positions (add_start_index)
* Whether to strip whitespace from chunks (strip_whitespace)

The implementation validates parameters on initialization to prevent invalid configurations that could cause infinite loops or errors.

== Code Reference ==

The implementation is in libs/text-splitters/langchain_text_splitters/base.py:

<syntaxhighlight lang="python">
class TextSplitter(BaseDocumentTransformer, ABC):
    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        keep_separator: bool | Literal["start", "end"] = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
    ) -> None:
        # Validation
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace
</syntaxhighlight>

== API ==

=== Basic Parameters ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Default configuration
splitter = RecursiveCharacterTextSplitter()
# chunk_size=4000, chunk_overlap=200

# Custom chunk size and overlap
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
</syntaxhighlight>

=== Token-based Sizing ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Use tiktoken to measure in tokens instead of characters
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=1000,  # tokens, not characters
    chunk_overlap=200
)

# Or specify a model name
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=8000,  # max tokens for GPT-4
    chunk_overlap=1000
)
</syntaxhighlight>

=== Custom Length Function ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Custom length function (e.g., count words instead of characters)
def word_count(text: str) -> int:
    return len(text.split())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # 100 words
    chunk_overlap=20,  # 20 word overlap
    length_function=word_count
)
</syntaxhighlight>

=== Separator Handling ===

<syntaxhighlight lang="python">
from langchain_text_splitters import CharacterTextSplitter

# Don't keep separators (default)
splitter = CharacterTextSplitter(
    separator="\n\n",
    keep_separator=False
)

# Keep separators at start of each chunk
splitter = CharacterTextSplitter(
    separator="\n\n",
    keep_separator="start"
)

# Keep separators at end of each chunk
splitter = CharacterTextSplitter(
    separator="\n\n",
    keep_separator="end"
)
</syntaxhighlight>

=== Metadata Options ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Add start index to metadata
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True  # Adds "start_index" to chunk metadata
)

# Control whitespace stripping
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    strip_whitespace=False  # Keep leading/trailing whitespace
)
</syntaxhighlight>

== I/O Contract ==

=== Input ===

All parameters are set during splitter initialization:

* '''chunk_size''' (int, default: 4000) - Maximum size of chunks. Must be > 0.
* '''chunk_overlap''' (int, default: 200) - Overlap between chunks. Must be >= 0 and < chunk_size.
* '''length_function''' (Callable[[str], int], default: len) - Function to measure chunk length.
* '''keep_separator''' (bool | Literal["start", "end"], default: False) - Whether/where to keep separators.
* '''add_start_index''' (bool, default: False) - Whether to add start_index to metadata.
* '''strip_whitespace''' (bool, default: True) - Whether to strip whitespace from chunks.

=== Output ===

Parameters affect the splitter's behavior:

* '''chunk_size''' and '''chunk_overlap''' determine the size and overlap of returned chunks
* '''length_function''' changes how size is measured (characters, tokens, words, etc.)
* '''add_start_index''' adds metadata to Document objects when using split_documents()
* Other parameters affect chunk content and formatting

=== Validation ===

The constructor validates:
* chunk_size > 0 (raises ValueError if not)
* chunk_overlap >= 0 (raises ValueError if not)
* chunk_overlap < chunk_size (raises ValueError if not)

== Usage Examples ==

=== Example 1: Standard Configuration for General Text ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Recommended settings for general documentation/articles
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # ~250 tokens
    chunk_overlap=200,    # ~50 tokens, 20% overlap
    length_function=len,
    add_start_index=True
)

text = "Your long document text..."
chunks = splitter.split_text(text)
</syntaxhighlight>

=== Example 2: Token-based Configuration for LLM Context ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ensure chunks fit in GPT-4 Turbo context (128k tokens)
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=4000,      # tokens
    chunk_overlap=400,    # tokens, 10% overlap
    add_start_index=True
)

# Each chunk is guaranteed to be <= 4000 tokens
chunks = splitter.split_text(text)
</syntaxhighlight>

=== Example 3: Minimal Overlap for QA Systems ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Precise, focused chunks for question answering
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # Small chunks for precise answers
    chunk_overlap=50,     # Minimal overlap (10%)
    strip_whitespace=True,
    add_start_index=True
)
</syntaxhighlight>

=== Example 4: Large Chunks for Summarization ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Large chunks to preserve narrative context
splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,      # Large chunks
    chunk_overlap=600,    # 20% overlap for continuity
    strip_whitespace=True
)
</syntaxhighlight>

=== Example 5: Custom Word-based Splitting ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

def word_length(text: str) -> int:
    """Count words instead of characters."""
    return len(text.split())

# Split by word count instead of character count
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,           # 200 words per chunk
    chunk_overlap=40,         # 40 word overlap
    length_function=word_length
)
</syntaxhighlight>

=== Example 6: Configuration with Start Index Tracking ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True  # Add position tracking
)

docs = [Document(page_content="Long text...", metadata={"source": "file.txt"})]
chunks = splitter.split_documents(docs)

# Each chunk has start_index in metadata
for chunk in chunks:
    print(f"Position {chunk.metadata['start_index']}: {chunk.page_content[:50]}...")
</syntaxhighlight>

=== Example 7: Validation Examples ===

<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# This will raise ValueError: chunk_size must be > 0
try:
    splitter = RecursiveCharacterTextSplitter(chunk_size=0)
except ValueError as e:
    print(f"Error: {e}")

# This will raise ValueError: chunk_overlap must be >= 0
try:
    splitter = RecursiveCharacterTextSplitter(chunk_overlap=-1)
except ValueError as e:
    print(f"Error: {e}")

# This will raise ValueError: overlap larger than chunk_size
try:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=200
    )
except ValueError as e:
    print(f"Error: {e}")
</syntaxhighlight>

== Related Pages ==

* [[langchain-ai_langchain_Chunk_Configuration|Chunk_Configuration]] - Principle of determining optimal chunk parameters
* [[langchain-ai_langchain_text_splitter_types|text_splitter_types]] - Different text splitter implementations
* [[langchain-ai_langchain_split_text_method|split_text_method]] - Using configured splitters on text
* [[langchain-ai_langchain_Separator_Configuration|Separator_Configuration]] - Configuring separator patterns
