{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|LangChain Text Splitters|https://docs.langchain.com/oss/python/langchain/overview]]
* [[source::Blog|Chunking Strategies|https://www.pinecone.io/learn/chunking-strategies/]]
|-
! Domains
| [[domain::NLP]], [[domain::RAG]], [[domain::Document_Processing]], [[domain::Chunking]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Design pattern for selecting the appropriate text splitting algorithm based on document type, content structure, and downstream requirements.

=== Description ===

Splitter Strategy Selection is the decision process for choosing how to divide documents into chunks for retrieval-augmented generation (RAG) and LLM processing. The choice of splitter significantly impacts:
* **Retrieval quality:** Semantic coherence of chunks affects search results
* **LLM comprehension:** Well-bounded chunks improve model understanding
* **Token efficiency:** Appropriate sizing maximizes context utilization

Different content types require different approaches:
* **Prose:** Paragraph and sentence boundaries
* **Code:** Function, class, and block boundaries
* **Structured formats:** HTML headers, JSON keys, markdown sections
* **Technical documents:** LaTeX sections, tables, equations

=== Usage ===

Use Splitter Strategy Selection when:
* Building RAG pipelines
* Preparing documents for LLM context
* Processing heterogeneous document collections
* Optimizing retrieval quality

Strategy decision factors:
* Document format (text, code, HTML, markdown, etc.)
* Content structure (prose, lists, tables, etc.)
* Downstream task (Q&A, summarization, code analysis)
* Model context window and tokenization

== Theoretical Basis ==

Splitter Strategy Selection implements **Domain-Specific Chunking** strategies.

'''1. Strategy Taxonomy'''

<syntaxhighlight lang="python">
# Pseudo-code for splitter selection
class SplitterType(Enum):
    CHARACTER = "character"           # Split by character count
    RECURSIVE_CHARACTER = "recursive" # Recursive separator fallback
    TOKEN = "token"                   # Split by token count
    SENTENCE = "sentence"             # NLP sentence segmentation
    SEMANTIC = "semantic"             # Embedding-based similarity
    CODE = "code"                     # Language-aware code splitting
    STRUCTURED = "structured"         # Format-aware (HTML, Markdown, JSON)

def select_splitter(doc_type: str, use_case: str) -> TextSplitter:
    if doc_type == "code":
        return RecursiveCharacterTextSplitter.from_language(language)
    elif doc_type == "html":
        return HTMLSectionSplitter()
    elif doc_type == "markdown":
        return MarkdownHeaderTextSplitter()
    elif use_case == "embedding":
        return SentenceTransformersTokenTextSplitter()
    else:
        return RecursiveCharacterTextSplitter()
</syntaxhighlight>

'''2. Recursive Splitting Algorithm'''

The core recursive algorithm:

<syntaxhighlight lang="python">
# Pseudo-code for recursive splitting
def recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    if not separators:
        # No more separators - return text as-is (may exceed chunk_size)
        return [text]

    separator = separators[0]
    remaining_separators = separators[1:]

    # Split by current separator
    splits = text.split(separator)

    final_chunks = []
    current_chunk = []

    for split in splits:
        if len(join(current_chunk + [split])) <= chunk_size:
            current_chunk.append(split)
        else:
            # Flush current chunk
            if current_chunk:
                final_chunks.append(join(current_chunk))
            # Handle oversized split
            if len(split) > chunk_size:
                # Recursively split with finer separator
                final_chunks.extend(recursive_split(split, remaining_separators, chunk_size))
                current_chunk = []
            else:
                current_chunk = [split]

    if current_chunk:
        final_chunks.append(join(current_chunk))

    return final_chunks
</syntaxhighlight>

'''3. Language-Specific Separators'''

Each programming language has semantic boundaries:

<syntaxhighlight lang="python">
# Pseudo-code for language separators
LANGUAGE_SEPARATORS = {
    "python": [
        "\nclass ",      # Class definitions
        "\ndef ",        # Function definitions
        "\n\n",          # Paragraph breaks
        "\n",            # Line breaks
        " ",             # Words
        "",              # Characters
    ],
    "javascript": [
        "\nfunction ",   # Function declarations
        "\nconst ",      # Const declarations
        "\nlet ",        # Let declarations
        "\nclass ",      # Class declarations
        "\n\n",
        "\n",
        " ",
        "",
    ],
    "html": [
        "<div",          # Div elements
        "<section",      # Section elements
        "<article",      # Article elements
        "<p>",           # Paragraphs
        "<br>",          # Line breaks
        " ",
        "",
    ],
}
</syntaxhighlight>

'''4. Chunk Overlap Strategy'''

Overlap maintains context across chunk boundaries:

<syntaxhighlight lang="python">
# Pseudo-code for overlap handling
def merge_splits_with_overlap(splits: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    current = []
    current_length = 0

    for split in splits:
        split_length = len(split)

        if current_length + split_length > chunk_size and current:
            # Emit current chunk
            chunks.append("".join(current))

            # Keep overlap portion
            overlap_text = ""
            while current and len(overlap_text) < overlap:
                overlap_text = current.pop() + overlap_text

            current = [overlap_text] if overlap_text else []
            current_length = len(overlap_text)

        current.append(split)
        current_length += split_length

    if current:
        chunks.append("".join(current))

    return chunks
</syntaxhighlight>

'''5. Selection Decision Tree'''

<syntaxhighlight lang="text">
                    Document Type?
                         │
         ┌───────────────┼───────────────┐
         │               │               │
       Code           Prose         Structured
         │               │               │
    from_language()  Recursive      Format-specific
         │               │               │
    ┌────┴────┐     ┌────┴────┐     ┌────┴────┐
  Python   JS/TS  Sentence Token  HTML  Markdown JSON
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:langchain-ai_langchain_RecursiveCharacterTextSplitter]]

=== Used By Workflows ===
* Text_Splitting_Workflow (Step 1)
