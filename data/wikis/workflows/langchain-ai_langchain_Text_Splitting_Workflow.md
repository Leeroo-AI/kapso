{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Text Splitters|https://docs.langchain.com/oss/python/langchain/text-splitters]]
|-
! Domains
| [[domain::NLP]], [[domain::RAG]], [[domain::Document_Processing]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
End-to-end process for splitting documents into semantic chunks optimized for retrieval and language model context windows using LangChain's text splitter utilities.

=== Description ===
This workflow covers the complete document chunking pipeline from raw text to indexed chunks. Text splitters break large documents into smaller, semantically coherent pieces that fit within model context windows while preserving meaningful boundaries. The library provides multiple strategies: character-based, recursive, language-aware, and format-specific (HTML, Markdown, JSON).

Key capabilities:
* Configurable chunk size and overlap
* Multiple length functions (character count, token count via tiktoken/HuggingFace)
* Language-specific separators for 30+ programming languages
* Format-aware splitting for HTML, Markdown, JSON, LaTeX
* Metadata preservation and start index tracking

=== Usage ===
Execute this workflow when preparing documents for:
* RAG (Retrieval-Augmented Generation) pipelines
* Vector store indexing
* Context window management
* Document summarization preprocessing

Typical inputs: PDFs, web pages, code files, documentation, chat logs.

== Execution Steps ==

=== Step 1: Splitter Selection ===
[[step::Principle:langchain-ai_langchain_Splitter_Strategy_Selection]]

Choose the appropriate text splitter based on document type and use case. The selection depends on content structure: use `RecursiveCharacterTextSplitter` for general text, language-specific splitters for code, or format-aware splitters for HTML/Markdown.

'''Selection criteria:'''
* General text: RecursiveCharacterTextSplitter (tries multiple separators)
* Code: RecursiveCharacterTextSplitter.from_language(Language.PYTHON, etc.)
* HTML: HTMLHeaderTextSplitter, HTMLSectionSplitter, or HTMLSemanticPreservingSplitter
* Markdown: MarkdownHeaderTextSplitter or ExperimentalMarkdownSyntaxTextSplitter
* JSON: RecursiveJsonSplitter (preserves structure)

=== Step 2: Length Function Configuration ===
[[step::Principle:langchain-ai_langchain_Length_Function_Setup]]

Configure the length measurement function based on your target model's tokenizer. Character count is fastest but token count ensures accurate fit within model context windows.

'''Options:'''
* `len` (default): Character count, fast but approximate
* `from_tiktoken_encoder()`: OpenAI-compatible token counting
* `from_huggingface_tokenizer()`: HuggingFace model-specific tokens
* Custom callable: Any function `str -> int`

=== Step 3: Chunk Parameters ===
[[step::Principle:langchain-ai_langchain_Chunk_Size_Configuration]]

Set chunk size and overlap parameters based on your embedding model and retrieval requirements. Larger chunks preserve more context but may exceed model limits or reduce retrieval precision.

'''Key parameters:'''
* `chunk_size`: Maximum length per chunk (default: 4000)
* `chunk_overlap`: Characters/tokens shared between adjacent chunks (default: 200)
* `keep_separator`: Whether to include separator in chunks (start/end/False)
* `strip_whitespace`: Remove leading/trailing whitespace

=== Step 4: Document Splitting ===
[[step::Principle:langchain-ai_langchain_Document_Transformation]]

Execute the split operation on input documents. The splitter applies its strategy recursively, trying each separator until chunks are small enough, then merges adjacent small pieces with overlap.

'''Core algorithm:'''
1. Split text using primary separator
2. If any piece exceeds chunk_size, recursively split with next separator
3. Merge small adjacent pieces until chunk_size is reached
4. Apply overlap by keeping trailing content from previous chunk

=== Step 5: Metadata Preservation ===
[[step::Principle:langchain-ai_langchain_Metadata_Handling]]

Preserve and augment document metadata through the splitting process. Original metadata is copied to each chunk, with optional start index tracking for document reconstruction.

'''Metadata features:'''
* Original metadata copied to all chunks
* `add_start_index=True`: Track character offset for each chunk
* Header-based splitters add hierarchical metadata (h1, h2, etc.)
* Custom metadata can be added during `create_documents()`

== Execution Diagram ==
{{#mermaid:graph TD
    A[Splitter Selection] --> B[Length Function Configuration]
    B --> C[Chunk Parameters]
    C --> D[Document Splitting]
    D --> E{Chunk Size OK?}
    E -->|No| F[Try Next Separator]
    F --> D
    E -->|Yes| G[Metadata Preservation]
    G --> H[Output Chunks]
}}

== Related Pages ==
* [[step::Principle:langchain-ai_langchain_Splitter_Strategy_Selection]]
* [[step::Principle:langchain-ai_langchain_Length_Function_Setup]]
* [[step::Principle:langchain-ai_langchain_Chunk_Size_Configuration]]
* [[step::Principle:langchain-ai_langchain_Document_Transformation]]
* [[step::Principle:langchain-ai_langchain_Metadata_Handling]]
