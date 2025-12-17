# Workflow: Text Splitting for RAG

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Text Splitters|https://docs.langchain.com/oss/python/langchain/text-splitters]]
|-
! Domains
| [[domain::RAG]], [[domain::Document_Processing]], [[domain::NLP]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
Process for chunking documents into appropriately sized segments for retrieval-augmented generation (RAG) pipelines.

=== Description ===
This workflow describes how to use LangChain's text splitters to break documents into chunks suitable for embedding and retrieval. The `langchain_text_splitters` package provides multiple splitter types optimized for different content: recursive character splitting for general text, language-aware splitters for code, format-specific splitters for HTML/Markdown/JSON, and NLP-based splitters using NLTK or spaCy for sentence boundaries.

=== Usage ===
Execute this workflow when you need to:
* Prepare documents for embedding in a vector database
* Chunk code files while preserving function/class boundaries
* Split structured documents (HTML, Markdown, JSON) semantically
* Ensure chunks fit within LLM context windows
* Maintain metadata and overlap between chunks

== Execution Steps ==

=== Step 1: Select Splitter Type ===
[[step::Principle:langchain-ai_langchain_Splitter_Selection]]

Choose the appropriate splitter based on your content type. Options include:
* `RecursiveCharacterTextSplitter`: General-purpose, tries multiple separators
* `CharacterTextSplitter`: Simple single-separator splitting
* Language-specific splitters: Python, JavaScript, Markdown, HTML, etc.
* NLP splitters: NLTK, spaCy for linguistic sentence boundaries

'''Key considerations:'''
* Recursive splitting preserves structure better than simple character splits
* Code splitters use language-specific boundaries (class, function definitions)
* HTML/Markdown splitters respect document structure

=== Step 2: Configure Chunk Parameters ===
[[step::Principle:langchain-ai_langchain_Chunk_Configuration]]

Set the chunk size and overlap parameters based on your embedding model's token limit and retrieval requirements.

'''Key considerations:'''
* `chunk_size`: Maximum characters per chunk (typical: 500-2000)
* `chunk_overlap`: Characters to repeat at chunk boundaries (typical: 50-200)
* `length_function`: Custom function to measure chunk length (default: `len`)
* `keep_separator`: Whether to include separators in chunks (start/end placement)

=== Step 3: Configure Separators ===
[[step::Principle:langchain-ai_langchain_Separator_Configuration]]

For recursive splitters, configure the hierarchy of separators to try. The splitter attempts each separator in order until chunks are small enough. Language-specific separators can be obtained via `get_separators_for_language()`.

'''Key considerations:'''
* Default separators: `["\n\n", "\n", " ", ""]`
* `is_separator_regex`: Enable regex patterns for complex splitting
* Language separators prioritize semantic boundaries (class/function definitions)

=== Step 4: Split Documents ===
[[step::Principle:langchain-ai_langchain_Document_Splitting]]

Call the `split_text()` or `split_documents()` method to chunk your content. The splitter recursively tries separators, merges small chunks, and ensures each chunk respects the size limit.

'''What happens:'''
* Text is split on the first matching separator
* Chunks smaller than `chunk_size` are merged
* Chunks larger than `chunk_size` are recursively split with finer separators
* Overlap is added between adjacent chunks

=== Step 5: Preserve Metadata (Optional) ===
[[step::Principle:langchain-ai_langchain_Metadata_Preservation]]

When using `split_documents()` with `Document` objects, metadata is automatically propagated to all resulting chunks. Additional chunk-specific metadata (position, total chunks) can be added.

'''Key considerations:'''
* Source document metadata is copied to all chunks
* Custom transformations can add chunk index/count metadata
* Headers in Markdown/HTML can be extracted as metadata

== Execution Diagram ==
{{#mermaid:graph TD
    A[Select Splitter Type] --> B[Configure Chunk Parameters]
    B --> C[Configure Separators]
    C --> D[Split Documents]
    D --> E[Preserve Metadata]
}}

== Related Pages ==
* [[step::Principle:langchain-ai_langchain_Splitter_Selection]]
* [[step::Principle:langchain-ai_langchain_Chunk_Configuration]]
* [[step::Principle:langchain-ai_langchain_Separator_Configuration]]
* [[step::Principle:langchain-ai_langchain_Document_Splitting]]
* [[step::Principle:langchain-ai_langchain_Metadata_Preservation]]
