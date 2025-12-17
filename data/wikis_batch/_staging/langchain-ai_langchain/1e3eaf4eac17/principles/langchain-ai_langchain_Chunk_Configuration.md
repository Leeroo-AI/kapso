= Chunk Configuration =

'''Sources:''' /tmp/praxium_repo_wjjl6pl8/libs/text-splitters/langchain_text_splitters/base.py:L47-86

'''Domains:''' Text Splitting, RAG, Document Processing, Parameter Tuning

'''Last Updated:''' 2025-12-17

== Overview ==

Chunk Configuration is the principle of determining optimal chunk size and overlap parameters for text splitting. These parameters control how much content fits in each chunk and how much context is shared between adjacent chunks, directly impacting retrieval quality and model performance in RAG applications.

== Description ==

Text splitting requires careful configuration of two fundamental parameters:

'''Chunk Size:''' The maximum length of each text chunk, measured in either:
* Characters (default) - Simple character count
* Tokens - More accurate for LLM context limits
* Custom metrics - Via length_function parameter

'''Chunk Overlap:''' The number of characters/tokens that adjacent chunks share, providing:
* Context continuity across chunk boundaries
* Better semantic coherence for spanning concepts
* Improved retrieval when queries match boundary content

Additional configuration options:

'''Length Function:''' Customizable function to measure chunk length:
* Default: len() counts characters
* Token-based: tiktoken or transformers tokenizers
* Custom: Any callable that returns an integer

'''Separator Handling:''' Control how separators appear in chunks:
* keep_separator=False - Remove separators between chunks
* keep_separator=True or "start" - Keep separator at chunk start
* keep_separator="end" - Keep separator at chunk end

'''Whitespace Stripping:''' Whether to strip leading/trailing whitespace from chunks (default: True)

'''Start Index Tracking:''' Add chunk's starting position to metadata (default: False)

== Theoretical Basis ==

The effectiveness of chunk configuration depends on understanding several tradeoffs:

'''Chunk Size Tradeoffs:'''

Larger chunks:
* Preserve more context and semantic coherence
* Reduce total number of chunks to search
* May exceed model context windows
* Can include less relevant information

Smaller chunks:
* Provide more precise retrieval matches
* Fit more easily within context limits
* May break semantic units
* Increase total chunks and retrieval time

'''Overlap Tradeoffs:'''

More overlap:
* Better handles concepts spanning boundaries
* Provides context redundancy for queries
* Increases storage requirements
* May cause duplicate retrieval

Less overlap:
* More efficient storage
* Faster indexing
* Risk of losing boundary context
* May miss cross-chunk relationships

'''Common Configuration Patterns:'''

'''For general text (articles, documentation):'''
* Chunk size: 1000-1500 characters (250-400 tokens)
* Overlap: 200 characters (50 tokens) or 10-20% of chunk size

'''For code:'''
* Chunk size: 1500-2000 characters (preserve functions/classes)
* Overlap: 200-400 characters (capture cross-function context)

'''For QA systems:'''
* Chunk size: 500-1000 characters (focused, precise answers)
* Overlap: 100-200 characters (minimal redundancy)

'''For summarization:'''
* Chunk size: 2000-4000 characters (preserve narrative flow)
* Overlap: 400-800 characters (continuity across chunks)

'''Token-based Measurement:'''

Using token-based chunk sizing is critical when:
* Targeting specific LLM context windows
* Dealing with multi-byte characters (non-English text)
* Ensuring accurate context limit compliance

The relationship between characters and tokens varies:
* English: ~4 characters per token on average
* Code: ~3.5 characters per token
* Non-English: can be 1-2 characters per token

'''Validation Rules:'''

The TextSplitter enforces:
* chunk_size must be > 0
* chunk_overlap must be >= 0
* chunk_overlap must be < chunk_size (otherwise infinite loops)

These constraints ensure valid splitting behavior.

== Related Pages ==

* [[langchain-ai_langchain_chunk_parameters|chunk_parameters]] - Implementation of chunk size and overlap parameters
* [[langchain-ai_langchain_Splitter_Selection|Splitter_Selection]] - Choosing appropriate text splitters
* [[langchain-ai_langchain_Separator_Configuration|Separator_Configuration]] - Configuring separator patterns
* [[langchain-ai_langchain_Document_Splitting|Document_Splitting]] - Applying splitting to documents
