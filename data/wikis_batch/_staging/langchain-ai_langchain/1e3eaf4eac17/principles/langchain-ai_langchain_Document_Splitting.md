= Document Splitting =

'''Sources:''' /tmp/praxium_repo_wjjl6pl8/libs/text-splitters/langchain_text_splitters/character.py:L100-151, /tmp/praxium_repo_wjjl6pl8/libs/text-splitters/langchain_text_splitters/base.py:L87-117

'''Domains:''' Text Splitting, RAG, Document Processing, Information Retrieval

'''Last Updated:''' 2025-12-17

== Overview ==

Document Splitting is the principle of transforming long-form text and documents into smaller, manageable chunks suitable for retrieval and language model processing. This fundamental operation enables effective Retrieval Augmented Generation (RAG) by creating semantically coherent chunks that can be indexed, searched, and provided as context to language models.

== Description ==

Document splitting addresses a core challenge in working with language models: text that exceeds context windows or contains more information than needed for a specific query. The principle encompasses:

'''Text-level Splitting:''' Converting raw strings into lists of smaller strings, each representing a semantically meaningful chunk.

'''Document-level Splitting:''' Transforming Document objects (which contain both content and metadata) into multiple smaller Document objects while preserving metadata.

The splitting process follows these stages:

'''1. Initial Split:''' Apply separators to divide text at natural boundaries
'''2. Chunk Merging:''' Combine small pieces until they approach the target chunk size
'''3. Overlap Addition:''' Include overlapping content from adjacent chunks
'''4. Metadata Propagation:''' Carry forward document metadata to all chunks

Key considerations in document splitting:

'''Semantic Coherence:''' Chunks should represent complete thoughts or logical units. Splitting mid-sentence or mid-function reduces usefulness for retrieval.

'''Context Preservation:''' Overlap between chunks ensures that concepts spanning boundaries are preserved. Without overlap, queries matching boundary content may miss relevant information.

'''Size Optimization:''' Chunks must be:
* Large enough to contain meaningful context
* Small enough to fit in retrieval systems and model context windows
* Consistent enough for fair comparison during retrieval

'''Metadata Retention:''' Source information, chapter numbers, timestamps, and other metadata must propagate to chunks so retrieved content can be traced back to its origin.

== Theoretical Basis ==

The principle of document splitting is grounded in several key concepts from information retrieval and natural language processing:

'''Chunking for Dense Retrieval:'''

Modern RAG systems use dense vector embeddings to represent and retrieve text. The quality of these embeddings depends on chunk characteristics:

* '''Optimal Embedding Size:''' Embedding models are trained on specific text lengths (typically 256-512 tokens). Chunks of similar size produce more consistent embeddings.

* '''Semantic Density:''' Each chunk should represent a coherent concept or topic. Mixed topics in a single chunk produce ambiguous embeddings that match poorly with queries.

* '''Context Sufficiency:''' Chunks must contain enough information to be understandable independently. A chunk with "it was successful" lacks meaning without knowing what "it" refers to.

'''Recursive Splitting Strategy:'''

The recursive approach implements a hierarchical decomposition:

1. '''Coarse-grained Split:''' First try large semantic boundaries (paragraphs, sections)
2. '''Refinement:''' If chunks exceed size limits, recursively split using finer boundaries
3. '''Base Case:''' Continue until all chunks meet constraints

This preserves maximum semantic coherence while guaranteeing size requirements.

The algorithm maintains:
* '''Semantic Preference:''' Higher-level separators (paragraphs) preferred over lower-level (words)
* '''Size Guarantee:''' All output chunks satisfy chunk_size constraint
* '''Context Continuity:''' Overlap ensures no information loss at boundaries

'''Overlap as Context Window:'''

Overlap serves multiple purposes:

* '''Query Matching:''' Concepts near chunk boundaries appear in multiple chunks, increasing retrieval probability
* '''Context Completion:''' Overlapping content provides context for understanding chunks independently
* '''Boundary Smoothing:''' Reduces artifacts from arbitrary splits

The optimal overlap is typically 10-20% of chunk size:
* Too little: Risk of losing boundary context
* Too much: Redundant storage and retrieval, diluted results

'''Document vs Text Splitting:'''

Two distinct operations serve different needs:

'''split_text(text: str) -> list[str]:'''
* Input: Plain string
* Output: List of string chunks
* Use case: Simple text processing, no metadata needed

'''split_documents(docs: list[Document]) -> list[Document]:'''
* Input: Document objects with content and metadata
* Output: Document chunks with preserved/augmented metadata
* Use case: RAG pipelines where source tracking is essential

Document splitting enables:
* Source attribution in generated responses
* Filtering by metadata during retrieval
* Tracking chunk origins for debugging
* Augmenting metadata with chunk-specific info (start_index)

'''Metadata Propagation Pattern:'''

When splitting documents:
1. Extract text and metadata from each input document
2. Split text into chunks
3. Create new Document for each chunk
4. Copy original metadata to each chunk document
5. Optionally add chunk-specific metadata (start_index)

This ensures every chunk retains provenance information.

'''Integration with RAG Pipeline:'''

Document splitting is typically the first step in a RAG pipeline:

1. '''Splitting:''' Documents -> Chunks
2. '''Embedding:''' Chunks -> Vectors
3. '''Indexing:''' Vectors -> Vector DB
4. '''Retrieval:''' Query -> Relevant chunks
5. '''Generation:''' Chunks + Query -> Response

The quality of splitting directly impacts:
* Retrieval precision (finding relevant chunks)
* Context quality (providing useful information to LLM)
* Response accuracy (LLM has right information to answer)

Poor splitting (e.g., breaking sentences, losing context) propagates through the entire pipeline, degrading end-to-end performance.

== Related Pages ==

* [[langchain-ai_langchain_split_text_method|split_text_method]] - Implementation of splitting methods
* [[langchain-ai_langchain_Metadata_Preservation|Metadata_Preservation]] - Preserving metadata during splits
* [[langchain-ai_langchain_Chunk_Configuration|Chunk_Configuration]] - Configuring chunk size and overlap
* [[langchain-ai_langchain_Splitter_Selection|Splitter_Selection]] - Choosing appropriate splitters
