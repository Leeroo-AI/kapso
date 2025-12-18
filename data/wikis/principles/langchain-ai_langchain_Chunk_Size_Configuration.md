{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Blog|Chunking Best Practices|https://www.pinecone.io/learn/chunking-strategies/]]
* [[source::Doc|LangChain Text Splitters|https://docs.langchain.com/oss/python/langchain/overview]]
|-
! Domains
| [[domain::NLP]], [[domain::RAG]], [[domain::Information_Retrieval]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Configuration strategy for determining optimal chunk size and overlap parameters based on use case, model, and retrieval requirements.

=== Description ===

Chunk Size Configuration is the critical decision of how large document chunks should be and how much they should overlap. These parameters have profound effects on:
* **Retrieval precision:** Smaller chunks = more specific results
* **Context coherence:** Larger chunks = more complete context
* **Storage efficiency:** Smaller chunks with less overlap = less redundancy
* **Model utilization:** Chunks should fit within context window budgets

There's no universal "best" configuration—optimal parameters depend on content type, retrieval patterns, and downstream tasks.

=== Usage ===

Configure chunk parameters when:
* Building RAG pipelines
* Optimizing search quality
* Managing storage costs
* Targeting specific model context windows

Key tradeoffs:
* Smaller chunks: Better precision, worse context
* Larger chunks: Better context, worse precision
* More overlap: Better continuity, more storage
* Less overlap: Less redundancy, potential context loss at boundaries

== Theoretical Basis ==

Chunk Size Configuration involves **parameter tuning** for optimal information retrieval.

'''1. The Chunk Size Spectrum'''

<syntaxhighlight lang="text">
Chunk Size Spectrum:

Small (100-300)          Medium (500-1000)         Large (1500-3000)
├─────────────────────────┼─────────────────────────┼─────────────────────
│ Precise retrieval       │ Balanced                │ Broad context
│ Specific facts          │ General RAG             │ Summarization
│ Q&A systems             │ Chat + retrieval        │ Long-form analysis
│ Many chunks retrieved   │ Moderate retrieval      │ Few chunks needed
│ High embedding costs    │ Balanced costs          │ Lower embedding costs
</syntaxhighlight>

'''2. Overlap Strategy'''

<syntaxhighlight lang="python">
# Pseudo-code for overlap reasoning
def determine_overlap(chunk_size: int, content_type: str) -> int:
    """Determine optimal overlap based on content type."""

    # Percentage-based baseline
    base_overlap_pct = 0.15  # 15% default

    if content_type == "technical_docs":
        # Higher overlap for technical content (equations, code spans)
        base_overlap_pct = 0.25
    elif content_type == "prose":
        # Lower overlap for flowing text
        base_overlap_pct = 0.10
    elif content_type == "code":
        # Moderate overlap for code (function context)
        base_overlap_pct = 0.20

    return int(chunk_size * base_overlap_pct)
</syntaxhighlight>

'''3. Context Window Budgeting'''

<syntaxhighlight lang="python">
# Pseudo-code for budget allocation
def budget_chunk_size(
    model_context: int,
    system_prompt_tokens: int,
    user_query_tokens: int,
    num_chunks_to_retrieve: int,
    generation_headroom: int,
) -> int:
    """Calculate chunk size from context budget."""

    available = model_context - system_prompt_tokens - user_query_tokens - generation_headroom
    chunk_budget = available / num_chunks_to_retrieve

    return int(chunk_budget)

# Example: GPT-4 (8192 context)
# System prompt: 500 tokens
# Query: 100 tokens
# Generation: 500 tokens
# Retrieve: 5 chunks
# Budget: (8192 - 500 - 100 - 500) / 5 = 1418 tokens per chunk
</syntaxhighlight>

'''4. The Overlap-Continuity Tradeoff'''

<syntaxhighlight lang="text">
Overlap Impact:

No Overlap (0):
  Chunk 1: [...........]
  Chunk 2:              [...........]
  Risk: Context lost at boundaries

Light Overlap (10%):
  Chunk 1: [...........]
  Chunk 2:          [...........]
  Balance: Some continuity, minimal redundancy

Heavy Overlap (30%):
  Chunk 1: [...........]
  Chunk 2:      [...........]
  Thorough: Good continuity, significant redundancy
</syntaxhighlight>

'''5. Content-Aware Sizing'''

<syntaxhighlight lang="python">
# Pseudo-code for content-aware configuration
CONTENT_TYPE_CONFIGS = {
    "api_documentation": {
        "chunk_size": 400,
        "chunk_overlap": 50,
        "reason": "API docs are dense; small chunks for precise function lookup"
    },
    "legal_contracts": {
        "chunk_size": 1500,
        "chunk_overlap": 300,
        "reason": "Legal clauses need full context; high overlap for section continuity"
    },
    "code_repository": {
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "reason": "Functions/classes as units; moderate overlap for imports/context"
    },
    "chat_history": {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "reason": "Conversational turns; overlap maintains dialogue continuity"
    },
    "research_papers": {
        "chunk_size": 800,
        "chunk_overlap": 150,
        "reason": "Balanced for section retrieval; overlap for cross-reference"
    },
}
</syntaxhighlight>

'''6. Empirical Tuning Approach'''

<syntaxhighlight lang="python">
# Pseudo-code for empirical tuning
def tune_chunk_parameters(
    documents: list[Document],
    test_queries: list[str],
    ground_truth: dict[str, list[int]],  # query -> relevant doc indices
    model: str,
) -> dict:
    """Grid search for optimal chunk parameters."""

    best_config = None
    best_score = 0

    for chunk_size in [256, 512, 768, 1024, 1536]:
        for overlap_pct in [0, 0.1, 0.2, 0.3]:
            overlap = int(chunk_size * overlap_pct)

            # Create splitter and split documents
            splitter = create_splitter(chunk_size, overlap)
            chunks = split_all(documents, splitter)

            # Build index and evaluate retrieval
            index = build_index(chunks, model)
            score = evaluate_retrieval(index, test_queries, ground_truth)

            if score > best_score:
                best_score = score
                best_config = {"chunk_size": chunk_size, "chunk_overlap": overlap}

    return best_config
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:langchain-ai_langchain_TextSplitter_init]]

=== Used By Workflows ===
* Text_Splitting_Workflow (Step 3)
