# Principle: RAG_Baseline

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Retrieval-Augmented Generation|https://arxiv.org/abs/2005.11401]]
* [[source::Doc|FAISS Documentation|https://faiss.ai/]]
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::RAG]], [[domain::Information_Retrieval]], [[domain::Baseline]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:30 GMT]]
|}

== Overview ==

Retrieval-Augmented Generation baseline system that indexes visited webpage content and retrieves relevant context for improved question answering.

=== Description ===

RAG Baseline implements a standard retrieval-augmented generation pipeline to serve as a comparison point for agentic approaches. The system:

1. **Chunks documents** - Splits webpage content into overlapping segments
2. **Embeds chunks** - Generates dense vector representations using sentence transformers
3. **Indexes vectors** - Stores embeddings in FAISS for efficient similarity search
4. **Retrieves context** - Finds most relevant chunks for a query
5. **Augments prompts** - Prepends retrieved context to LLM input

This provides a non-agentic baseline: instead of having an agent actively search and reason, the RAG system relies on pre-indexed content.

=== Usage ===

Use RAG Baseline when:
- Comparing agentic vs retrieval approaches
- Building systems with pre-indexed content
- Need fast, non-iterative responses
- Establishing baseline performance metrics

== Theoretical Basis ==

RAG combines retrieval and generation:

<math>
P(y|x) = \sum_{d \in D} P(d|x) \cdot P(y|x, d)
</math>

Where x is query, y is answer, d is retrieved document, and D is corpus.

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# RAG Baseline Pattern
class RAGBaseline:
    def __init__(self, embedding_model, chunk_size=512):
        self.embedder = SentenceTransformer(embedding_model)
        self.index = faiss.IndexFlatIP(self.embedder.dimension)
        self.chunks = []
        self.chunk_size = chunk_size

    def index_document(self, content, metadata):
        """Add document to retrieval index."""
        chunks = self.chunk_text(content, self.chunk_size)
        embeddings = self.embedder.encode(chunks)
        self.index.add(embeddings)
        self.chunks.extend([(c, metadata) for c in chunks])

    def retrieve(self, query, top_k=5):
        """Retrieve most relevant chunks."""
        query_embedding = self.embedder.encode([query])
        scores, indices = self.index.search(query_embedding, top_k)
        return [(self.chunks[i], scores[0][j])
                for j, i in enumerate(indices[0])]

    def generate(self, query, context):
        """Generate answer with retrieved context."""
        prompt = f"Context: {context}\n\nQuestion: {query}"
        return self.llm.generate(prompt)
</syntaxhighlight>

Key components:
- **Embedding model**: Converts text to dense vectors
- **Vector index**: FAISS for efficient similarity search
- **Chunking strategy**: Overlapping windows preserve context
- **Retrieval threshold**: Filter low-confidence matches

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_RAG_System]]

=== Related Principles ===
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Webpage_Visitation]]
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Context_Management]]
