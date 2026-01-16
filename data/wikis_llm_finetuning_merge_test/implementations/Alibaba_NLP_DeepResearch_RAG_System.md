# Implementation: RAG_System

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::RAG]], [[domain::Information_Retrieval]], [[domain::Vector_Search]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:00 GMT]]
|}

== Overview ==
Retrieval-Augmented Generation system for WebWalker that indexes webpage content with embeddings and retrieves relevant context for question answering.

=== Description ===
The `rag_system.py` module implements a complete RAG pipeline for enhancing WebWalker agent responses with retrieved context. It provides:

- Document chunking and embedding generation using sentence transformers
- FAISS-based vector index for efficient similarity search
- Context retrieval with configurable top-k and similarity thresholds
- Integration with the WebWalker agent for augmented response generation
- Persistent index storage for incremental webpage indexing

The system chunks webpage content into overlapping segments, generates embeddings, and stores them in a FAISS index. During query time, it retrieves the most relevant chunks and prepends them to the agent context.

=== Usage ===
Use this module when building RAG-enhanced web research agents that need to recall previously visited webpage content. Useful for multi-turn conversations where earlier retrieved information is relevant.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebWalker/src/rag_system.py WebAgent/WebWalker/src/rag_system.py]
* '''Lines:''' 1-335

=== Signature ===
<syntaxhighlight lang="python">
class RAGSystem:
    """Retrieval-Augmented Generation system for web content."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        index_path: Optional[str] = None
    ):
        """
        Initialize RAG system.

        Args:
            embedding_model: HuggingFace model for embeddings
            chunk_size: Max tokens per chunk
            chunk_overlap: Overlap between chunks
            index_path: Path to persist/load FAISS index
        """
        ...

    def add_document(
        self,
        content: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add document content to the index.

        Args:
            content: Text content to index
            metadata: Optional metadata (url, title, etc.)
        """
        ...

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity score

        Returns:
            List of dicts with 'content', 'score', 'metadata'
        """
        ...

    def augment_context(
        self,
        query: str,
        base_context: str,
        top_k: int = 3
    ) -> str:
        """
        Augment context with retrieved information.

        Args:
            query: User query
            base_context: Original context
            top_k: Retrieved chunks to add

        Returns:
            Augmented context string
        """
        ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from WebAgent.WebWalker.src.rag_system import RAGSystem
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| embedding_model || str || No || HuggingFace model name (default MiniLM)
|-
| chunk_size || int || No || Tokens per chunk (default 512)
|-
| chunk_overlap || int || No || Overlap tokens (default 50)
|-
| index_path || str || No || Path to persist FAISS index
|-
| query || str || Yes || Search query for retrieval
|-
| top_k || int || No || Number of results (default 5)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| retrieved_chunks || List[Dict] || Retrieved content with scores
|-
| augmented_context || str || Context enriched with retrieved info
|-
| index || faiss.Index || FAISS vector index
|}

== Usage Examples ==

=== Basic RAG Pipeline ===
<syntaxhighlight lang="python">
from WebAgent.WebWalker.src.rag_system import RAGSystem

# Initialize RAG system
rag = RAGSystem(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=512,
    index_path="./webwalker_index"
)

# Index webpage content
webpage_content = """
The ACL 2025 conference will be held in Vienna, Austria.
The paper submission deadline is February 15, 2025.
The conference dates are July 27 - August 1, 2025.
"""
rag.add_document(
    content=webpage_content,
    metadata={"url": "https://2025.aclweb.org", "title": "ACL 2025"}
)

# Retrieve relevant chunks
results = rag.retrieve(
    query="When is the ACL 2025 deadline?",
    top_k=3
)
for r in results:
    print(f"Score: {r['score']:.3f}")
    print(f"Content: {r['content']}")
</syntaxhighlight>

=== Integrating with WebWalker Agent ===
<syntaxhighlight lang="python">
from WebAgent.WebWalker.src.rag_system import RAGSystem
from WebAgent.WebWalker.src.agent import WebWalker

# Create RAG-enhanced agent
rag = RAGSystem(index_path="./walker_index")

# After visiting pages, index them
for url, content in visited_pages.items():
    rag.add_document(content, metadata={"url": url})

# Augment agent context with retrieval
user_query = "What is the venue for ACL 2025?"
base_context = "You are a helpful assistant."
augmented = rag.augment_context(
    query=user_query,
    base_context=base_context,
    top_k=3
)

# Use augmented context in agent
messages = [
    {"role": "system", "content": augmented},
    {"role": "user", "content": user_query}
]
</syntaxhighlight>

=== Persistent Index ===
<syntaxhighlight lang="python">
# Save index after indexing
rag = RAGSystem(index_path="./my_index")
rag.add_document("Document 1 content...")
rag.add_document("Document 2 content...")
rag.save_index()  # Persist to disk

# Load existing index
rag_loaded = RAGSystem(index_path="./my_index")
rag_loaded.load_index()  # Resume from saved state
results = rag_loaded.retrieve("search query")
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_RAG_Baseline]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_Python_Dependencies]]
