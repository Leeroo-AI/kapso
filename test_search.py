from src.knowledge.search import KnowledgeSearchFactory, KGSearchFilters, PageType
from dotenv import load_dotenv

load_dotenv()

query = "How to fine-tune LLM?"

# =============================================================================
# Test WITHOUT LLM Reranker (semantic search only)
# =============================================================================
print("=" * 60)
print("WITHOUT LLM Reranker (semantic search only)")
print("=" * 60)

search_no_rerank = KnowledgeSearchFactory.create(
    "kg_graph_search", 
    params={"use_llm_reranker": False}
)

print(f"\nQuery: {query}")
result = search_no_rerank.search(query, filters=KGSearchFilters(top_k=10))
for item in result:
    print(f"  - {item.page_title} ({item.page_type}) score={item.score:.2f}")

# =============================================================================
# Test WITH LLM Reranker
# =============================================================================
print("\n" + "=" * 60)
print("WITH LLM Reranker (gpt-4.1-mini)")
print("=" * 60)

search_with_rerank = KnowledgeSearchFactory.create(
    "kg_graph_search",
    params={"use_llm_reranker": True}
)

print(f"\nQuery: {query}")
result = search_with_rerank.search(query, filters=KGSearchFilters(top_k=10))
for item in result:
    print(f"  - {item.page_title} ({item.page_type}) score={item.score:.2f}")

# =============================================================================
# Test with filters + reranker + graph enrichment
# =============================================================================
print("\n" + "=" * 60)
print("WITH Filters + Reranker + Graph Enrichment")
print("=" * 60)

search_full = KnowledgeSearchFactory.create("kg_graph_search")

print("\nQuery: LoRA best practices (filtered to Heuristic/Workflow, LLMs domain)")
result = search_full.search(
    query="LoRA best practices",
    filters=KGSearchFilters(
        top_k=5,
        page_types=[PageType.HEURISTIC, PageType.WORKFLOW],
        domains=["LLMs"],
    ),
)
for item in result:
    print(f"  - {item.page_title} ({item.page_type}) score={item.score:.2f}")
    connected = item.metadata.get("connected_pages", [])
    if connected:
        print(f"    Connected: {len(connected)} pages")