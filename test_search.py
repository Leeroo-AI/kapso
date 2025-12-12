from src.knowledge.search import KnowledgeSearchFactory, KGSearchFilters, PageType
from dotenv import load_dotenv

load_dotenv()

# Create search (connections auto-close when script ends)
search = KnowledgeSearchFactory.create("kg_graph_search", enabled=True)

# Basic search
print("Query: How to fine-tune LLM?")
result = search.search("How to fine-tune LLM?")
for item in result:
    print(f"  - {item.page_title} ({item.page_type}) score={item.score:.2f}")

# Search with filters
print("\nQuery: LoRA best practices (filtered)")
result = search.search(
    query="LoRA best practices",
    filters=KGSearchFilters(
        top_k=5,
        min_score=0.5,
        page_types=[PageType.HEURISTIC, PageType.WORKFLOW],
        domains=["LLMs"],
    ),
)
for item in result:
    print(f"  - {item.page_title} ({item.page_type}) score={item.score:.2f}")
    connected = item.metadata.get("connected_pages", [])
    if connected:
        print(f"    Connected: {len(connected)} pages")