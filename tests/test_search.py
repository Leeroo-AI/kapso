"""
Test Knowledge Search - KG Graph Search

Tests indexing and search with/without LLM reranker.
Requires: Weaviate, Neo4j, OpenAI API key
"""

from pathlib import Path
from src.knowledge.search import KnowledgeSearchFactory, KGIndexInput, KGSearchFilters, PageType
from dotenv import load_dotenv

load_dotenv()


def main():
    """Run search tests."""
    
    # =========================================================================
    # Setup: Create search backend
    # =========================================================================
    print("=" * 60)
    print("KG Graph Search Test")
    print("=" * 60)
    
    # Create search backend with reranker enabled by default
    search = KnowledgeSearchFactory.create("kg_graph_search")
    
    # =========================================================================
    # Step 1: Index wiki pages
    # =========================================================================
    wiki_dir = Path("data/wikis")
    persist_path = Path("data/indexes/wikis.json")
    
    if wiki_dir.exists():
        print(f"\nIndexing wiki pages from {wiki_dir}...")
        search.index(KGIndexInput(
            wiki_dir=wiki_dir,
            persist_path=persist_path,
        ))
        print("Indexing complete!")
    else:
        print(f"\nWarning: {wiki_dir} not found. Skipping indexing.")
        print("Search will use existing indexed data if available.")
    
    # =========================================================================
    # Step 2: Test WITHOUT LLM Reranker (semantic search only)
    # =========================================================================
    query = "How to fine-tune LLM?"
    
    print("\n" + "=" * 60)
    print("TEST 1: WITHOUT LLM Reranker (semantic search only)")
    print("=" * 60)
    
    print(f"\nQuery: {query}")
    result = search.search(
        query, 
        filters=KGSearchFilters(top_k=5),
        use_llm_reranker=False,  # Explicitly disable reranker
    )
    
    print(f"Results ({result.total_found} found, reranked={result.search_metadata.get('reranked')}):")
    for item in result:
        print(f"  - {item.page_title} ({item.page_type}) score={item.score:.3f}")
    
    # =========================================================================
    # Step 3: Test WITH LLM Reranker
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 2: WITH LLM Reranker (gpt-4.1-mini)")
    print("=" * 60)
    
    print(f"\nQuery: {query}")
    result = search.search(
        query, 
        filters=KGSearchFilters(top_k=5),
        use_llm_reranker=True,  # Explicitly enable reranker
    )
    
    print(f"Results ({result.total_found} found, reranked={result.search_metadata.get('reranked')}):")
    for item in result:
        print(f"  - {item.page_title} ({item.page_type}) score={item.score:.3f}")
    
    # =========================================================================
    # Step 4: Test with filters + reranker + graph enrichment
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 3: WITH Filters + Reranker + Graph Enrichment")
    print("=" * 60)
    
    query2 = "LoRA best practices"
    print(f"\nQuery: {query2}")
    print("Filters: page_types=[Heuristic, Workflow], domains=[LLMs]")
    
    result = search.search(
        query=query2, 
        filters=KGSearchFilters(
            top_k=5, 
            page_types=[PageType.HEURISTIC, PageType.WORKFLOW], 
            domains=["LLMs"],
        ),
        use_llm_reranker=True,
    )
    
    print(f"\nResults ({result.total_found} found, reranked={result.search_metadata.get('reranked')}):")
    for item in result:
        print(f"  - {item.page_title} ({item.page_type}) score={item.score:.3f}")
        connected = item.metadata.get("connected_pages", [])
        if connected:
            print(f"    └─ Connected: {len(connected)} pages")
            for conn in connected[:3]:  # Show first 3 connections
                print(f"       - {conn.get('title', conn.get('id'))} ({conn.get('type')})")
    
    # =========================================================================
    # Step 5: Test get_page (direct lookup)
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 4: Direct Page Lookup (get_page)")
    print("=" * 60)
    
    # Try to get a page from the top result
    if result.top_result:
        page_title = result.top_result.page_title
        print(f"\nLooking up page: {page_title}")
        page = search.get_page(page_title)
        if page:
            print(f"  Found: {page.page_title} ({page.page_type})")
            print(f"  Domains: {page.domains}")
            print(f"  Overview: {page.overview[:200]}..." if len(page.overview) > 200 else f"  Overview: {page.overview}")
        else:
            print(f"  Page not found")
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    print("\n" + "=" * 60)
    search.close()
    print("Test complete!")


if __name__ == "__main__":
    main()
