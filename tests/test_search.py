"""
Test Knowledge Search - KG Graph Search

Tests indexing and search with/without LLM reranker.
Requires: Weaviate, Neo4j, OpenAI API key
"""

from pathlib import Path
from src.knowledge.search import KnowledgeSearchFactory, KGIndexInput, KGEditInput, KGSearchFilters, PageType
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
    # Step 6: Test edit function (update page and indexes)
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 5: Edit Page Function")
    print("=" * 60)
    
    # We'll edit a heuristic page to test the edit functionality
    test_page_id = "Heuristic/LoRA_Rank_Selection"
    
    # First, get the original page to compare later
    print(f"\nEditing page: {test_page_id}")
    original_page = search.get_page("LoRA Rank Selection")
    if original_page:
        print(f"  Original domains: {original_page.domains}")
        print(f"  Original overview: {original_page.overview[:80]}...")
    
    # -------------------------------------------------------------------------
    # Test 5a: Edit domains (index-only, skip source files)
    # -------------------------------------------------------------------------
    print("\n  [5a] Edit domains (index only, skip source files)...")
    test_domains = ["LLMs", "PEFT", "LoRA", "Hyperparameters", "Test_Tag"]
    success = search.edit(KGEditInput(
        page_id=test_page_id,
        domains=test_domains,
        update_source_files=False,  # Only update indexes, not raw files
        update_persist_cache=False,  # Skip JSON cache too
        auto_timestamp=False,  # Don't change timestamp
    ))
    print(f"  Edit result: {'✓ Success' if success else '✗ Failed'}")
    
    # Verify the edit in Weaviate by searching
    if success:
        verify_result = search.search(
            "LoRA rank selection hyperparameters",
            filters=KGSearchFilters(top_k=1, page_types=[PageType.HEURISTIC]),
            use_llm_reranker=False,
        )
        if verify_result.top_result:
            print(f"  Verified domains in index: {verify_result.top_result.domains}")
    
    # -------------------------------------------------------------------------
    # Test 5b: Edit OVERVIEW (triggers re-embedding in Weaviate)
    # -------------------------------------------------------------------------
    print("\n  [5b] Edit OVERVIEW (triggers re-embedding)...")
    
    # Save original overview
    original_overview = original_page.overview if original_page else ""
    
    # Update overview with a test value
    test_overview = "TEST: Practical guidelines for selecting the LoRA rank parameter (r) to optimize the trade-off between model quality and training efficiency."
    
    overview_success = search.edit(KGEditInput(
        page_id=test_page_id,
        overview=test_overview,
        update_source_files=True,  # Update the .md file
        update_persist_cache=True,
        auto_timestamp=False,
    ))
    print(f"  Edit result: {'✓ Success' if overview_success else '✗ Failed'}")
    print(f"  requires_reembedding: {KGEditInput(page_id=test_page_id, overview='x').requires_reembedding}")
    
    # Verify overview was updated
    if overview_success:
        verify_page = search.get_page("LoRA Rank Selection")
        if verify_page:
            print(f"  Updated overview: {verify_page.overview[:80]}...")
    
    # Restore original overview
    if original_overview:
        restore_overview = search.edit(KGEditInput(
            page_id=test_page_id,
            overview=original_overview,
            update_source_files=True,
            update_persist_cache=True,
            auto_timestamp=False,
        ))
        print(f"  Restored overview: {'✓' if restore_overview else '✗'}")
    
    # -------------------------------------------------------------------------
    # Test 5c: Edit full CONTENT (replaces entire file)
    # -------------------------------------------------------------------------
    print("\n  [5c] Edit full CONTENT (file replacement)...")
    
    # Read the current file content to preserve it
    source_file = Path("data/wikis/heuristics/LoRA_Rank_Selection.md")
    original_content = source_file.read_text() if source_file.exists() else ""
    
    # Create test content with a marker we can verify
    test_content = original_content.replace(
        "== Overview ==",
        "== Overview ==\n<!-- TEST MARKER: Content edit test -->"
    )
    
    content_success = search.edit(KGEditInput(
        page_id=test_page_id,
        content=test_content,
        update_source_files=True,
        update_persist_cache=True,
        auto_timestamp=False,
    ))
    print(f"  Edit result: {'✓ Success' if content_success else '✗ Failed'}")
    
    # Verify content was updated
    if content_success and source_file.exists():
        updated_content = source_file.read_text()
        has_marker = "TEST MARKER" in updated_content
        print(f"  Test marker in file: {'✓ Found' if has_marker else '✗ Not found'}")
    
    # Restore original content
    if original_content:
        restore_content = search.edit(KGEditInput(
            page_id=test_page_id,
            content=original_content,
            update_source_files=True,
            update_persist_cache=True,
            auto_timestamp=False,
        ))
        print(f"  Restored content: {'✓' if restore_content else '✗'}")
        
        # Verify marker is gone
        if source_file.exists():
            restored_content = source_file.read_text()
            marker_removed = "TEST MARKER" not in restored_content
            print(f"  Test marker removed: {'✓' if marker_removed else '✗'}")
    
    # -------------------------------------------------------------------------
    # Test 5d: Restore original domains with full update
    # -------------------------------------------------------------------------
    print("\n  [5d] Restore original domains (full update)...")
    
    if original_page:
        restore_success = search.edit(KGEditInput(
            page_id=test_page_id,
            domains=original_page.domains,  # Restore original
            update_source_files=True,  # Update raw .md file
            update_persist_cache=True,  # Update JSON cache
            auto_timestamp=True,  # Update timestamp
        ))
        print(f"  Restore result: {'✓ Success' if restore_success else '✗ Failed'}")
        
        # Verify by getting page again
        restored_page = search.get_page("LoRA Rank Selection")
        if restored_page:
            print(f"  Restored domains: {restored_page.domains}")
    
    # -------------------------------------------------------------------------
    # Test 5e: Test KGEditInput defaults and properties
    # -------------------------------------------------------------------------
    print("\n  [5e] Test KGEditInput defaults and properties...")
    test_edit = KGEditInput(page_id="Workflow/QLoRA_Finetuning")
    print(f"  Default wiki_dir: {test_edit.wiki_dir}")
    print(f"  Default persist_path: {test_edit.persist_path}")
    print(f"  requires_reembedding (no overview): {test_edit.requires_reembedding}")
    print(f"  requires_edge_rebuild (no links): {test_edit.requires_edge_rebuild}")
    
    # Test with overview change
    test_edit_overview = KGEditInput(page_id="test", overview="new overview")
    print(f"  requires_reembedding (with overview): {test_edit_overview.requires_reembedding}")
    
    # Test with outgoing_links change
    test_edit_links = KGEditInput(page_id="test", outgoing_links=[{"edge_type": "step", "target_type": "Principle", "target_id": "Test"}])
    print(f"  requires_edge_rebuild (with links): {test_edit_links.requires_edge_rebuild}")
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    print("\n" + "=" * 60)
    search.close()
    print("Test complete!")


if __name__ == "__main__":
    main()
