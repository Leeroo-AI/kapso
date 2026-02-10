"""
Test Index & Search — wikis_index_test

Indexes data/wikis_index_test (Unsloth wiki pages) into kg_graph_search,
then runs a few search queries to verify indexing and retrieval work end-to-end.

Requires: Weaviate, Neo4j, OpenAI API key
Usage:    python tests/test_index_search_wikis.py
"""

from pathlib import Path
from dotenv import load_dotenv

from kapso.knowledge_base.search import (
    KnowledgeSearchFactory,
    KGIndexInput,
    KGSearchFilters,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WIKI_DIR = Path("data/wikis_index_test")

# Use a dedicated Weaviate collection so we don't clobber the main index
TEST_COLLECTION = "KGWikiPagesTest"


def main():
    print("=" * 60)
    print("Index & Search Test  —  wikis_index_test")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Clean slate — delete test Weaviate collection if it exists
    #    This prevents duplicate objects from accumulating across runs.
    # ------------------------------------------------------------------
    try:
        import weaviate
        wv = weaviate.connect_to_local()
        wv.collections.delete(TEST_COLLECTION)
        wv.close()
        print(f"Cleaned up existing collection '{TEST_COLLECTION}'")
    except Exception:
        pass  # collection didn't exist yet — that's fine

    # ------------------------------------------------------------------
    # 2. Create search backend with a test-specific collection
    # ------------------------------------------------------------------
    search = KnowledgeSearchFactory.create(
        "kg_graph_search",
        params={"weaviate_collection": TEST_COLLECTION},
    )

    # ------------------------------------------------------------------
    # 3. Index the wiki pages
    # ------------------------------------------------------------------
    assert WIKI_DIR.exists(), f"Test wiki dir not found: {WIKI_DIR}"
    print(f"\nIndexing wiki pages from {WIKI_DIR} …")

    search.index(KGIndexInput(wiki_dir=WIKI_DIR))

    page_count = search.get_indexed_count()
    print(f"Indexed {page_count} pages.")
    assert page_count > 0, "Expected at least one page to be indexed"

    # ------------------------------------------------------------------
    # 4. Run search queries
    # ------------------------------------------------------------------
    queries = [
        # Semantic match — LoRA / fine-tuning topic
        ("How does LoRA adapter injection work for fine-tuning?", None),
        # Type-filtered — only Implementations
        ("MoE expert routing kernel", ["Implementation"]),
        # Broad query — should return something from the corpus
        ("How to export a model to GGUF format?", None),
    ]

    for query, page_types in queries:
        print(f"\n{'—' * 60}")
        print(f"Query: {query}")
        if page_types:
            print(f"Filter: page_types={page_types}")

        filters = KGSearchFilters(
            top_k=5,
            page_types=page_types,
        )

        result = search.search(query, filters)

        print(f"Results: {result.total_found} found")
        assert len(result.results) > 0, f"Expected results for: {query}"

        for item in result.results:
            print(f"  [{item.score:.3f}] {item.id} ({item.page_type})")

            # Show Neo4j graph connections (incoming & outgoing)
            connected = item.metadata.get("connected_pages", [])
            outgoing = [c for c in connected if c["direction"] == "outgoing"]
            incoming = [c for c in connected if c["direction"] == "incoming"]
            if outgoing:
                print(f"           → outgoing: {', '.join(c['id'] for c in outgoing)}")
            if incoming:
                print(f"           ← incoming: {', '.join(c['id'] for c in incoming)}")

    # ------------------------------------------------------------------
    # 5. Test get_page on first result
    # ------------------------------------------------------------------
    top = result.top_result
    if top:
        print(f"\n{'—' * 60}")
        print(f"get_page({top.id!r})")
        page = search.get_page(top.id)
        assert page is not None, f"get_page returned None for {top.id}"
        print(f"  id:          {page.id}")
        print(f"  type:        {page.page_type}")
        print(f"  overview:    {page.overview[:120]}…" if len(page.overview) > 120 else f"  overview:    {page.overview}")
        print(f"  description: {page.description[:120]}…" if len(page.description) > 120 else f"  description: {page.description}")
        print(f"  domains:     {page.domains}")
        print(f"  outgoing:    {page.outgoing_links}")

    # ------------------------------------------------------------------
    # 6. Cleanup — remove test Weaviate collection
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    try:
        import weaviate
        wv = weaviate.connect_to_local()
        wv.collections.delete(TEST_COLLECTION)
        wv.close()
        print(f"Deleted Weaviate collection '{TEST_COLLECTION}'")
    except Exception as e:
        print(f"Warning: could not delete collection: {e}")

    search.close()
    print("Test complete!")


if __name__ == "__main__":
    main()
