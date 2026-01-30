"""
Test WikiIdeaSearch and WikiCodeSearch

Tests wiki-based retrieval tools using data/wikis_llm_finetuning.
Requires: Weaviate, Neo4j, OpenAI API key
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def main():
    """Test WikiIdeaSearch and WikiCodeSearch."""
    from kapso.knowledge_base.search import KnowledgeSearchFactory
    from kapso.knowledge_base.search.idea_impl_search import WikiIdeaSearch, WikiCodeSearch
    from kapso.knowledge_base.search.base import KGIndexInput
    
    print("=" * 60)
    print("WikiIdeaSearch & WikiCodeSearch Test")
    print("=" * 60)
    
    # =========================================================================
    # Setup: Index wiki data
    # =========================================================================
    wiki_dir = Path("data/wikis_llm_finetuning")
    persist_path = Path("data/indexes/wikis_llm_finetuning.json")
    
    print(f"\nIndexing wiki pages from {wiki_dir}...")
    # Use factory to get params from config (weaviate_collection, etc.)
    kg_search = KnowledgeSearchFactory.create("kg_graph_search")
    # Clear old data to avoid duplicates from previous runs
    kg_search.clear()
    kg_search.index(KGIndexInput(wiki_dir=wiki_dir, persist_path=persist_path))
    print("Indexing complete!")
    
    # =========================================================================
    # Test 1: WikiIdeaSearch (Principles & Heuristics)
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 1: WikiIdeaSearch (Principles & Heuristics)")
    print("=" * 60)
    
    idea_search = WikiIdeaSearch(kg_search=kg_search)
    
    query = "How to configure LoRA for efficient fine-tuning?"
    print(f"\nQuery: {query}")
    
    result = idea_search.search(query, top_k=5)
    print(f"Results ({len(result.items)} found):")
    for item in result.items:
        print(f"  - {item.title} ({item.item_type}) score={item.score:.3f}")
    
    # =========================================================================
    # Test 2: WikiCodeSearch (Implementations & Environments)
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 2: WikiCodeSearch (Implementations & Environments)")
    print("=" * 60)
    
    code_search = WikiCodeSearch(kg_search=kg_search)
    
    query = "FastLanguageModel from_pretrained API usage"
    print(f"\nQuery: {query}")
    
    result = code_search.search(query, top_k=5)
    print(f"Results ({len(result.items)} found):")
    for item in result.items:
        print(f"  - {item.title} ({item.item_type}) score={item.score:.3f}")
    
    # =========================================================================
    # Test 3: Context string output
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 3: Context String Output")
    print("=" * 60)
    
    query = "gradient checkpointing memory optimization"
    result = idea_search.search(query, top_k=2)
    
    print(f"\nQuery: {query}")
    print("\n--- Context String (for LLM) ---")
    print(result.to_context_string(max_items=2)[:1000] + "...")
    
    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
