"""
Test WorkflowRepoSearch (C3)

Tests workflow repository search over indexed workflow pages.
Requires: Weaviate, Neo4j, OpenAI API key
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def main():
    """Test WorkflowRepoSearch."""
    from src.knowledge_base.search import KnowledgeSearchFactory
    from src.knowledge_base.search.workflow_search import WorkflowRepoSearch, extract_github_url
    from src.knowledge_base.search.base import KGIndexInput
    
    print("=" * 60)
    print("WorkflowRepoSearch (C3) Test")
    print("=" * 60)
    
    # =========================================================================
    # Test 0: GitHub URL extraction
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 0: GitHub URL Extraction")
    print("=" * 60)
    
    # Test MediaWiki syntax
    content1 = "* [[source::Repo|keras|https://github.com/keras-team/keras]]"
    url1 = extract_github_url(content1)
    print(f"  MediaWiki syntax: {url1}")
    assert url1 == "https://github.com/keras-team/keras", f"Expected keras URL, got {url1}"
    
    # Test Github URL section
    content2 = "== Github URL ==\nhttps://github.com/pytorch/pytorch"
    url2 = extract_github_url(content2)
    print(f"  Section syntax: {url2}")
    assert url2 == "https://github.com/pytorch/pytorch", f"Expected pytorch URL, got {url2}"
    
    # Test raw URL
    content3 = "Check out https://github.com/scikit-learn/scikit-learn for ML"
    url3 = extract_github_url(content3)
    print(f"  Raw URL: {url3}")
    assert "scikit-learn" in url3, f"Expected scikit-learn URL, got {url3}"
    
    print("  ✓ All extraction tests passed")
    
    # =========================================================================
    # Setup: Index workflow repo pages
    # =========================================================================
    print("\n" + "=" * 60)
    print("Setup: Index Workflow Repo Pages")
    print("=" * 60)
    
    # KGGraphSearch expects type subdirectories (workflows/, principles/, etc.)
    wiki_dir = Path("data/workflow_repos")
    persist_path = Path("data/indexes/workflow_repos.json")
    
    if not (wiki_dir / "workflows").exists() or not list((wiki_dir / "workflows").glob("*.md")):
        print(f"\n⚠️  No workflow pages found in {wiki_dir}")
        print("Run: python scripts/generate_workflow_repo_pages.py --top-k 10")
        return
    
    print(f"\nIndexing workflow pages from {wiki_dir}...")
    kg_search = KnowledgeSearchFactory.create("kg_graph_search")
    kg_search.clear()  # Clear old data
    kg_search.index(KGIndexInput(wiki_dir=wiki_dir, persist_path=persist_path))
    print("Indexing complete!")
    
    # =========================================================================
    # Test 1: Search for ML training
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 1: Search for ML training frameworks")
    print("=" * 60)
    
    search = WorkflowRepoSearch(kg_search=kg_search)
    
    query = "train deep learning models efficiently"
    print(f"\nQuery: {query}")
    
    result = search.search(query, top_k=3)
    print(f"Results ({len(result.items)} found):")
    for item in result.items:
        print(f"  - {item.title} (score={item.score:.3f})")
        print(f"    GitHub: {item.github_url}")
    
    # =========================================================================
    # Test 2: Search for data processing
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 2: Search for data processing")
    print("=" * 60)
    
    query = "manipulate tabular data dataframes"
    print(f"\nQuery: {query}")
    
    result = search.search(query, top_k=3)
    print(f"Results ({len(result.items)} found):")
    for item in result.items:
        print(f"  - {item.title} (score={item.score:.3f})")
        print(f"    GitHub: {item.github_url}")
    
    # =========================================================================
    # Test 3: find_starter_repo convenience method
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 3: find_starter_repo")
    print("=" * 60)
    
    problem = "deploy ML models for production inference"
    print(f"\nProblem: {problem}")
    
    github_url = search.find_starter_repo(problem)
    print(f"Starter repo URL: {github_url}")
    
    # =========================================================================
    # Test 4: Context string output
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 4: Context String Output")
    print("=" * 60)
    
    query = "graph neural networks"
    print(f"\nQuery: {query}")
    
    result = search.search(query, top_k=2)
    print("\n--- Context String (for LLM) ---")
    print(result.to_context_string()[:800] + "...")
    
    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
