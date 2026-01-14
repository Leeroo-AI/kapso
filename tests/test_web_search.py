"""
Test WebIdeaSearch and WebCodeSearch

Tests web-based retrieval tools using LLM web search.
Requires: OpenAI API key (for gpt-4o-search-preview)
"""

from dotenv import load_dotenv

load_dotenv()


def main():
    """Test WebIdeaSearch and WebCodeSearch."""
    from src.execution.web_search import WebIdeaSearch, WebCodeSearch
    
    print("=" * 60)
    print("WebIdeaSearch & WebCodeSearch Test")
    print("=" * 60)
    
    # =========================================================================
    # Test 1: WebIdeaSearch (Concepts & Best Practices)
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 1: WebIdeaSearch (Concepts & Best Practices)")
    print("=" * 60)
    
    idea_search = WebIdeaSearch()
    
    query = "best practices for LoRA rank selection in LLM fine-tuning"
    print(f"\nQuery: {query}")
    print("Searching web...")
    
    result = idea_search.search(query)
    
    if result.is_empty:
        print("No results found (check API key)")
    else:
        print(f"\n--- Web Idea Results ---")
        print(result.items[0].content[:800] + "...")
    
    # =========================================================================
    # Test 2: WebCodeSearch (Code Examples)
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 2: WebCodeSearch (Code Examples)")
    print("=" * 60)
    
    code_search = WebCodeSearch()
    
    query = "unsloth FastLanguageModel example"
    print(f"\nQuery: {query}")
    print("Searching web...")
    
    result = code_search.search(query, language="Python", framework="PyTorch")
    
    if result.is_empty:
        print("No results found (check API key)")
    else:
        print(f"\n--- Web Code Results ---")
        print(result.items[0].content[:800] + "...")
    
    # =========================================================================
    # Test 3: Context string output
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 3: Context String Output")
    print("=" * 60)
    
    query = "gradient checkpointing PyTorch"
    print(f"\nQuery: {query}")
    print("Searching web...")
    
    result = code_search.search(query, language="Python")
    
    if not result.is_empty:
        print("\n--- Context String (for LLM) ---")
        print(result.to_context_string()[:1000] + "...")
    
    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
