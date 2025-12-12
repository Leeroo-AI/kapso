#!/usr/bin/env python3
"""
Comprehensive test for Wiki MCP Server

Tests:
1. Knowledge search backend (index, search, get_page)
2. MCP server tools and resources
3. End-to-end search scenarios

Run from project root:
    python tests/test_wiki_mcp.py
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---\n")


# =============================================================================
# Test 1: Knowledge Search Backend
# =============================================================================

def test_search_backend():
    """Test the KGGraphSearch backend directly."""
    print_header("TEST 1: Knowledge Search Backend")
    
    from src.knowledge.search.factory import KnowledgeSearchFactory
    from src.knowledge.search.base import KGIndexInput, KGSearchFilters, PageType
    
    # Create search backend
    print_subheader("Creating KGGraphSearch backend")
    search = KnowledgeSearchFactory.create("kg_graph_search")
    print("✓ Backend created")
    
    # Index wiki pages
    print_subheader("Indexing wiki pages from data/wikis")
    wiki_dir = Path("data/wikis")
    
    try:
        search.index(KGIndexInput(
            wiki_dir=wiki_dir,
            persist_path=Path("data/indexes/wikis_test.json"),
        ))
        print("✓ Indexing completed")
    except Exception as e:
        print(f"✗ Indexing failed: {e}")
        return False
    
    # Test search queries
    print_subheader("Testing search queries")
    
    test_queries = [
        {
            "query": "How to fine-tune LLM with limited GPU memory?",
            "filters": KGSearchFilters(top_k=3, page_types=[PageType.WORKFLOW.value]),
            "expected_type": "Workflow",
        },
        {
            "query": "What is LoRA and how does it work?",
            "filters": KGSearchFilters(top_k=3, page_types=[PageType.PRINCIPLE.value]),
            "expected_type": "Principle",
        },
        {
            "query": "Best practices for learning rate selection",
            "filters": KGSearchFilters(top_k=3, page_types=[PageType.HEURISTIC.value]),
            "expected_type": "Heuristic",
        },
        {
            "query": "HuggingFace Trainer configuration",
            "filters": KGSearchFilters(top_k=3),
            "expected_type": None,  # Any type
        },
    ]
    
    all_passed = True
    for i, test in enumerate(test_queries, 1):
        print(f"\nQuery {i}: \"{test['query']}\"")
        if test['filters'].page_types:
            print(f"  Filter: page_types={test['filters'].page_types}")
        
        try:
            result = search.search(test["query"], test["filters"])
            print(f"  Found: {result.total_found} results")
            
            if result.is_empty:
                print("  ✗ No results found")
                all_passed = False
            else:
                for j, item in enumerate(result.results[:3], 1):
                    print(f"    [{j}] {item.page_title} ({item.page_type}) - Score: {item.score:.3f}")
                    
                    # Check connected pages
                    connected = item.metadata.get("connected_pages", [])
                    if connected:
                        print(f"        Connected: {len(connected)} pages")
                
                # Verify expected type
                if test["expected_type"] and result.top_result:
                    if result.top_result.page_type == test["expected_type"]:
                        print(f"  ✓ Top result is {test['expected_type']}")
                    else:
                        print(f"  ! Top result is {result.top_result.page_type}, expected {test['expected_type']}")
                else:
                    print("  ✓ Search completed")
                    
        except Exception as e:
            print(f"  ✗ Search failed: {e}")
            all_passed = False
    
    # Test search with/without reranker
    print_subheader("Testing use_llm_reranker override")
    
    query = "How to fine-tune LLM?"
    
    # Without reranker
    print(f"\nQuery: \"{query}\" (use_llm_reranker=False)")
    try:
        result = search.search(query, KGSearchFilters(top_k=3), use_llm_reranker=False)
        print(f"  Reranked: {result.search_metadata.get('reranked')}")
        print(f"  Found: {result.total_found} results")
        if result.top_result:
            print(f"  Top: {result.top_result.page_title} (score={result.top_result.score:.3f})")
        print("  ✓ Search without reranker completed")
    except Exception as e:
        print(f"  ✗ Search failed: {e}")
        all_passed = False
    
    # With reranker
    print(f"\nQuery: \"{query}\" (use_llm_reranker=True)")
    try:
        result = search.search(query, KGSearchFilters(top_k=3), use_llm_reranker=True)
        print(f"  Reranked: {result.search_metadata.get('reranked')}")
        print(f"  Found: {result.total_found} results")
        if result.top_result:
            print(f"  Top: {result.top_result.page_title} (score={result.top_result.score:.3f})")
        print("  ✓ Search with reranker completed")
    except Exception as e:
        print(f"  ✗ Search failed: {e}")
        all_passed = False
    
    # Test get_page
    print_subheader("Testing get_page")
    
    test_pages = [
        "QLoRA_Finetuning",
        "Low_Rank_Adaptation",
        "Learning_Rate_Tuning",
    ]
    
    for title in test_pages:
        try:
            page = search.get_page(title)
            if page:
                print(f"✓ Found page: {page.page_title} ({page.page_type})")
                print(f"  Overview: {page.overview[:100]}...")
            else:
                print(f"✗ Page not found: {title}")
                all_passed = False
        except Exception as e:
            print(f"✗ get_page failed for {title}: {e}")
            all_passed = False
    
    # Cleanup
    search.close()
    
    return all_passed


# =============================================================================
# Test 2: MCP Server Tools
# =============================================================================

async def test_mcp_tools():
    """Test MCP server tools asynchronously."""
    print_header("TEST 2: MCP Server Tools")
    
    from src.knowledge.wiki_mcps.mcp_server import (
        _handle_search,
        _handle_get_page,
        _handle_list_types,
        _handle_search_with_context,
        reset_search_backend,
    )
    
    all_passed = True
    
    # Test search_knowledge tool
    print_subheader("Testing search_knowledge tool")
    
    try:
        result = await _handle_search({
            "query": "How to do QLoRA fine-tuning?",
            "top_k": 3,
            "page_types": ["Workflow"],
        })
        
        if result and len(result) > 0:
            text = result[0].text
            print(f"✓ search_knowledge returned results")
            print(f"  Response preview:\n{text[:500]}...")
        else:
            print("✗ search_knowledge returned empty")
            all_passed = False
            
    except Exception as e:
        print(f"✗ search_knowledge failed: {e}")
        all_passed = False
    
    # Test get_wiki_page tool
    print_subheader("Testing get_wiki_page tool")
    
    try:
        result = await _handle_get_page({
            "page_title": "QLoRA_Finetuning",
        })
        
        if result and len(result) > 0:
            text = result[0].text
            if "not found" in text.lower():
                print(f"✗ Page not found")
                all_passed = False
            else:
                print(f"✓ get_wiki_page returned page content")
                print(f"  Response preview:\n{text[:500]}...")
        else:
            print("✗ get_wiki_page returned empty")
            all_passed = False
            
    except Exception as e:
        print(f"✗ get_wiki_page failed: {e}")
        all_passed = False
    
    # Test list_page_types tool
    print_subheader("Testing list_page_types tool")
    
    try:
        result = await _handle_list_types({})
        
        if result and len(result) > 0:
            text = result[0].text
            print(f"✓ list_page_types returned reference")
            # Check for expected types
            expected_types = ["Workflow", "Principle", "Implementation", "Environment", "Heuristic"]
            found = [t for t in expected_types if t in text]
            print(f"  Found types: {found}")
        else:
            print("✗ list_page_types returned empty")
            all_passed = False
            
    except Exception as e:
        print(f"✗ list_page_types failed: {e}")
        all_passed = False
    
    # Test search_with_context tool
    print_subheader("Testing search_with_context tool")
    
    try:
        result = await _handle_search_with_context({
            "query": "learning rate",
            "context": "I'm fine-tuning a 7B LLM with QLoRA and getting loss spikes",
            "top_k": 3,
        })
        
        if result and len(result) > 0:
            text = result[0].text
            print(f"✓ search_with_context returned results")
            print(f"  Response preview:\n{text[:400]}...")
        else:
            print("✗ search_with_context returned empty")
            all_passed = False
            
    except Exception as e:
        print(f"✗ search_with_context failed: {e}")
        all_passed = False
    
    return all_passed


# =============================================================================
# Test 3: MCP Server Resources
# =============================================================================

async def test_mcp_resources():
    """Test MCP server resources."""
    print_header("TEST 3: MCP Server Resources")
    
    from src.knowledge.wiki_mcps.mcp_server import (
        _get_overview_resource,
        _get_page_types_resource,
    )
    
    all_passed = True
    
    # Test overview resource
    print_subheader("Testing knowledge://overview resource")
    
    try:
        content = _get_overview_resource()
        if content and len(content) > 100:
            print(f"✓ Overview resource returned ({len(content)} chars)")
            # Check for key sections
            if "Page Types" in content and "How to Search" in content:
                print("  ✓ Contains expected sections")
            else:
                print("  ! Missing expected sections")
        else:
            print("✗ Overview resource too short or empty")
            all_passed = False
            
    except Exception as e:
        print(f"✗ Overview resource failed: {e}")
        all_passed = False
    
    # Test page-types resource
    print_subheader("Testing knowledge://page-types resource")
    
    try:
        content = _get_page_types_resource()
        if content and len(content) > 100:
            print(f"✓ Page types resource returned ({len(content)} chars)")
            # Check for all types
            types_found = sum(1 for t in ["Workflow", "Principle", "Implementation", "Environment", "Heuristic"] if t in content)
            print(f"  Found {types_found}/5 page types documented")
        else:
            print("✗ Page types resource too short or empty")
            all_passed = False
            
    except Exception as e:
        print(f"✗ Page types resource failed: {e}")
        all_passed = False
    
    return all_passed


# =============================================================================
# Test 4: End-to-End Search Scenarios
# =============================================================================

async def test_e2e_scenarios():
    """Test realistic end-to-end search scenarios."""
    print_header("TEST 4: End-to-End Search Scenarios")
    
    from src.knowledge.wiki_mcps.mcp_server import _handle_search, _handle_get_page
    
    all_passed = True
    
    scenarios = [
        {
            "name": "Debug memory issues during training",
            "query": "GPU out of memory during fine-tuning",
            "expected_keywords": ["memory", "gradient", "batch"],
        },
        {
            "name": "Understand DPO alignment",
            "query": "How does Direct Preference Optimization work?",
            "expected_keywords": ["preference", "DPO", "alignment"],
        },
        {
            "name": "Configure LoRA parameters",
            "query": "What LoRA rank should I use for 7B model?",
            "expected_keywords": ["rank", "LoRA", "parameter"],
        },
        {
            "name": "Export model for inference",
            "query": "How to export fine-tuned model to GGUF format?",
            "expected_keywords": ["GGUF", "export", "save"],
        },
    ]
    
    for scenario in scenarios:
        print_subheader(f"Scenario: {scenario['name']}")
        print(f"Query: \"{scenario['query']}\"")
        
        try:
            result = await _handle_search({
                "query": scenario["query"],
                "top_k": 3,
            })
            
            if result and len(result) > 0:
                text = result[0].text
                
                # Check for expected keywords
                keywords_found = [kw for kw in scenario["expected_keywords"] if kw.lower() in text.lower()]
                
                if keywords_found:
                    print(f"✓ Found relevant content (keywords: {keywords_found})")
                    
                    # Extract first result title
                    import re
                    title_match = re.search(r'\[1\]\s+(.+?)(?:\n|$)', text)
                    if title_match:
                        print(f"  Top result: {title_match.group(1)}")
                else:
                    print(f"! Results may not be relevant (missing: {scenario['expected_keywords']})")
            else:
                print("✗ No results")
                all_passed = False
                
        except Exception as e:
            print(f"✗ Failed: {e}")
            all_passed = False
    
    return all_passed


# =============================================================================
# Main Test Runner
# =============================================================================

def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  WIKI MCP SERVER - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Backend
    results["Backend"] = test_search_backend()
    
    # Test 2-4: Async tests
    async def run_async_tests():
        results["MCP Tools"] = await test_mcp_tools()
        results["MCP Resources"] = await test_mcp_resources()
        results["E2E Scenarios"] = await test_e2e_scenarios()
    
    asyncio.run(run_async_tests())
    
    # Summary
    print_header("TEST SUMMARY")
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed!")
        return 0
    else:
        print("\n  ⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

