# Test script for hierarchical knowledge merger
#
# This script tests the KnowledgeMerger by adding pages from staging directories
# one at a time to simulate incremental merging.
#
# Usage:
#     python tests/test_knowledge_merger_incremental.py
#
# Test Data:
#     data/wikis_test_research/_staging/ contains 3 test directories:
#     1. idea_Best_practices_for_LLM_fine_tu_32be8ec18368/
#     2. implementation_How_to_implement_RAG_with_Lang_4284e5484393/
#     3. researchreport_Comparison_of_LoRA_QLoRA_and_f_6f518d536006/
#
# Each run adds one directory's pages to the wiki_dir.

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge.learners.merger import KnowledgeMerger, MergeResult
from src.knowledge.search.kg_graph_search import parse_wiki_directory


# =============================================================================
# Configuration
# =============================================================================

# Source directories with test pages
STAGING_DIR = project_root / "data" / "wikis_test_research" / "_staging"

# Target wiki directory for merge tests
WIKI_DIR = project_root / "data" / "wiki_merge_test"

# Test directories in order of complexity
TEST_DIRS = [
    "idea_Best_practices_for_LLM_fine_tu_32be8ec18368",
    "implementation_How_to_implement_RAG_with_Lang_4284e5484393",
    "researchreport_Comparison_of_LoRA_QLoRA_and_f_6f518d536006",
]


# =============================================================================
# Test Functions
# =============================================================================

def load_pages_from_staging(staging_subdir: str):
    """Load WikiPage objects from a staging subdirectory."""
    source_dir = STAGING_DIR / staging_subdir
    if not source_dir.exists():
        raise FileNotFoundError(f"Staging directory not found: {source_dir}")
    
    pages = parse_wiki_directory(source_dir)
    return pages


def run_merge_test(staging_subdir: str, merger: KnowledgeMerger) -> MergeResult:
    """Run merge test for a single staging directory."""
    print(f"\n{'='*60}")
    print(f"Testing: {staging_subdir}")
    print(f"{'='*60}")
    
    # Load pages
    pages = load_pages_from_staging(staging_subdir)
    print(f"Loaded {len(pages)} pages:")
    for page in pages:
        print(f"  - {page.id} ({page.page_type})")
    
    # Run merge
    print(f"\nRunning merge to: {WIKI_DIR}")
    result = merger.merge(pages, wiki_dir=WIKI_DIR)
    
    # Print results
    print(f"\nResult: {result}")
    print(f"  Created: {result.created}")
    print(f"  Edited: {result.edited}")
    print(f"  Failed: {result.failed}")
    if result.errors:
        print(f"  Errors: {result.errors}")
    if result.plan_path:
        print(f"  Plan: {result.plan_path}")
    
    return result


def main():
    """Main test runner."""
    print("="*60)
    print("Knowledge Merger Incremental Test")
    print("="*60)
    print(f"Staging dir: {STAGING_DIR}")
    print(f"Wiki dir: {WIKI_DIR}")
    
    # Ensure wiki_dir exists
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create merger (no kg_index_path = create all as new)
    # To test with merge mode, provide kg_index_path pointing to a .index file
    merger = KnowledgeMerger()
    
    # Track overall results
    all_results = []
    
    # Process each test directory
    for i, test_dir in enumerate(TEST_DIRS, 1):
        print(f"\n\n{'#'*60}")
        print(f"# Round {i}/{len(TEST_DIRS)}")
        print(f"{'#'*60}")
        
        try:
            result = run_merge_test(test_dir, merger)
            all_results.append((test_dir, result))
        except Exception as e:
            print(f"\nERROR: {e}")
            all_results.append((test_dir, None))
    
    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    total_created = 0
    total_edited = 0
    total_failed = 0
    
    for test_dir, result in all_results:
        if result:
            status = "SUCCESS" if result.success else "PARTIAL"
            print(f"  {test_dir}: {status}")
            print(f"    Created: {len(result.created)}, Edited: {len(result.edited)}, Failed: {len(result.failed)}")
            total_created += len(result.created)
            total_edited += len(result.edited)
            total_failed += len(result.failed)
        else:
            print(f"  {test_dir}: FAILED")
    
    print(f"\nTotals:")
    print(f"  Created: {total_created}")
    print(f"  Edited: {total_edited}")
    print(f"  Failed: {total_failed}")
    
    print(f"\nWiki directory: {WIKI_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
