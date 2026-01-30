#!/usr/bin/env python3
"""
End-to-end test for agentic research ingestors.

Tests the full pipeline:
1. Research → Idea/Implementation/ResearchReport
2. Ingest → WikiPage objects with proper structure
3. Validate → Pages follow wiki structure definitions

Usage:
    conda activate praxium_conda
    cd /home/ubuntu/kapso
    python tests/test_research_ingestors.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.knowledge_base_base.types import Source
from src.researcher import Researcher
from src.knowledge_base_base.learners.ingestors import IdeaIngestor, ImplementationIngestor, ResearchReportIngestor
from src.knowledge_base_base.search.base import WikiPage


def test_idea_ingestor_e2e():
    """Test: Research idea mode → IdeaIngestor → WikiPages."""
    print("\n" + "=" * 60)
    print("Test: Idea Mode → Ingestor → WikiPages")
    print("=" * 60 + "\n")
    
    # Step 1: Research
    researcher = Researcher()
    query = "Best practices for LLM fine-tuning with LoRA"
    
    print(f"Researching: {query}")
    ideas = researcher.research(
        query=query,
        mode="idea",
        top_k=2,  # Small for testing
        depth="light",
    )
    
    print(f"Research returned {len(ideas)} ideas")
    assert len(ideas) > 0, "Expected at least 1 idea"
    
    # Step 2: Ingest the first idea
    ingestor = IdeaIngestor(params={
        "wiki_dir": "data/wikis_test_research",
        "cleanup_staging": False,  # Keep for inspection
    })
    
    idea = ideas[0]
    print(f"\nIngesting idea: {idea.query[:50]}...")
    print(f"  Source: {idea.source}")
    print(f"  Content length: {len(idea.content)} chars")
    
    pages = ingestor.ingest(idea)
    
    print(f"\nCreated {len(pages)} pages:")
    assert len(pages) > 0, "Expected at least 1 page"
    
    for page in pages:
        assert isinstance(page, WikiPage), f"Expected WikiPage, got {type(page)}"
        print(f"  - {page.page_type}: {page.page_title}")
        
        # Validate page has required sections
        assert page.content, "Page content should not be empty"
        assert "== Overview ==" in page.content, "Page should have Overview section"
    
    print(f"\nStaging directory: {ingestor.get_staging_dir()}")
    print("\n✅ Idea ingestor test passed!")
    return pages


def test_implementation_ingestor_e2e():
    """Test: Research implementation mode → ImplementationIngestor → WikiPages."""
    print("\n" + "=" * 60)
    print("Test: Implementation Mode → Ingestor → WikiPages")
    print("=" * 60 + "\n")
    
    # Step 1: Research
    researcher = Researcher()
    query = "How to implement RAG with LangChain"
    
    print(f"Researching: {query}")
    impls = researcher.research(
        query=query,
        mode="implementation",
        top_k=2,
        depth="light",
    )
    
    print(f"Research returned {len(impls)} implementations")
    assert len(impls) > 0, "Expected at least 1 implementation"
    
    # Step 2: Ingest the first implementation
    ingestor = ImplementationIngestor(params={
        "wiki_dir": "data/wikis_test_research",
        "cleanup_staging": False,
    })
    
    impl = impls[0]
    print(f"\nIngesting implementation: {impl.query[:50]}...")
    print(f"  Source: {impl.source}")
    print(f"  Content length: {len(impl.content)} chars")
    
    pages = ingestor.ingest(impl)
    
    print(f"\nCreated {len(pages)} pages:")
    assert len(pages) > 0, "Expected at least 1 page"
    
    for page in pages:
        assert isinstance(page, WikiPage), f"Expected WikiPage, got {type(page)}"
        print(f"  - {page.page_type}: {page.page_title}")
        
        # Validate page has required sections
        assert page.content, "Page content should not be empty"
        assert "== Overview ==" in page.content, "Page should have Overview section"
    
    print(f"\nStaging directory: {ingestor.get_staging_dir()}")
    print("\n✅ Implementation ingestor test passed!")
    return pages


def test_research_report_ingestor_e2e():
    """Test: Research study mode → ResearchReportIngestor → WikiPages."""
    print("\n" + "=" * 60)
    print("Test: Study Mode → Ingestor → WikiPages")
    print("=" * 60 + "\n")
    
    # Step 1: Research
    researcher = Researcher()
    query = "Comparison of LoRA, QLoRA, and full fine-tuning for LLMs"
    
    print(f"Researching: {query}")
    report = researcher.research(
        query=query,
        mode="study",
        depth="light",
    )
    
    print(f"Research returned report with {len(report.content)} chars")
    assert isinstance(report, Source.ResearchReport), f"Expected Source.ResearchReport, got {type(report)}"
    
    # Step 2: Ingest the report
    ingestor = ResearchReportIngestor(params={
        "wiki_dir": "data/wikis_test_research",
        "cleanup_staging": False,
        "timeout": 900,  # Longer timeout for comprehensive reports
    })
    
    print(f"\nIngesting research report: {report.query[:50]}...")
    print(f"  Content length: {len(report.content)} chars")
    
    pages = ingestor.ingest(report)
    
    print(f"\nCreated {len(pages)} pages:")
    assert len(pages) > 0, "Expected at least 1 page"
    
    for page in pages:
        assert isinstance(page, WikiPage), f"Expected WikiPage, got {type(page)}"
        print(f"  - {page.page_type}: {page.page_title}")
        
        # Validate page has required sections
        assert page.content, "Page content should not be empty"
        assert "== Overview ==" in page.content, "Page should have Overview section"
    
    print(f"\nStaging directory: {ingestor.get_staging_dir()}")
    print("\n✅ Research report ingestor test passed!")
    return pages


def test_full_pipeline_with_learn():
    """Test: Research → Ingest → Learn (full pipeline)."""
    print("\n" + "=" * 60)
    print("Test: Full Pipeline (Research → Ingest → Learn)")
    print("=" * 60 + "\n")
    
    from src.knowledge_base.learners import KnowledgePipeline
    
    # Step 1: Research in multiple modes
    researcher = Researcher()
    
    print("Researching ideas...")
    ideas = researcher.research(
        query="LoRA fine-tuning best practices",
        mode="idea",
        top_k=1,
        depth="light",
    )
    
    print("Researching implementations...")
    impls = researcher.research(
        query="How to use Unsloth for fine-tuning",
        mode="implementation",
        top_k=1,
        depth="light",
    )
    
    print(f"\nResearch returned {len(ideas)} ideas, {len(impls)} implementations")
    
    # Step 2: Run through pipeline (ingest only, skip merge)
    pipeline = KnowledgePipeline(wiki_dir="data/wikis_test_research")
    
    # Ingest ideas
    for idea in ideas:
        print(f"\nIngesting idea via pipeline...")
        result = pipeline.run(idea, skip_merge=True)
        print(f"  Pages extracted: {result.total_pages_extracted}")
        print(f"  Errors: {result.errors}")
        assert result.total_pages_extracted > 0, "Expected pages from idea"
    
    # Ingest implementations
    for impl in impls:
        print(f"\nIngesting implementation via pipeline...")
        result = pipeline.run(impl, skip_merge=True)
        print(f"  Pages extracted: {result.total_pages_extracted}")
        print(f"  Errors: {result.errors}")
        assert result.total_pages_extracted > 0, "Expected pages from implementation"
    
    print("\n✅ Full pipeline test passed!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Research Ingestor End-to-End Tests")
    print("=" * 60)
    print("\nEnvironment: praxium_conda")
    
    # Run tests one at a time to avoid rate limits
    # Uncomment the tests you want to run:
    
    # test_idea_ingestor_e2e()
    # test_implementation_ingestor_e2e()
    test_research_report_ingestor_e2e()
    # test_full_pipeline_with_learn()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
