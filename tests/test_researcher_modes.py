# Test Researcher in different modes.
#
# Usage:
#   conda activate praxium_conda
#   python -m tests.test_researcher_modes
#
# This script tests the Researcher class with the new API:
# - mode="idea" returns List[Idea]
# - mode="implementation" returns List[Implementation]
# - mode="study" returns ResearchReport

from typing import List

from dotenv import load_dotenv

from src.knowledge.researcher import (
    Researcher,
    Idea,
    Implementation,
    ResearchReport,
)

load_dotenv()


def test_idea_mode():
    """Test idea mode: returns List[Idea]."""
    print("\n" + "=" * 60)
    print("Test: Idea mode")
    print("=" * 60 + "\n")
    
    researcher = Researcher()
    query = "Best practices for LLM fine-tuning"
    
    ideas: List[Idea] = researcher.research(
        query=query,
        mode="idea",
        top_k=5,
        depth="deep",
    )
    
    assert isinstance(ideas, list), f"Expected list, got {type(ideas)}"
    print(f"Query: {query}")
    print(f"Results: {len(ideas)} ideas")
    
    for i, idea in enumerate(ideas, 1):
        assert isinstance(idea, Idea), f"Expected Idea, got {type(idea)}"
        assert idea.query == query, f"Expected query '{query}', got '{idea.query}'"
        print(f"\n{i}. Source: {idea.source}")
        print(f"   Content: {idea.content[:200]}...")
        print(f"   to_string():\n{idea.to_string()[:300]}...")


def test_implementation_mode():
    """Test implementation mode: returns List[Implementation]."""
    print("\n" + "=" * 60)
    print("Test: Implementation mode")
    print("=" * 60 + "\n")
    
    researcher = Researcher()
    query = "How to implement RAG with LangChain"
    
    impls: List[Implementation] = researcher.research(
        query=query,
        mode="implementation",
        top_k=3,
        depth="light",
    )
    
    assert isinstance(impls, list), f"Expected list, got {type(impls)}"
    print(f"Query: {query}")
    print(f"Results: {len(impls)} implementations")
    
    for i, impl in enumerate(impls, 1):
        assert isinstance(impl, Implementation), f"Expected Implementation, got {type(impl)}"
        assert impl.query == query, f"Expected query '{query}', got '{impl.query}'"
        print(f"\n{i}. Source: {impl.source}")
        print(f"   Content: {impl.content[:200]}...")
        print(f"   to_string():\n{impl.to_string()[:300]}...")


def test_study_mode():
    """Test study mode: returns ResearchReport."""
    print("\n" + "=" * 60)
    print("Test: Study mode")
    print("=" * 60 + "\n")
    
    researcher = Researcher()
    query = "How to improve DPO LLM post-training with unsloth"
    
    report: ResearchReport = researcher.research(
        query=query,
        mode="study",
        depth="deep",
    )
    
    assert isinstance(report, ResearchReport), f"Expected ResearchReport, got {type(report)}"
    assert report.query == query, f"Expected query '{query}', got '{report.query}'"
    
    print(f"Query: {query}")
    print(f"Content length: {len(report.content)} chars")
    print(f"\nReport preview:\n{report.content[:1000]}...")
    print(f"\nto_string() preview:\n{report.to_string()[:500]}...")


def main():
    """Run all tests."""
    # Run tests one at a time to avoid rate limits
    # Uncomment the test you want to run:
    
    #test_idea_mode()
    #test_implementation_mode()
    test_study_mode()


if __name__ == "__main__":
    main()
