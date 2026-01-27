# Test Researcher in different modes.
#
# Usage:
#   conda activate praxium_conda
#   python -m tests.test_researcher_modes
#
# This script tests the Researcher class with the new API:
# - Default: all three modes (idea, implementation, freeform)
# - Single mode
# - Multiple modes

from dotenv import load_dotenv

from src.knowledge.researcher import Researcher, ResearchMode

load_dotenv()


def test_default_all_modes():
    """Test default behavior: runs all three modes."""
    print("\n" + "=" * 60)
    print("Test: Default (all three modes)")
    print("=" * 60 + "\n")
    
    researcher = Researcher()
    topic = "How to implement RAG with LangChain"
    
    findings = researcher.research(
        query=topic,
        top_k=3,
        depth="light",
    )
    
    print(f"Query: {findings.query}")
    print(f"Modes run: {findings.modes}")
    print(f"Top K: {findings.top_k}")
    
    # Ideas
    print(f"\n--- Ideas ({len(findings.ideas)}) ---")
    for i, idea in enumerate(findings.ideas, 1):
        print(f"{i}. {idea.content[:100]}...")
        print(f"   Source: {idea.source}")
    
    # Implementations
    print(f"\n--- Implementations ({len(findings.implementations)}) ---")
    for i, impl in enumerate(findings.implementations, 1):
        print(f"{i}. {impl.content[:150]}...")
        print(f"   Source: {impl.source}")
    
    # Report
    print(f"\n--- Report ---")
    if findings.report:
        print(findings.report.content[:500] + "...")
    else:
        print("No report")


def test_single_mode():
    """Test single mode: idea only."""
    print("\n" + "=" * 60)
    print("Test: Single mode (idea)")
    print("=" * 60 + "\n")
    
    researcher = Researcher()
    topic = "Best practices for LLM fine-tuning"
    
    findings = researcher.research(
        query=topic,
        mode="idea",
        top_k=5,
        depth="deep",
    )
    
    print(f"Query: {findings.query}")
    print(f"Modes run: {findings.modes}")
    
    print(f"\n--- Ideas ({len(findings.ideas)}) ---")
    for i, idea in enumerate(findings.ideas, 1):
        print(f"{i}. {idea.content}")
        print(f"   Source: {idea.source}")
    
    # Should be empty
    print(f"\nImplementations: {len(findings.implementations)} (should be 0)")
    print(f"Report: {findings.report} (should be None)")


def test_multiple_modes():
    """Test multiple modes: idea + implementation."""
    print("\n" + "=" * 60)
    print("Test: Multiple modes (idea + implementation)")
    print("=" * 60 + "\n")
    
    researcher = Researcher()
    topic = "How to improve DPO LLM post-training with unsloth and make it faster"
    
    findings = researcher.research(
        query=topic,
        mode=["study"],
        depth="deep",
    )
    
    print(f"Query: {findings.query}")
    print(f"Modes run: {findings.modes}")
    
    #print(f"\n--- Ideas ({len(findings.ideas)}) ---")
    #for i, idea in enumerate(findings.ideas, 1):
    #    print(f"{i}. {idea.content}")
    
    print(f"\n--- Implementations ({len(findings.implementations)}) ---")
    for i, impl in enumerate(findings.implementations, 1):
        print(f"{i}. {impl.content}")
    
    # Should be None
    print(f"\nReport: {findings.report} (should be None)")


def main():
    """Run all tests."""
    # Run tests one at a time to avoid rate limits
    #test_single_mode()
    # Uncomment to run more tests:
    test_multiple_modes()
    #test_default_all_modes()


if __name__ == "__main__":
    main()
