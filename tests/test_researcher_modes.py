# Test Researcher in different modes with light depth.
#
# Usage:
#   python -m tests.test_researcher_modes
#
# This script calls the Researcher class in all three modes (idea, implementation, both)
# for topics about LLM post-training.

from src.knowledge.researcher import Researcher, ResearchMode

def main():
    researcher = Researcher()
    
    topic = "LLM post-training techniques: RLHF, DPO, and instruction tuning"
    
    modes: list[ResearchMode] = ["idea", "implementation", "both"]
    
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Mode: {mode}")
        print(f"{'='*60}\n")
        
        findings = researcher.research(
            objective=topic,
            mode=mode,
            depth="light",
        )
        
        # Print the report
        print(findings.source.report_markdown)
        
        # Print parsed repos and ideas
        print(f"\n--- Parsed Repos ({len(findings.repos())}) ---")
        for repo in findings.repos(top_k=5):
            print(f"  - {repo.url}")
        
        print(f"\n--- Parsed Ideas ({len(findings.ideas())}) ---")
        for idea in findings.ideas(top_k=3):
            print(f"  - {idea.title}")


if __name__ == "__main__":
    main()
