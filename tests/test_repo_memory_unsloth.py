"""
Test RepoMemoryManager with the real Unsloth repository.

Usage:
    cd /home/ubuntu/praxium
    conda activate praxium_conda
    python tests/test_repo_memory_unsloth.py
"""

import tempfile
from pathlib import Path

import git
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()

from src.core.llm import LLMBackend
from src.repo_memory import RepoMemoryManager
from src.repo_memory.builders import build_repo_map, validate_evidence


def main():
    # Clone unsloth repo (shallow for speed)
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_dir = Path(tmp_dir) / "unsloth"
        
        print("Cloning unsloth repo (shallow)...")
        git.Repo.clone_from(
            "https://github.com/unslothai/unsloth.git",
            repo_dir,
            depth=1,
        )
        print(f"Cloned to {repo_dir}")
        
        # Build repo map (deterministic, no LLM)
        print("\nBuilding repo map...")
        repo_map = build_repo_map(str(repo_dir))
        print(f"File count: {repo_map['file_count']}")
        print(f"Languages: {list(repo_map['languages_by_extension'].keys())}")
        
        # Bootstrap memory with LLM
        print("\nBootstrapping repo memory (LLM call)...")
        llm = LLMBackend()
        RepoMemoryManager.bootstrap_baseline_model(
            repo_root=str(repo_dir),
            llm=llm,
            seed_repo_path="https://github.com/unslothai/unsloth",
        )
        
        # Load and print results
        doc = RepoMemoryManager.load_from_worktree(str(repo_dir))
        print(f"\nDoc: {doc}")
        
        if doc:
            repo_model = doc.get("repo_model", {})
            quality = doc.get("quality", {})
            
            print(f"\n=== Results ===")
            print(f"Evidence OK: {quality.get('evidence_ok')}")
            print(f"Claim count: {quality.get('claim_count')}")
            print(f"Summary: {repo_model.get('summary', '')[:500]}")
            
            # Render brief
            brief = RepoMemoryManager.render_brief(doc)
            print(f"\n=== Brief ===\n{brief[:1500]}")


if __name__ == "__main__":
    main()
