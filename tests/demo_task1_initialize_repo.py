#!/usr/bin/env python3
"""
Task 1 & 2: Initialize Repo + Setup Directories - Demo Script

Demonstrates the full end-to-end flow:
1. Index wiki data into Knowledge Graph
2. Given a goal, search for relevant workflow
3. Clone the workflow repo as initial_repo
4. Add kapso_evaluation/ and kapso_datasets/ to the workspace
5. Show final directory structure

Usage:
    conda activate praxium_conda
    cd /home/ubuntu/kapso
    python tests/demo_task1_initialize_repo.py
"""

import os
import sys
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.kapso import Kapso

# =============================================================================
# Step 1: Index wiki data into Knowledge Graph
# =============================================================================
print("=" * 60)
print("Step 1: Index wiki data")
print("=" * 60)

wiki_dir = "data/wikis_llm_finetuning_test"
index_path = "data/indexes/demo_task1.index"

kapso = Kapso()
kapso.index_kg(
    wiki_dir=wiki_dir,
    save_to=index_path,
)
print(f"Indexed {wiki_dir} -> {index_path}")

# =============================================================================
# Step 2: Define goal and search for workflow
# =============================================================================
print("\n" + "=" * 60)
print("Step 2: Search for workflow based on goal")
print("=" * 60)

# This goal should match the PicoGPT workflow
goal = "Generate text using GPT-2 model with pure NumPy implementation"
print(f"Goal: {goal}")

kapso = Kapso(kg_index=index_path)

from src.knowledge.search.workflow_search import WorkflowRepoSearch

search = WorkflowRepoSearch(kg_search=kapso.knowledge_search)
result = search.search(goal, top_k=1)

print(f"\nWorkflow search result:")
if result.top_result:
    print(f"  Title: {result.top_result.title}")
    print(f"  GitHub URL: {result.top_result.github_url}")
    print(f"  Score: {result.top_result.score:.3f}")
else:
    print("  No workflow found!")
    sys.exit(1)

# =============================================================================
# Step 3: Clone workflow repo as initial_repo
# =============================================================================
print("\n" + "=" * 60)
print("Step 3: Clone workflow repo as initial_repo")
print("=" * 60)

github_url = result.top_result.github_url
initial_repo = kapso._clone_github_repo(github_url)
print(f"Cloned to: {initial_repo}")

print(f"\nInitial repo contents:")
for item in sorted(os.listdir(initial_repo)):
    print(f"  {item}")

# =============================================================================
# Step 4: Create test eval_dir and data_dir
# =============================================================================
print("\n" + "=" * 60)
print("Step 4: Create test eval_dir and data_dir")
print("=" * 60)

eval_dir = tempfile.mkdtemp(prefix="demo_eval_")
data_dir = tempfile.mkdtemp(prefix="demo_data_")

# Add evaluation script
with open(os.path.join(eval_dir, "evaluate.py"), "w") as f:
    f.write("""#!/usr/bin/env python3
\"\"\"Evaluation script for text generation quality.\"\"\"
import sys

def evaluate(generated_text: str) -> float:
    # Simple length-based score for demo
    score = min(len(generated_text) / 100, 1.0)
    return score

if __name__ == "__main__":
    # Read generated text from stdin or file
    text = "Sample generated text for evaluation"
    score = evaluate(text)
    print(f"SCORE: {score:.2f}")
""")

# Add test data
with open(os.path.join(data_dir, "prompts.txt"), "w") as f:
    f.write("""Alan Turing theorized that computers would one day become
The meaning of life is
In a galaxy far far away""")

print(f"Created eval_dir: {eval_dir}")
print(f"  - evaluate.py")
print(f"Created data_dir: {data_dir}")
print(f"  - prompts.txt")

# =============================================================================
# Step 5: Create workspace with initial_repo + kapso directories
# =============================================================================
print("\n" + "=" * 60)
print("Step 5: Create workspace with initial_repo + kapso directories")
print("=" * 60)

from src.execution.search_strategies.factory import SearchStrategyFactory
from src.execution.coding_agents.factory import CodingAgentFactory
from src.environment.handlers.generic import GenericProblemHandler
from src.core.llm import LLMBackend

# Create workspace directory (must be empty for seeding)
workspace_dir = tempfile.mkdtemp(prefix="demo_workspace_")
os.rmdir(workspace_dir)  # Remove so it can be seeded

handler = GenericProblemHandler(
    problem_description=goal,
    main_file="main.py",
    language="python",
)
coding_agent_config = CodingAgentFactory.build_config()
llm = LLMBackend()

strategy = SearchStrategyFactory.create(
    strategy_type="linear_search",
    problem_handler=handler,
    llm=llm,
    coding_agent_config=coding_agent_config,
    workspace_dir=workspace_dir,
    initial_repo=initial_repo,  # Use the cloned workflow repo
    eval_dir=eval_dir,
    data_dir=data_dir,
)

print(f"Workspace created at: {workspace_dir}")

# =============================================================================
# Step 6: Show final directory structure
# =============================================================================
print("\n" + "=" * 60)
print("Step 6: Final workspace directory structure")
print("=" * 60)

print(f"\n{workspace_dir}/")
for item in sorted(os.listdir(workspace_dir)):
    item_path = os.path.join(workspace_dir, item)
    if os.path.isdir(item_path):
        print(f"├── {item}/")
        # Show contents of kapso directories
        if item.startswith("kapso_") or item == ".kapso":
            for subitem in sorted(os.listdir(item_path)):
                print(f"│   └── {subitem}")
    else:
        print(f"├── {item}")

# =============================================================================
# Step 7: Verify git tracking
# =============================================================================
print("\n" + "=" * 60)
print("Step 7: Verify git tracking")
print("=" * 60)

repo = strategy.workspace.repo
tracked_files = repo.git.ls_files().split('\n')

print(f"\nTracked kapso files:")
for f in sorted(tracked_files):
    if 'kapso_' in f or '.kapso' in f:
        print(f"  {f}")

print(f"\nRecent commits:")
commits = list(repo.iter_commits('HEAD', max_count=5))
for c in commits:
    print(f"  - {c.message.strip()}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Demo complete! Directories preserved for inspection:")
print("=" * 60)
print(f"\n  Workspace (contains workflow repo + kapso dirs):")
print(f"    {workspace_dir}")
print(f"\n  Original eval_dir:")
print(f"    {eval_dir}")
print(f"\n  Original data_dir:")
print(f"    {data_dir}")
print(f"\n  KG index:")
print(f"    {index_path}")
print()
print("The workspace contains:")
print("  - All files from the workflow repo (gpt2.py, encoder.py, etc.)")
print("  - kapso_evaluation/ with your evaluation scripts")
print("  - kapso_datasets/ with your data files")
print("  - .kapso/repo_memory.json for agent context")
