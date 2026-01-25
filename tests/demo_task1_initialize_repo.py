#!/usr/bin/env python3
"""
Task 1: Initialize Repo - Demo Script

Demonstrates the initial_repo resolution logic:
1. Real KG indexing from wiki data
2. Real workflow search with OpenAI embeddings
3. Real GitHub repo cloning

Usage:
    conda activate praxium_conda
    cd /home/ubuntu/kapso
    python tests/demo_task1_initialize_repo.py
"""

import os
import sys

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
# Step 2: Load KG and search for workflow
# =============================================================================
print("\n" + "=" * 60)
print("Step 2: Search for workflow repo")
print("=" * 60)

kapso = Kapso(kg_index=index_path)

from src.knowledge.search.workflow_search import WorkflowRepoSearch

search = WorkflowRepoSearch(kg_search=kapso.knowledge_search)
result = search.search("text generation with GPT", top_k=3)

print(f"\nSearch results for 'text generation with GPT':")
for item in result.items:
    print(f"  - {item.title} (score={item.score:.3f})")
    print(f"    GitHub: {item.github_url}")

# =============================================================================
# Step 3: Get GitHub URL from search result
# =============================================================================
print("\n" + "=" * 60)
print("Step 3: Get GitHub URL from search result")
print("=" * 60)

# Use top_result.github_url directly
github_url = result.top_result.github_url if result.top_result else None
print(f"GitHub URL from top result: {github_url}")

# =============================================================================
# Step 4: Clone the repo
# =============================================================================
print("\n" + "=" * 60)
print("Step 4: Clone GitHub repo")
print("=" * 60)

cloned_path = kapso._clone_github_repo("https://github.com/jaymody/picoGPT")
print(f"Cloned to: {cloned_path}")

# Verify contents
import os
if cloned_path and os.path.exists(cloned_path):
    files = os.listdir(cloned_path)
    print(f"Contents: {files[:10]}...")
    
    # Cleanup
    import shutil
    shutil.rmtree(cloned_path, ignore_errors=True)
    print("Cleaned up cloned repo")

# =============================================================================
# Step 5: Full resolve_initial_repo flow
# =============================================================================
print("\n" + "=" * 60)
print("Step 5: Full _resolve_initial_repo flow")
print("=" * 60)

# With None -> triggers workflow search -> clones repo
result = kapso._resolve_initial_repo(
    initial_repo=None,
    goal="generate text using GPT-2 with NumPy"
)
print(f"Resolved repo path: {result}")

if result and os.path.exists(result):
    print(f"Repo exists: {os.path.exists(result)}")
    shutil.rmtree(result, ignore_errors=True)
    print("Cleaned up")

# Cleanup index
Path(index_path).unlink(missing_ok=True)

print("\n" + "=" * 60)
print("Demo complete!")
print("=" * 60)
