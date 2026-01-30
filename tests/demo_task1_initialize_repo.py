#!/usr/bin/env python3
"""
Full evolve() Demo - End-to-end test of the new design.

1. Index wiki data into Knowledge Graph
2. Run evolve() with a goal that matches PicoGPT workflow
3. System finds workflow, clones repo, runs developer agent loop
4. Check logs and output

Usage:
    conda activate praxium_conda
    cd /home/ubuntu/kapso
    python tests/demo_task1_initialize_repo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from kapso.kapso import Kapso

# =============================================================================
# Step 1: Index wiki data
# =============================================================================
print("=" * 60)
print("Step 1: Index wiki data")
print("=" * 60)

wiki_dir = "data/wikis_llm_finetuning_test"
index_path = "data/indexes/demo_evolve.index"

kapso = Kapso()
kapso.index_kg(wiki_dir=wiki_dir, save_to=index_path)
print(f"Indexed {wiki_dir} -> {index_path}")

# =============================================================================
# Step 2: Run evolve() with goal matching PicoGPT
# =============================================================================
print("\n" + "=" * 60)
print("Step 2: Run evolve()")
print("=" * 60)

# Goal similar to PicoGPT workflow overview
goal = """
Generate text using a minimal GPT-2 implementation in pure NumPy.
The solution should:
1. Load pre-trained GPT-2 weights
2. Tokenize input text using BPE
3. Run transformer forward pass
4. Generate text using greedy decoding

Evaluation: Generate at least 40 tokens from the prompt "Alan Turing theorized that computers would one day become".
Success metric: Output should be coherent English text with perplexity < 50.
"""

kapso = Kapso(kg_index=index_path)

solution = kapso.evolve(
    goal=goal,
    max_iterations=5,  # Single iteration for demo
    mode="MINIMAL_TREE"
)

# =============================================================================
# Step 3: Check results
# =============================================================================
print("\n" + "=" * 60)
print("Step 3: Results")
print("=" * 60)

print(f"\nGoal achieved: {solution.succeeded}")
print(f"Final score: {solution.final_score}")
print(f"Stopped reason: {solution.metadata.get('stopped_reason', 'N/A')}")
print(f"Iterations: {solution.metadata.get('iterations', 'N/A')}")
print(f"Cost: {solution.metadata.get('cost', 'N/A')}")
print(f"\nCode path: {solution.code_path}")

# Show workspace structure
print(f"\nWorkspace contents:")
if os.path.exists(solution.code_path):
    for item in sorted(os.listdir(solution.code_path))[:15]:
        print(f"  {item}")

print("\n" + "=" * 60)
print("Demo complete!")
print("=" * 60)
