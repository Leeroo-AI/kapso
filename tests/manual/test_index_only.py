# Test: Index data/wikis only (no search, no staging)
#
# Simple script that indexes all wiki pages in data/wikis/
# into Weaviate (embeddings) and Neo4j (graph).
#
# Usage:
#   conda activate kapso_conda
#   python tests/test_index_only.py
#
# Required:
#   - Neo4j and Weaviate running (via infrastructure docker-compose)
#   - OPENAI_API_KEY set in .env
#   - Wiki pages in data/wikis/{workflows,principles,...}/*.md

from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from kapso import Kapso

# --- Config ---
wiki_dir = Path("data/wikis")
index_path = wiki_dir / ".index"

# --- Verify wiki directory exists ---
if not wiki_dir.exists():
    print(f"ERROR: {wiki_dir} not found.")
    exit(1)

# --- Index ---
print(f"Indexing wiki pages from {wiki_dir} ...")
print("=" * 60)

kapso = Kapso()
result = kapso.index_kg(
    wiki_dir=str(wiki_dir),
    save_to=str(index_path),
    force=True,
)

print(f"\nIndex file: {result}")
print("Done.")
