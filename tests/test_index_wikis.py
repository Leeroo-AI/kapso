# Test: Publish staged pages and index wiki
#
# Publishes pages from a staging directory into data/wikis/,
# then runs index_kg to create the .index metadata file.
#
# Usage:
#   conda activate kapso_conda
#   python tests/test_index_wikis.py
#
# Required:
#   - Neo4j and Weaviate running (via infrastructure docker-compose)
#   - Staged pages in data/wikis/_staging/

import shutil
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# --- Config ---
staging_dir = Path("data/wikis/_staging/Openclaw_Openclaw/51d6d81e4377")
wiki_dir = Path("data/wikis")
index_path = wiki_dir / ".index"
page_folders = ["workflows", "principles", "implementations", "environments", "heuristics"]

# --- Step 1: Publish from staging to wiki_dir ---
print("Step 1: Publishing staged pages to data/wikis/")
print("=" * 60)

total_published = 0
for folder in page_folders:
    src = staging_dir / folder
    dst = wiki_dir / folder
    if not src.exists():
        continue

    dst.mkdir(parents=True, exist_ok=True)
    pages = list(src.glob("*.md"))
    for page in pages:
        shutil.copy2(page, dst / page.name)
    total_published += len(pages)
    print(f"  {folder}: {len(pages)} pages")

print(f"\nPublished {total_published} pages from staging.")

if total_published == 0:
    print("ERROR: No pages found in staging. Nothing to index.")
    exit(1)

# --- Step 2: Index wiki pages ---
print(f"\nStep 2: Indexing wiki pages -> {index_path}")
print("=" * 60)

from kapso import Kapso

kapso = Kapso()
result = kapso.index_kg(
    wiki_dir=str(wiki_dir),
    save_to=str(index_path),
    force=True,
)

print(f"\nIndex file created: {result}")

# --- Step 3: Verify ---
print(f"\nStep 3: Verification")
print("=" * 60)

if index_path.exists():
    import json
    data = json.loads(index_path.read_text())
    print(f"  Version:  {data.get('version')}")
    print(f"  Backend:  {data.get('search_backend')}")
    print(f"  Pages:    {data.get('page_count')}")
    print(f"  Created:  {data.get('created_at')}")
    print("\nSUCCESS: .index file created and verified.")
else:
    print("\nFAILED: .index file was not created!")
    exit(1)
