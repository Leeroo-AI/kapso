# Test: Learn from Unsloth Repository
#
# Simple script to learn from https://github.com/unslothai/unsloth
# and save wiki pages to data/wiki_unsloth
#
# Usage:
#   python tests/test_learn_unsloth.py
#
# Required environment variables:
#   - AWS_BEARER_TOKEN_BEDROCK or AWS credentials for Bedrock
#   - AWS_REGION (defaults to us-east-1)

from dotenv import load_dotenv
load_dotenv()

from kapso import Kapso, Source

# Initialize Kapso (no KG index needed for learning)
kapso = Kapso()

# Learn from unsloth repo
# - skip_merge=True: Only extract wiki pages, don't merge into Neo4j/Weaviate
# - wiki_dir: Where to save the extracted wiki pages
result = kapso.learn(
    Source.Repo("https://github.com/unslothai/unsloth"),
    wiki_dir="data/wiki_unsloth",
    skip_merge=True,  # Skip merge step (avoids needing Neo4j/Weaviate)
)

# Print results
print(f"\n{'='*60}")
print("Learn Results:")
print(f"  Sources processed: {result.sources_processed}")
print(f"  Pages extracted: {result.total_pages_extracted}")
print(f"  Success: {result.success}")

if result.errors:
    print(f"\nErrors ({len(result.errors)}):")
    for error in result.errors:
        print(f"  - {error}")

if result.extracted_pages:
    print(f"\nExtracted Pages ({len(result.extracted_pages)}):")
    for page in result.extracted_pages[:10]:
        print(f"  - {page.page_title} ({page.page_type})")
    if len(result.extracted_pages) > 10:
        print(f"  ... and {len(result.extracted_pages) - 10} more")

print(f"\nWiki pages saved to: data/wiki_unsloth")
print(f"{'='*60}")
