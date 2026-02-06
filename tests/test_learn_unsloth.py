# Test: Learn from Multiple Repositories
#
# Loops over repos from repos.json and learns from each one.
# Wiki pages are saved to data/wikis
#
# Usage:
#   python tests/test_learn_unsloth.py
#
# Required environment variables:
#   - AWS_BEARER_TOKEN_BEDROCK or AWS credentials for Bedrock
#   - AWS_REGION (defaults to us-east-1)

import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from kapso import Kapso, Source

# Load repos from repos.json
repos_file = Path(__file__).parent.parent / "repos.json"
with open(repos_file) as f:
    repos = json.load(f)

print(f"Found {len(repos)} repositories to process")
print(f"{'='*60}\n")

# Initialize Kapso once (index is created/auto-detected per wiki_dir)
kapso = Kapso()

# Track overall stats
total_success = 0
total_failed = 0
failed_repos = []

# Process each repo
for i, repo_url in enumerate(repos, 1):
    print(f"\n[{i}/{len(repos)}] Processing: {repo_url}")
    print("-" * 60)
    
    try:
        result = kapso.learn(
            Source.Repo(repo_url),
            wiki_dir="data/wikis",
            github_org="leeroopedia",
            is_private=False,
        )
        
        # Print results for this repo
        print(f"  Sources processed: {result.sources_processed}")
        print(f"  Pages extracted: {result.total_pages_extracted}")
        print(f"  Success: {result.success}")
        
        if result.errors:
            print(f"  Errors ({len(result.errors)}):")
            for error in result.errors[:3]:
                print(f"    - {error}")
            if len(result.errors) > 3:
                print(f"    ... and {len(result.errors) - 3} more")
        
        if result.success:
            total_success += 1
        else:
            total_failed += 1
            failed_repos.append(repo_url)
            
    except Exception as e:
        print(f"  ERROR: {e}")
        total_failed += 1
        failed_repos.append(repo_url)

# Print final summary
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
print(f"  Total repos: {len(repos)}")
print(f"  Successful: {total_success}")
print(f"  Failed: {total_failed}")

if failed_repos:
    print(f"\nFailed repos:")
    for repo in failed_repos:
        print(f"  - {repo}")

print(f"\nWiki pages saved to: data/wikis")
print(f"{'='*60}")
